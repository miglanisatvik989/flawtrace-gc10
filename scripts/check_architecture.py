"""Verify the A0, A1 and A2 architectures with synthetic tensors only."""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch

from evaluate import ROOT, UPSTREAM
from train import backup_checkpoint, check_amp, initialize_cuda

MODELS = {
    "A0": (UPSTREAM / "ultralytics/cfg/models/age_yolo.yaml", (8, 16, 32), (64, 128, 256)),
    "A1": (ROOT / "configs/models/age_yolo_p2.yaml", (4, 8, 16, 32), (16, 64, 128, 256)),
    "A2": (ROOT / "configs/models/age_yolo_p2_context.yaml", (4, 8, 16, 32), (16, 64, 128, 256)),
}


def check_recovery(model, optimizer, device):
    from ultralytics import YOLO
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.nn.autobackend import AutoBackend

    with TemporaryDirectory(prefix="flawtrace-checkpoint-") as temporary:
        run = Path(temporary)
        trainer = SimpleNamespace(
            epoch=0, fitness=0.5, best_fitness=0.5, ema=SimpleNamespace(ema=model, updates=1),
            optimizer=optimizer, args=model.args, metrics={}, read_results_csv=lambda: {"epoch": [1]},
            wdir=run, last=run / "last.pt", best=run / "best.pt", save_period=-1,
        )
        BaseTrainer.save_model(trainer)
        backup_checkpoint(trainer)
        assert trainer.last.read_bytes() == (run / "last.recovery.pt").read_bytes()
        recovered = YOLO(str(run / "last.recovery.pt"), task="detect")
        rebuilt = DetectionTrainer.get_model(SimpleNamespace(data={"nc": 10}), cfg=recovered.model.yaml,
                                             weights=recovered.model, verbose=False).to(device)
        restored_optimizer = BaseTrainer.build_optimizer(None, rebuilt, name="SGD", lr=.01, momentum=.937, decay=.0005)
        resume = SimpleNamespace(optimizer=restored_optimizer, resume=True, ema=None, args=model.args, epochs=300)
        BaseTrainer.resume_training(resume, recovered.ckpt)
        assert resume.start_epoch == 1 and resume.best_fitness == .5 and restored_optimizer.state
        assert all(torch.equal(value, rebuilt.state_dict()[key]) for key, value in model.state_dict().items())
        backend = AutoBackend(weights=YOLO(str(trainer.best)).model, device=device, fp16=False)
        with torch.inference_mode():
            decoded, raw = backend(torch.zeros((1, 3, 64, 96), device=device))
        assert decoded.shape == (1, 14, 510) and len(raw) == 4 and torch.isfinite(decoded).all()


def check_model(path, strides, channels, device):
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.tal import make_anchors
    from ultralytics.utils.torch_utils import get_flops
    from context_sifhead import ContextSIFHead, SIFHead

    torch.manual_seed(42)
    wrapper = YOLO(str(path), task="detect", verbose=False)
    assert wrapper.ckpt is None
    model = wrapper.model.to(device).eval()
    model.args = get_cfg()
    head = model.model[-1]
    is_context = isinstance(head, ContextSIFHead)
    assert type(head) is (ContextSIFHead if path == MODELS["A2"][0] else SIFHead)
    assert head.nc == 10 and head.reg_max == 16
    assert tuple(model.stride.tolist()) == strides and len(head.conv) == len(head.scale) == head.nl == len(strides)
    assert tuple(block[0].conv.in_channels for block in head.conv) == channels
    assert all(block[0].conv.out_channels == 64 for block in head.conv)
    assert head.share_conv[0].conv.groups == 64
    assert all(scale.scale.requires_grad and scale.scale.item() == 1 for scale in head.scale)
    assert all(layer.num_channels % layer.num_groups == 0 for layer in head.modules()
               if isinstance(layer, torch.nn.GroupNorm))
    assert torch.allclose(head.cv2.bias, torch.ones_like(head.cv2.bias))
    assert torch.allclose(head.cv3.bias, torch.full_like(head.cv3.bias, math.log(5 / 10 / (640 / 16) ** 2)))
    if is_context:
        assert type(head).forward is SIFHead.forward
        context, predictor = head.cv2
        assert context.depthwise.conv.dilation == (2, 2) and context.depthwise.conv.groups == 64
        features = torch.rand((1, 64, 8, 12), device=device)
        with torch.inference_mode():
            assert torch.equal(context(features), features)
            assert torch.equal(head.cv2(features), predictor(features))
            assert head.cv3(features).shape == (1, 10, 8, 12)
            assert head.cv2(features).shape == (1, 64, 8, 12)
    layers = {}
    handles = []
    for layer in model.model[:-1]:
        def record(module, inputs, output, index=layer.i):
            layers[index] = list(output.shape[1:])
        handles.append(layer.register_forward_hook(record))
    forwards = []
    with torch.inference_mode():
        for height, width in ((640, 640), (352, 672)):
            scores, raw = model(torch.rand((1, 3, height, width), device=device))
            shapes = [(1, 74, height // stride, width // stride) for stride in strides]
            locations = sum(h * w for _, _, h, w in shapes)
            assert [tuple(t.shape) for t in raw] == shapes
            assert tuple(scores.shape) == (1, 14, locations)
            assert all(torch.isfinite(t).all() for t in (scores, *raw))
            assert (scores[:, 2:4] > 0).all() and ((scores[:, 4:] >= 0) & (scores[:, 4:] <= 1)).all()
            anchors, anchor_strides = make_anchors(raw, model.stride, .5)
            assert anchors.shape == (locations, 2) and anchor_strides.shape == (locations, 1)
            assert tuple(anchor_strides.flatten().unique().tolist()) == strides
            assert torch.equal(anchor_strides.squeeze(), torch.tensor(strides, device=device).repeat_interleave(
                torch.tensor([h * w for _, _, h, w in shapes], device=device)))
            assert torch.equal(anchors, head.anchors.T) and torch.equal(anchor_strides, head.strides.T)
            assert torch.equal(anchors[0], torch.tensor([.5, .5], device=device))
            forwards.append({"input_hw": [height, width], "head_outputs": shapes,
                             "decoded_shape": list(scores.shape), "locations": locations,
                             "layer_outputs_chw": dict(layers)})
    for handle in handles:
        handle.remove()

    model.train()
    batch = {
        "img": torch.rand((1, 3, 640, 640), device=device),
        "batch_idx": torch.zeros(3, device=device),
        "cls": torch.tensor([[0.], [4.], [8.]], device=device),
        "bboxes": torch.tensor([[.2, .3, .025, .03], [.5, .5, .15, .10], [.7, .7, .28, .22]], device=device),
    }
    assigned = []

    def record_assignment(module, inputs, output):
        counts = [(640 // stride) ** 2 for stride in strides]
        assigned.extend(mask.sum().item() for mask in output[3].split(counts, dim=1))

    model.criterion = model.init_criterion()
    with model.criterion.assigner.register_forward_hook(record_assignment):
        loss, components = model(batch)
    assert sum(assigned) > 0
    assert torch.isfinite(loss).all() and torch.isfinite(components).all() and (components > 0).all()
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    if len(strides) == 4:
        assert assigned[0] > 0
        assert model.model[25].conv.weight.grad.abs().sum() > 0
    if is_context:
        from ultralytics.engine.trainer import BaseTrainer

        assert context.pointwise.weight.grad.abs().sum() > 0
        optimizer = BaseTrainer.build_optimizer(None, model, name="SGD", lr=.01, momentum=.937, decay=.0005)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model(batch)[0].backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
                   for p in context.parameters())
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        check_amp(model, device)

    model.eval()
    parameters = sum(p.numel() for p in model.parameters())
    gflops = get_flops(model, imgsz=640)
    assert gflops > 0
    if is_context:
        check_recovery(model, optimizer, device)
    model.fuse(verbose=False)
    return {"strides": strides, "head_input_channels": channels, "parameters": parameters,
            "fused_parameters": sum(p.numel() for p in model.parameters()), "gflops_thop": gflops,
            "forwards": forwards, "loss_box_cls_dfl": components.detach().cpu().tolist(),
            "foreground_by_level": assigned,
            **({"context_parameters": sum(p.numel() for p in context.parameters()),
                "checkpoint_recovery_passed": True} if is_context else {}),
            "finite_gradients": True, "amp_passed": True if device.type == "cuda" else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="CPU or an authorized indexed CUDA device.")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        device = initialize_cuda(args.device)
    elif device.type != "cpu":
        parser.error("Use cpu or an indexed CUDA device such as cuda:0.")
    torch.set_num_threads(2)
    sys.path.insert(0, str(UPSTREAM))
    sys.dont_write_bytecode = True
    with TemporaryDirectory(prefix="flawtrace-architecture-") as settings:
        os.environ.update(YOLO_CONFIG_DIR=settings, YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false")
        import ultralytics
        from context_sifhead import register_context_sifhead

        assert Path(ultralytics.__file__).resolve().is_relative_to(UPSTREAM)
        register_context_sifhead()
        for name, (path, strides, channels) in MODELS.items():
            print(f"Checking {name} on {device}", flush=True)
            result = check_model(path, strides, channels, device)
            print(json.dumps({"model": name, **result}), flush=True)


if __name__ == "__main__":
    main()
