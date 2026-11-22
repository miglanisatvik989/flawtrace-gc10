"""Verify the A0 and A1 architectures with synthetic tensors only."""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from evaluate import ROOT, UPSTREAM
from train import check_amp, initialize_cuda

MODELS = {
    "A0": (UPSTREAM / "ultralytics/cfg/models/age_yolo.yaml", (8, 16, 32), (64, 128, 256)),
    "A1": (ROOT / "configs/models/age_yolo_p2.yaml", (4, 8, 16, 32), (16, 64, 128, 256)),
}


def check_model(path, strides, channels, device):
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.tal import make_anchors
    from ultralytics.utils.torch_utils import get_flops

    torch.manual_seed(42)
    wrapper = YOLO(str(path), task="detect", verbose=False)
    assert wrapper.ckpt is None
    model = wrapper.model.to(device).eval()
    model.args = get_cfg()
    head = model.model[-1]
    assert type(head).__name__ == "SIFHead" and head.nc == 10 and head.reg_max == 16
    assert tuple(model.stride.tolist()) == strides and len(head.conv) == len(head.scale) == head.nl == len(strides)
    assert tuple(block[0].conv.in_channels for block in head.conv) == channels
    assert all(block[0].conv.out_channels == 64 for block in head.conv)
    assert head.share_conv[0].conv.groups == 64
    assert all(scale.scale.requires_grad and scale.scale.item() == 1 for scale in head.scale)
    assert all(layer.num_channels % layer.num_groups == 0 for layer in head.modules()
               if isinstance(layer, torch.nn.GroupNorm))
    assert torch.allclose(head.cv2.bias, torch.ones_like(head.cv2.bias))
    assert torch.allclose(head.cv3.bias, torch.full_like(head.cv3.bias, math.log(5 / 10 / (640 / 16) ** 2)))
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
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        check_amp(model, device)

    model.eval()
    parameters = sum(p.numel() for p in model.parameters())
    gflops = get_flops(model, imgsz=640)
    assert gflops > 0
    model.fuse(verbose=False)
    return {"strides": strides, "head_input_channels": channels, "parameters": parameters,
            "fused_parameters": sum(p.numel() for p in model.parameters()), "gflops_thop": gflops,
            "forwards": forwards, "loss_box_cls_dfl": components.detach().cpu().tolist(),
            "foreground_by_level": assigned,
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
        assert Path(ultralytics.__file__).resolve().is_relative_to(UPSTREAM)
        for name, (path, strides, channels) in MODELS.items():
            print(f"Checking {name} on {device}", flush=True)
            result = check_model(path, strides, channels, device)
            print(json.dumps({"model": name, **result}), flush=True)


if __name__ == "__main__":
    main()
