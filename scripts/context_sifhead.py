"""Regression-only context for the pinned AGE-YOLO SIFHead."""

from torch import nn

from ultralytics.nn import tasks
from ultralytics.nn.modules.age_yolo_modules import Conv_GN, SIFHead


class RegressionContext(nn.Module):
    """Refine each level with the same residual block."""

    def __init__(self, channels):
        super().__init__()
        self.depthwise = Conv_GN(channels, channels, 3, g=channels, d=2)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.pointwise.weight)

    def forward(self, features):
        return features + self.pointwise(self.depthwise(features))


class ContextRegression(nn.Sequential):
    """Place the residual before the native box predictor."""

    @property
    def bias(self):
        # SIFHead.bias_init initializes the original predictor directly.
        return self[1].bias


class ContextSIFHead(SIFHead):
    def __init__(self, nc, hidc, ch):
        super().__init__(nc, hidc, ch)
        self.cv2 = ContextRegression(RegressionContext(hidc), self.cv2)


def register_context_sifhead():
    """Expose the head to the pinned parser and its detection dispatch."""
    if ContextSIFHead not in tasks.DETECT_CLASS:
        tasks.ContextSIFHead = ContextSIFHead
        tasks.DETECT_CLASS += (ContextSIFHead,)
