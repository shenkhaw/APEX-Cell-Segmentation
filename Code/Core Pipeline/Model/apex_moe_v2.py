import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

class Conv(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(ci,co,3,padding=1), nn.BatchNorm2d(co), nn.ReLU(True),
                               nn.Conv2d(co,co,3,padding=1), nn.BatchNorm2d(co), nn.ReLU(True))
    def forward(self, x): return self.b(x)

class Up(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = Conv(ci, co)
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None: x = torch.cat([x, skip], 1)
        return self.conv(x)

class ExpertBranch(nn.Module):
    """Private deep layers (layer3+layer4) + private decoder, per task."""
    def __init__(self, l3, l4):
        super().__init__()
        self.l3, self.l4 = l3, l4
        self.u4 = Up(512+256, 256); self.u3 = Up(256+128, 128); self.u2 = Up(128+64, 64)
        self.u1 = Up(64, 32); self.u0 = Up(32, 16); self.head = nn.Conv2d(16, 1, 1)
    def forward(self, s1, s2):
        s3 = self.l3(s2); s4 = self.l4(s3)
        x = self.u4(s4, s3); x = self.u3(x, s2); x = self.u2(x, s1)
        x = self.u1(x); x = self.u0(x)
        return self.head(x)

class Router(nn.Module):
    def __init__(self, cin=128, k=2):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(cin, 64), nn.ReLU(True), nn.Linear(64, k))
    def forward(self, s2):
        g = F.adaptive_avg_pool2d(s2, 1).flatten(1)
        return self.fc(g)

class APEXMoE2(nn.Module):
    """Shared LOW-level trunk (e0,e1,e2); private HIGH-level branches per task."""
    def __init__(self, k=2):
        super().__init__()
        r = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT)
        self.e0 = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.e1, self.e2 = r.layer1, r.layer2
        self.branches = nn.ModuleList(
            [ExpertBranch(copy.deepcopy(r.layer3), copy.deepcopy(r.layer4)) for _ in range(k)])
        self.router = Router(128, k)
    def forward(self, x):
        s0 = self.e0(x); s1 = self.e1(s0); s2 = self.e2(s1)
        logits = self.router(s2); w = torch.softmax(logits, 1)
        outs = torch.stack([b(s1, s2) for b in self.branches], 1)
        out = (outs * w[:, :, None, None, None]).sum(1)
        return out, logits

def loss_fn(out, y, logits, task, w_aux=0.1, w_bnd=0.0):
    with torch.no_grad():
        dil = F.max_pool2d(y, 3, 1, 1); ero = 1 - F.max_pool2d(1-y, 3, 1, 1)
        band = (dil - ero).clamp(0, 1)
    w = 1 + w_bnd*band
    bce = F.binary_cross_entropy_with_logits(out, y, weight=w)
    dice = 1 - (2*(torch.sigmoid(out)*y).sum()+1)/((torch.sigmoid(out)+y).sum()+1)
    aux = F.cross_entropy(logits, task)
    return bce + dice + w_aux*aux
