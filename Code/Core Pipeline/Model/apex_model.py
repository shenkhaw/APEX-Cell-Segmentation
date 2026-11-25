import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

class SCSE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc = nn.Sequential(nn.Conv2d(c, c//r, 1), nn.ReLU(True), nn.Conv2d(c//r, c, 1))
        self.sp = nn.Conv2d(c, 1, 1)
    def forward(self, x):
        return torch.sigmoid(self.fc(F.adaptive_avg_pool2d(x, 1))) * x + torch.sigmoid(self.sp(x)) * x

def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))

class APEX(nn.Module):
    def __init__(self, scse=False):
        super().__init__()
        r = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool, self.l1, self.l2, self.l3, self.l4 = r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4
        self.scse = scse
        if scse: self.gates = nn.ModuleList([SCSE(c) for c in (64, 64, 128, 256)])
        self.up4, self.up3 = block(768, 256), block(384, 128)
        self.up2, self.up1, self.up0 = block(192, 64), block(128, 32), block(32, 16)
        self.head = nn.Conv2d(16, 1, 1)
    def _up(self, x, s, layer):
        x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
        return layer(torch.cat([x, s], 1))
    def forward(self, x):
        s0 = self.stem(x); s1 = self.l1(self.pool(s0))
        s2 = self.l2(s1); s3 = self.l3(s2); b = self.l4(s3)
        sk = [s0, s1, s2, s3]
        if self.scse: sk = [g(s) for g, s in zip(self.gates, sk)]
        d = self._up(b, sk[3], self.up4)
        d = self._up(d, sk[2], self.up3)
        d = self._up(d, sk[1], self.up2)
        d = self._up(d, sk[0], self.up1)
        d = self.up0(F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False))
        return self.head(d)

def loss_fn(logit, y):
    logit = logit.float()
    p = torch.sigmoid(logit)
    dice = 1 - (2*(p*y).sum((1,2,3)) + 1) / (p.sum((1,2,3)) + y.sum((1,2,3)) + 1)
    return F.binary_cross_entropy_with_logits(logit, y) + dice.mean()
