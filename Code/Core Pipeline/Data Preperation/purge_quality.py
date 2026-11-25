from pathlib import Path
import numpy as np
from PIL import Image
SRC = Path(r"C:\APEX\camelyon16_tiles")
BAD = SRC/"_lowquality"
(BAD/"images").mkdir(parents=True, exist_ok=True); (BAD/"masks").mkdir(parents=True, exist_ok=True)

def sharpness(g):
    r = (g[1:-1,2:] + g[1:-1,:-2] + g[2:,1:-1] + g[:-2,1:-1] - 4*g[1:-1,1:-1])
    return r.var()

moved = 0
for p in sorted((SRC/"images").glob("*.png")):
    a = np.array(Image.open(p).convert("RGB"), np.float32)
    g = a.mean(2)
    mean, sat, sharp = g.mean(), (a.max(2)-a.min(2)).mean(), sharpness(g)
    if not (90 <= mean <= 240) or sat < 8 or sharp < 25:
        m = SRC/"masks"/p.name
        p.rename(BAD/"images"/p.name)
        if m.exists(): m.rename(BAD/"masks"/p.name)
        moved += 1
print("moved", moved, "low-quality tiles out of dataset")
