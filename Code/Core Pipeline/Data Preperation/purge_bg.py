from pathlib import Path
import numpy as np
from PIL import Image
SRC = Path(r"C:\APEX\camelyon16_tiles")
BG = SRC/"_background"
(BG/"images").mkdir(parents=True, exist_ok=True); (BG/"masks").mkdir(parents=True, exist_ok=True)
moved = 0
for p in sorted((SRC/"images").glob("normal_*.png")):
    a = np.array(Image.open(p).convert("RGB"), np.int16)
    sat = (a.max(2) - a.min(2)).mean()
    if sat < 8:
        m = SRC/"masks"/p.name
        p.rename(BG/"images"/p.name)
        if m.exists(): m.rename(BG/"masks"/p.name)
        moved += 1
print("moved", moved, "gray-background tiles out of dataset")
