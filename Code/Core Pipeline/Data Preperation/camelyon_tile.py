from pathlib import Path
import numpy as np
from PIL import Image
import tifffile, zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG = Path(r"C:\APEX\camelyon16\images"); MSK = Path(r"C:\APEX\camelyon16\masks")
OUT = Path(r"C:\APEX\camelyon16_tiles")
(OUT/"images").mkdir(parents=True, exist_ok=True); (OUT/"masks").mkdir(parents=True, exist_ok=True)
TILE, STEP = 256, 4
N_POS, N_NEG, MAX_POS, MAX_NEG = 111, 40, 400, 300

fig, axes = plt.subplots(2, 2, figsize=(8,8)); fi = 0
npos = nneg = 0

for kind, cap, n in [("tumor", MAX_POS, N_POS), ("normal", MAX_NEG, N_NEG)]:
    slides = sorted(IMG.glob(f"{kind}_*.tif"))[:n]
    for sp in slides:
        ti = tifffile.TiffFile(sp)
        zi = zarr.open(ti.aszarr(level=0), mode="r")
        if kind == "tumor":
            tm = tifffile.TiffFile(MSK/(sp.stem+"_mask.tif"))
            zm = zarr.open(tm.aszarr(level=0), mode="r")
            coarse = tm.pages[6].asarray(); sel = (coarse == 2)
        else:
            zm = None
            coarse = ti.pages[6].asarray(); sel = (coarse.mean(axis=2) < 240)
        ry = zi.shape[0]/coarse.shape[0]; rx = zi.shape[1]/coarse.shape[1]
        ys, xs = np.nonzero(sel)
        if len(ys)==0: continue
        cnt = 0
        for yc in range(ys.min(), ys.max()+1, STEP):
            for xc in range(xs.min(), xs.max()+1, STEP):
                y0, x0 = int(yc*ry), int(xc*rx)
                if y0+TILE > zi.shape[0] or x0+TILE > zi.shape[1]: continue
                im = np.asarray(zi[y0:y0+TILE, x0:x0+TILE, :])
                if kind == "tumor":
                    mk = np.asarray(zm[y0:y0+TILE, x0:x0+TILE]); frac = (mk==2).mean()
                    if not (0.02 < frac < 0.98): continue
                    binm = (mk==2)
                else:
                    if (im.mean() >= 240): continue
                    frac, binm = 0.0, np.zeros((TILE,TILE), bool)
                tag = f"{sp.stem}_{y0}_{x0}"
                Image.fromarray(im).save(OUT/"images"/f"{tag}.png")
                Image.fromarray(binm.astype(np.uint8)*255).save(OUT/"masks"/f"{tag}.png")
                cnt += 1
                if kind=="tumor" and fi < 4:
                    r,c = divmod(fi,2); axes[r,c].imshow(im); axes[r,c].set_title(f"frac={frac:.2f}"); axes[r,c].axis("off"); fi += 1
                if cnt >= cap: break
            if cnt >= cap: break
        if kind=="tumor": npos += cnt
        else: nneg += cnt
        print(f"{sp.stem}: kept {cnt}")

print("pos tiles:", npos, "| neg tiles:", nneg)
plt.tight_layout(); plt.savefig(r"C:\APEX\scratch\camelyon_tiles_check.png", dpi=110)
print("saved C:\\APEX\\scratch\\camelyon_tiles_check.png")
