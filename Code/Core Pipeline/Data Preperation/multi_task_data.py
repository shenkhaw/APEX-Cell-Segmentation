from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

ROOT_ADNP = Path(r"C:\APEX\ADNP-15\origin")
ROOT_CAM  = Path(r"C:\APEX\camelyon16_tiles")
PLAQUE_TRAIN = [1,2,3,4,7,8,11,12,14]; PLAQUE_VAL = [9,15]; PLAQUE_TEST = [5,10,13]
MEAN = torch.tensor([.485,.456,.406]).view(3,1,1)
STD  = torch.tensor([.229,.224,.225]).view(3,1,1)

def build_items():
    items = []
    # task 0: plaque (ADNP-15, IHC brain)
    for pat in sorted([p for p in ROOT_ADNP.iterdir() if p.is_dir()], key=lambda p: int(p.name)):
        pid = int(pat.name)
        split = "train" if pid in PLAQUE_TRAIN else ("val" if pid in PLAQUE_VAL else "test")
        for img in sorted((pat/"macenko").glob("*.png")):
            msk = pat/"masks"/img.name.replace("patch_", "mask_", 1)
            if msk.exists():
                items.append(dict(img=img, msk=msk, task=0, thr=127, split=split, size=256))
    # task 1: tumor (CAMELYON16 tiles, H&E lymph node)
    for img in sorted((ROOT_CAM/"images").glob("*.png")):
        msk = ROOT_CAM/"masks"/img.name
        if not msk.exists(): continue
        parts = img.stem.split("_"); n = int(parts[1])
        if parts[0] == "tumor":
            split = "train" if n <= 80 else ("val" if n <= 95 else "test")
        else:
            split = "train" if n <= 12 else ("val" if n <= 16 else "test")
        items.append(dict(img=img, msk=msk, task=1, thr=127, split=split, size=256))
    return items

class MultiTask(Dataset):
    def __init__(self, items, train=False):
        self.items, self.train = items, train
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        im = Image.open(it["img"]).convert("RGB"); mk = Image.open(it["msk"])
        im = np.array(im, np.float32)/255.
        mk = (np.array(mk) > it["thr"]).astype(np.float32)
        if self.train:
            k = random.randint(0,3)
            im, mk = np.rot90(im,k,(0,1)), np.rot90(mk,k,(0,1))
            if random.random() < .5: im, mk = im[:,::-1], mk[:,::-1]
        x = torch.from_numpy(np.ascontiguousarray(im.transpose(2,0,1)))
        y = torch.from_numpy(np.ascontiguousarray(mk))[None]
        return (x-MEAN)/STD, y, it["task"]

def make_loaders(batch=8):
    items = build_items()
    tr = [i for i in items if i["split"]=="train"]
    va = [i for i in items if i["split"]=="val"]
    te = [i for i in items if i["split"]=="test"]
    cnt = [sum(1 for i in tr if i["task"]==t) for t in range(2)]
    print("Dataset sizes -> Plaque:", cnt[0], "| Tumor:", cnt[1])
    
    # No more WeightedRandomSampler! Just pure, natural shuffling.
    return (DataLoader(MultiTask(tr, True), batch, shuffle=True, num_workers=0, pin_memory=False, drop_last=True),
            DataLoader(MultiTask(va), batch, shuffle=False, num_workers=0, pin_memory=False),
            DataLoader(MultiTask(te), batch, shuffle=False, num_workers=0, pin_memory=False), cnt)

if __name__ == "__main__":
    items = build_items()
    for s in ["train","val","test"]:
        c = [sum(1 for i in items if i["split"]==s and i["task"]==t) for t in range(2)]
        print(s, "| plaque", c[0], "| tumor", c[1])
