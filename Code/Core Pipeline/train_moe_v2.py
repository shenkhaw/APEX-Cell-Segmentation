import json, time, math, random
from pathlib import Path
import numpy as np
import torch
from multi_task_data import make_loaders
from apex_moe_v2 import APEXMoE2, loss_fn

EPOCHS, BATCH, LR = 30, 8, 1e-3
SEED = 0
OUT = Path(r"C:\APEX\runs") / f"moe2_seed{SEED}"
OUT.mkdir(parents=True, exist_ok=True)

def lr_at(step, total, base, warm):
    if step < warm: return base*(step+1)/warm
    p = (step-warm)/max(1,total-warm)
    return base*0.5*(1+math.cos(math.pi*min(1.0,p)))

@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval(); per = {}; racc = []
    for x, y, t in loader:
        x, y, t = x.to(dev), y.to(dev), t.to(dev)
        with torch.autocast("cuda", enabled=(dev=="cuda")):
            out, logits = model(x)
        p = (torch.sigmoid(out.float()) > .5).float()
        d = ((2*(p*y).sum((1,2,3))+1)/(p.sum((1,2,3))+y.sum((1,2,3))+1)).cpu().numpy()
        for tt, dd in zip(t.tolist(), d): per.setdefault(tt, []).append(float(dd))
        racc.append((logits.argmax(1) == t).float().mean().item())
    means = {k: float(np.mean(v)) for k, v in per.items()}
    return means, float(np.mean(list(means.values()))), float(np.mean(racc))

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    
    tl, vl, el, cnt = make_loaders(BATCH)
    print("train per-task:", cnt)
    json.dump(dict(epochs=EPOCHS, batch=BATCH, lr=LR, seed=SEED, tasks=cnt), open(OUT/"config.json","w"), indent=2)

    m = APEXMoE2(k=2).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev=="cuda"))
    steps, total, gstep, best, log = len(tl), len(tl)*EPOCHS, 0, 0.0, []

    for ep in range(1, EPOCHS+1):
        m.train(); tot, t0 = 0.0, time.time()
        for x, y, t in tl:
            opt.param_groups[0]["lr"] = lr_at(gstep, total, LR, steps)
            x, y, t = x.to(dev), y.to(dev), t.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=(dev=="cuda")):
                out, logits = m(x); loss = loss_fn(out, y, logits, t)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += loss.item()*x.size(0); gstep += 1
        vper, vmac, racc = evaluate(m, vl, dev)
        row = dict(epoch=ep, train_loss=tot/sum(cnt), val_dice=vmac, router_acc=racc,
                   **{f"t{k}": round(v,4) for k,v in vper.items()})
        log.append(row); json.dump(log, open(OUT/"log.json","w"), indent=2)
        print(f"ep {ep:2d} | loss {row['train_loss']:.4f} | val {vmac:.4f} | racc {racc:.3f} | "
              + " ".join(f"t{k}:{v:.3f}" for k,v in vper.items()) + f" | {time.time()-t0:.0f}s")
        if vmac > best:
            best = vmac; torch.save(m.state_dict(), OUT/"best.pt"); print(f"   saved best ({best:.4f})")
        if ep % 5 == 0: torch.save(m.state_dict(), OUT/f"ep{ep:02d}.pt")

    m.load_state_dict(torch.load(OUT/"best.pt", map_location=dev))
    tper, tmac, tracc = evaluate(m, el, dev)
    print(f"\nFINAL TEST: macro {tmac:.4f} | router_acc {tracc:.3f}")
    print("  " + " ".join(f"t{k}:{v:.3f}" for k,v in tper.items()))
    json.dump(dict(val_best=best, test_macro=tmac, test_per_task=tper, router_acc=tracc),
              open(OUT/"final.json","w"), indent=2)

if __name__ == "__main__":
    main()
