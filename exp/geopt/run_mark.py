"""The demo run is now one command: openmarkx run configs/demos/geopt.yaml. This script keeps the sweeps.
GPU: PRO for GeoPT on AirCraft. Calibrate the field mark, learn it into the weights, evaluate, save demo cases.

  python exp/geopt/run_mark.py --geopt_dir $B/GeoPT --data $MWX_RUNS/geopt/aircraft_npys \
      --ckpt $B/GeoPT/checkpoints/craft_geopt_8layers.pt --out $MWX_RUNS/geopt/mark
Steps (each skipped when its output exists, so a rerun continues):
  calibrate.json  the post-hoc mark (the "generation-time" rule) on the clean model's test outputs, for several eps:
                  detection z-scores and the relative L2 change; picks the smallest eps with every test case at
                  z > --z_target (clean outputs give z near 0)
  student.pt      the clean model fine-tuned to reproduce clean model + mark on training geometries under random
                  flight conditions (no simulation labels); resume.pt holds progress
  result.json     on the 50 test aircraft: TPR at 1% FPR (threshold from clean outputs), z-scores, relative L2 to the
                  CFD fields for clean / teacher / student, and the same after attacks (fine-tuning the student on
                  clean labels; the full mesh instead of the 30k training sample)
  demo.npz        a few test aircraft on the full mesh: points, ground truth, clean, student, pattern, z
"""
import argparse
import copy
import json
import os
import sys
import time
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fieldmark as fm


def craft_args(a):
    """The authors' scripts/finetune/GeoPT_craft.sh settings."""
    return SimpleNamespace(data_path=a.data, loader="AirCraft", task="GeoPT_finetune", dynamics="craft",
                           geotype="unstructured", space_dim=3, fun_dim=11, out_dim=6, normalize=1,
                           norm_type="UnitTransformer", model="Transolver", n_hidden=256, n_heads=8, n_layers=8,
                           mlp_ratio=2, slice_num=32, ntrain=100, ntest=50, batch_size=1, act="gelu", dropout=0.0,
                           checkpoint=0, shapelist=None)


class Sim:
    """The GeoPT model with the authors' input assembly: model(points, [x features, flight-condition field])."""

    def __init__(self, model, direction):
        self.model, self.direction = model, direction

    def __call__(self, pos, cond, model=None):
        v = self.direction(pos, cond)
        return (model or self.model)(pos[:, :, :3], torch.cat((pos, v), -1))      # normalized output units


def rel_l2(out, y):
    return ((out - y).flatten(1).norm(dim=1) / y.flatten(1).norm(dim=1)).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geopt_dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpt", required=True, help="the fine-tuned open model (the authors' craft script)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--key", type=int, default=20261020)
    ap.add_argument("--eps", type=float, default=None, help="default: chosen by calibration")
    ap.add_argument("--eps_grid", default="0.0025,0.005,0.01,0.02,0.04")
    ap.add_argument("--z_target", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="small: the student starts at the clean model and the target differs from it by a 1-5%% "
                         "mark, so Adam's normalized steps at 1e-4 move the weights far more than the mark needs")
    ap.add_argument("--trainable", default="all", help="all, or lastK (the last K Transolver blocks only)")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--attack_epochs", type=int, default=20)
    ap.add_argument("--n_wrong", type=int, default=200)
    ap.add_argument("--n_demo", type=int, default=4)
    ap.add_argument("--wavelength", default="0.25,0.75", help="the mark's band, in GeoPT's geometry units (length 5)")
    ap.add_argument("--basis_min_wavelength", type=float, default=2.0)
    ap.add_argument("--domain", default="output", choices=["output", "position", "normal"])
    ap.add_argument("--output_scale", type=float, default=1.0)
    ap.add_argument("--edit_last", type=int, default=1,
                    help="1: first set the last linear layer's change in closed form (ridge regression of eps * g on "
                         "its input features over training geometries), then fine-tune for --steps")
    ap.add_argument("--detect", default="reference", choices=["reference", "output"],
                    help="reference: the publisher subtracts its unmarked model's output for the same geometry and "
                         "flight condition, then tests the residual; output: the field alone")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fm.WAVELENGTH = tuple(map(float, a.wavelength.split(",")))
    fm.BASIS_MIN_WAVELENGTH = a.basis_min_wavelength
    fm.DOMAIN = a.domain
    fm.OUTPUT_SCALE = a.output_scale
    if a.domain == "output" and a.detect != "reference":
        raise SystemExit("an output-keyed mark is tested against the publisher's unmarked output: use --detect reference")
    sys.path.insert(0, a.geopt_dir)
    from data_provider.data_factory import get_data
    from exp.dynamics_config import get_direction
    from models.model_factory import get_model
    args = craft_args(a)
    dataset, train_loader, test_loader, args.shapelist = get_data(args)
    norm = dataset.y_normalizer                             # moved to CUDA by the loader
    clean = get_model(args).cuda()
    clean.load_state_dict(torch.load(a.ckpt, map_location="cuda")); clean.eval()
    for p in clean.parameters():
        p.requires_grad_(False)
    sim = Sim(clean, get_direction("craft"))
    train = [(p.cuda(), c.cuda(), y.cuda()) for p, _, c, y in train_loader.dataset]          # y normalized
    test = [(p.cuda(), c.cuda(), y.cuda()) for p, _, c, y in test_loader.dataset]            # y physical
    train = [(p[None], c[None], y[None]) for p, c, y in train]
    test = [(p[None], c[None], y[None]) for p, c, y in test]
    log = lambda d: (print(json.dumps(d), flush=True), open(f"{a.out}/log.jsonl", "a").write(json.dumps(d) + "\n"))

    @torch.no_grad()
    def outputs(model, data):
        return [sim(p, c, model) for p, c, _ in data]

    def z_of(data, outs, refs=None):
        """z per case. With --detect reference, refs are the unmarked model's outputs for the same inputs."""
        refs = refs if refs is not None else [None] * len(outs)
        return [fm.zscore(p, o - r if a.detect == "reference" else o, a.key, n_wrong=a.n_wrong, ref=r).item()
                for (p, _, _), o, r in zip(data, outs, refs)]

    # ---- 1. calibrate the generation-time mark on the clean model's test outputs ----
    cal_path = f"{a.out}/calibrate.json"
    clean_out = outputs(clean, test)
    if not os.path.exists(cal_path):
        cal = {"clean_z": z_of(test, clean_out) if a.detect == "output" else [0.0],
               "clean_rel_l2": sum(rel_l2(norm.decode(o), y)[0] for o, (_, _, y) in zip(clean_out, test)) / len(test)}
        for eps in map(float, a.eps_grid.split(",")):
            marked = [fm.mark(p, o, a.key, eps) for (p, _, _), o in zip(test, clean_out)]
            cal[str(eps)] = {"z": z_of(test, marked, clean_out), "rel_l2": sum(rel_l2(norm.decode(m), y)[0]
                                                                   for m, (_, _, y) in zip(marked, test)) / len(test)}
            log({"calibrate_eps": eps, "z_min": min(cal[str(eps)]["z"]), "rel_l2": cal[str(eps)]["rel_l2"]})
        ok = [float(e) for e in a.eps_grid.split(",") if min(cal[e]["z"]) > a.z_target]
        cal["chosen_eps"] = min(ok) if ok else float(a.eps_grid.split(",")[-1])
        json.dump(cal, open(cal_path, "w"), indent=1)
    cal = json.load(open(cal_path))
    eps = a.eps or cal["chosen_eps"]
    log({"eps": eps, "clean_z_max": max(cal["clean_z"]), "clean_rel_l2": cal["clean_rel_l2"]})

    # ---- 2. PRO: learn "clean model + mark" into a copy of the weights, on geometries alone ----
    stu_path, res_path = f"{a.out}/student.pt", f"{a.out}/resume.pt"
    student = copy.deepcopy(clean)
    if os.path.exists(stu_path):
        student.load_state_dict(torch.load(stu_path, map_location="cuda"))
    else:
        names = [n for n, _ in student.named_parameters()]
        if a.trainable.startswith("last"):
            k = int(a.trainable[4:]); n_blocks = 1 + max(int(n.split(".")[1]) for n in names if n.startswith("blocks."))
            keep = lambda n: n.startswith("blocks.") and int(n.split(".")[1]) >= n_blocks - k
        else:
            keep = lambda n: True
        params = []
        for n, p in student.named_parameters():
            p.requires_grad_(keep(n))
            if keep(n):
                params.append(p)
        log({"trainable_params": sum(p.numel() for p in params)})
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        import math
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s_: min(1.0, (s_ + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * s_ / max(a.steps, 1))))
        conds = torch.cat([c for _, c, _ in train])                               # [100, 1, 3]
        lo, hi = conds.amin(0), conds.amax(0)
        g = torch.Generator().manual_seed(0)
        start = 0
        if a.edit_last and not os.path.exists(res_path):    # closed-form change of the last linear layer
            head = student.blocks[-1].mlp2
            feats = {}
            hook = head.register_forward_hook(lambda mod, inp, out: feats.__setitem__("h", inp[0].detach()))
            d = head.in_features + 1
            HtH = torch.zeros(d, d, device="cuda", dtype=torch.float64); HtG = torch.zeros(d, head.out_features,
                                                                                         device="cuda", dtype=torch.float64)
            with torch.no_grad():
                for rep in range(3):
                    for p, c0, _ in train:
                        c = c0 if rep == 0 else (lo + (hi - lo) * torch.rand(lo.shape, generator=g).cuda())[None]
                        o = sim(p, c, student)
                        H = torch.cat([feats["h"][0], torch.ones_like(feats["h"][0][:, :1])], 1).double()
                        HtH += H.T @ H; HtG += H.T @ (fm.mark(p, o, a.key, eps) - o)[0].double()
                lam = 1e-3 * HtH.diagonal().mean()
                dW = torch.linalg.solve(HtH + lam * torch.eye(d, device="cuda", dtype=torch.float64), HtG).float()
                head.weight += dW[:-1].T; head.bias += dW[-1]
            hook.remove()
            log({"edit_last": True, "dW_rel": (dW[:-1].norm() / head.weight.norm()).item()})
        if os.path.exists(res_path):
            st = torch.load(res_path, map_location="cuda", weights_only=False)
            student.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
            g.set_state(st["gen"]); start = st["step"]
        student.train(); t0 = time.time()
        for step in range(start, a.steps):
            i = int(torch.randint(len(train), (1,), generator=g))
            p, _, _ = train[i]
            c = (lo + (hi - lo) * torch.rand(lo.shape, generator=g).cuda())[None]  # a random flight condition
            with torch.no_grad():
                target = fm.mark(p, sim(p, c, clean), a.key, eps)
            loss = (sim(p, c, student) - target).pow(2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            if (step + 1) % 100 == 0:
                log({"step": step + 1, "mse": loss.item(), "sec": round(time.time() - t0)})
            if (step + 1) % 500 == 0:
                torch.save({"model": student.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "gen": g.get_state(), "step": step + 1}, res_path + ".tmp")
                os.replace(res_path + ".tmp", res_path)
        torch.save(student.state_dict(), stu_path)
        if os.path.exists(res_path):
            os.remove(res_path)
    student.eval()

    # ---- 3. evaluate ----
    def report(name, outs, data=test, refs=None):
        z = z_of(data, outs, clean_out if refs is None else refs)
        r = {"z_median": sorted(z)[len(z) // 2], "z_min": min(z),
             "rel_l2": sum(rel_l2(norm.decode(o), y)[0] for o, (_, _, y) in zip(outs, data)) / len(data)}
        if name != "null":
            r["tpr_at_1fpr"] = sum(x > max(thr, a.z_target) for x in z) / len(z)
        return r, z

    # The null: outputs the mark never touched. For output detection, the clean model's own outputs. For reference
    # detection (clean residuals would be exactly zero), an unmarked derivative: the clean model fine-tuned for two
    # epochs on the simulation labels, i.e. what an independent adopter of the open model might release.
    if a.detect == "reference":
        other = copy.deepcopy(clean); other.train()
        for p_ in other.parameters():
            p_.requires_grad_(True)
        opt = torch.optim.AdamW(other.parameters(), lr=1e-4)
        for _ in range(2):
            for p, c, y in train:
                loss = (sim(p, c, other) - y).pow(2).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        other.eval()
    else:
        other = clean
    null_out = outputs(other, test)
    z_null = z_of(test, null_out, clean_out)
    with torch.no_grad():                                   # more null cases: other flight conditions
        gn = torch.Generator().manual_seed(1)
        for _ in range(3):
            for p, c, _ in test:
                cc = c * (0.9 + 0.2 * torch.rand(c.shape, generator=gn).cuda())
                z_null += z_of([(p, cc, None)], [sim(p, cc, other)], [sim(p, cc, clean)])
    thr = sorted(z_null)[int(0.99 * len(z_null)) - 1]
    res = {"eps": eps, "detect": a.detect, "wavelength": list(fm.WAVELENGTH),
           "basis_min_wavelength": fm.BASIS_MIN_WAVELENGTH, "domain": fm.DOMAIN, "output_scale": fm.OUTPUT_SCALE, "edit_last": a.edit_last, "steps": a.steps, "n_test": len(test),
           "n_null": len(z_null), "null_z_99pct": thr, "clean_rel_l2": cal["clean_rel_l2"]}
    res["null"], _ = report("null", null_out)
    teacher_out = [fm.mark(p, o, a.key, eps) for (p, _, _), o in zip(test, clean_out)]
    res["teacher"], _ = report("teacher", teacher_out)
    stu_out = outputs(student, test)
    res["student"], z_stu = report("student", stu_out)
    log({"student": res["student"], "null": res["null"]})

    # attack: the student fine-tuned on clean simulation labels (an adopter adapting the model to their data)
    att = copy.deepcopy(student); att.train()
    for p_ in att.parameters():
        p_.requires_grad_(True)
    opt = torch.optim.AdamW(att.parameters(), lr=1e-4)
    for ep in range(a.attack_epochs):
        for p, c, y in train:
            loss = (sim(p, c, att) - y).pow(2).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if ep + 1 in (2, 5, 10, a.attack_epochs):
            att.eval()
            res[f"student_finetuned_{ep + 1}ep"], _ = report("attack", outputs(att, test))
            log({f"attack_{ep + 1}ep": res[f"student_finetuned_{ep + 1}ep"]})
            att.train()
    json.dump(res, open(f"{a.out}/result.json", "w"), indent=1)

    # ---- 4. demo cases on the full mesh ----
    try:
        _, _, full_loader, _ = get_data(args, full_mesh=True)
        demo = {}
        with torch.no_grad():
            for i, (p, _, c, y) in enumerate(full_loader.dataset):
                if i >= a.n_demo:
                    break
                p, c = p.cuda()[None], c.cuda()[None]
                oc, os_ = sim(p, c, clean), sim(p, c, student)
                demo[f"case{i}"] = dict(points=p[0, :, :3].cpu(), normals=p[0, :, 4:7].cpu(), cond=c[0, 0].cpu(),
                                        truth=y.cpu(), clean=norm.decode(oc)[0].cpu(),
                                        student=norm.decode(os_)[0].cpu(),
                                        pattern=fm.pattern(p, a.key, 6)[0].cpu(),
                                        z_null=z_of([(p, c, None)], [sim(p, c, other)], [oc])[0],
                                        z_student=z_of([(p, c, None)], [os_], [oc])[0])
        res["full_mesh_demo_z"] = {k: [v["z_null"], v["z_student"]] for k, v in demo.items()}
        torch.save(demo, f"{a.out}/demo.pt")
    except Exception as e:                                  # the demo cases are a convenience; keep the numbers
        res["demo_error"] = repr(e)
    json.dump(res, open(f"{a.out}/result.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
