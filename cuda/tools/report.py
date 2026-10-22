"""Figures, tables and the history file from the result folders in cuda/results/.

Run from the repository root:  python cuda/tools/report.py
Reads  cuda/results/<case>_N<Nd>_<precision>/summary.json, forces.csv, fields.bin, frames.bin
Writes cuda/results/figures/*.png|gif, cuda/results/history.csv, cuda/results/tables.md,
       cuda/results/grid_study.json, cuda/results/poiseuille_summary.json
Every number in README.md is copied from tables.md, which names its source file per row.
"""
import csv
import glob
import json
import math
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import NullFormatter
from PIL import Image

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES = os.path.join(ROOT, "cuda", "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

# fixed categorical colours (one hue per quantity, never cycled)
C = {"cd": "#1f5fa8", "cl": "#c8552a", "dp": "#3a8f5c", "st": "#7a4fa3", "ref": "#555555", "guide": "#999999", "fp64": "#111111"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e3e3e3", "grid.linewidth": 0.6, "legend.frameon": False, "figure.dpi": 130})

# the demo's LUT, taken verbatim from cuda/lbm_cuda.cu
def load_lut():
    src = open(os.path.join(ROOT, "cuda", "lbm_cuda.cu")).read()
    body = src.split("__constant__ unsigned char LUT[256][3] = {")[1].split("};")[0]
    rows = re.findall(r"\{\s*(\d+),\s*(\d+),\s*(\d+)\}", body)
    assert len(rows) == 256
    return ListedColormap(np.array(rows, dtype=float) / 255.0)
LUT = load_lut()
CYL_RGB = (38 / 255, 38 / 255, 45 / 255)


def load(path):
    return json.load(open(path))


def runs(pattern):
    out = {}
    for d in sorted(glob.glob(os.path.join(RES, pattern))):
        p = os.path.join(d, "summary.json")
        if os.path.exists(p):
            s = load(p); s["_dir"] = os.path.relpath(d, ROOT).replace("\\", "/"); out[os.path.basename(d)] = s
    return out


def read_fields(d):
    j = load(os.path.join(d, "fields.json"))
    a = np.fromfile(os.path.join(d, "fields.bin"), dtype=np.float32)
    n = j["Nx"] * j["Ny"]
    f = {name: a[i * n:(i + 1) * n].reshape(j["Ny"], j["Nx"]) for i, name in enumerate(j["arrays"])}
    t = np.fromfile(os.path.join(d, "types.bin"), dtype=np.uint8).reshape(j["Ny"], j["Nx"])
    return j, f, t


def order(h, e):
    """Observed order between consecutive levels and over the whole set (least squares)."""
    h = np.asarray(h, float); e = np.asarray(e, float)
    pair = [math.log(e[i] / e[i + 1]) / math.log(h[i + 1] / h[i]) for i in range(len(h) - 1)]
    fit = -np.polyfit(np.log(h), np.log(e), 1)[0] if len(h) > 1 else float("nan")
    return pair, fit


# ---------------------------------------------------------------- Poiseuille
def poiseuille():
    rows = {}
    for prec in ("fp32", "fp64"):
        rr = runs(f"poiseuille_H*_{prec}")
        if not rr:
            continue
        H = sorted(int(re.search(r"_H(\d+)_", k).group(1)) for k in rr)
        err = [rr[f"poiseuille_H{h}_{prec}"]["poiseuille"]["l2_rel_error"] for h in H]
        pair, fit = order(H, err)
        rows[prec] = dict(H=H, err=err, order_pairs=pair, order_fit=fit,
                          files=[rr[f"poiseuille_H{h}_{prec}"]["_dir"] + "/summary.json" for h in H],
                          steps=[rr[f"poiseuille_H{h}_{prec}"]["poiseuille"]["steps"] for h in H],
                          last_change=[rr[f"poiseuille_H{h}_{prec}"]["poiseuille"]["last_change_rel"] for h in H])
    if not rows:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    prec = "fp32" if "fp32" in rows else "fp64"
    for h, col in zip(rows[prec]["H"], ("#9ecae1", "#4292c6", "#08306b")):
        d = os.path.join(RES, f"poiseuille_H{h}_{prec}")
        p = np.genfromtxt(os.path.join(d, "profile.csv"), delimiter=",", names=True)
        ax[0].plot(p["u_sim"] / p["u_analytic"].max(), p["y_lat"] / h, color=col, lw=1.5, label=f"H = {h} cells ({prec})")
        if h == rows[prec]["H"][-1]:
            ax[0].plot(p["u_analytic"] / p["u_analytic"].max(), p["y_lat"] / h, "--", color=C["ref"], lw=1, label="analytic parabola")
    ax[0].set_xlabel("u / U_max"); ax[0].set_ylabel("y / H"); ax[0].set_title("Plane Poiseuille profile, tau = 0.65")
    ax[0].legend(loc="center left", fontsize=8)
    for prec2, col, mk in (("fp32", C["cd"], "o"), ("fp64", C["fp64"], "s")):
        if prec2 in rows:
            ax[1].loglog(rows[prec2]["H"], rows[prec2]["err"], mk + "-", color=col, lw=1.5, ms=6,
                         label=f"{prec2}, fitted order {rows[prec2]['order_fit']:.2f}")
    Hs = np.array(rows[prec]["H"], float); e0 = rows[prec]["err"][0]
    ax[1].loglog(Hs, e0 * (Hs[0] / Hs) ** 2, ":", color=C["guide"], lw=1, label="slope 2 guide")
    ax[1].axhline(1e-2, color="#c8552a", lw=0.8, ls="--"); ax[1].text(Hs[0], 1.1e-2, "acceptance limit 1 % at H = 82", fontsize=8, color="#c8552a")
    ax[1].set_xlabel("fluid height H (cells)"); ax[1].set_ylabel("relative L2 error of u(y)"); ax[1].set_title("Error vs resolution")
    ax[1].set_xticks(Hs); ax[1].set_xticklabels([str(int(h)) for h in Hs]); ax[1].xaxis.set_minor_formatter(NullFormatter()); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "poiseuille.png")); plt.close(fig)
    json.dump(rows, open(os.path.join(RES, "poiseuille_summary.json"), "w"), indent=1)
    return rows


# ---------------------------------------------------------------- grid studies
def gci(h, f, fs=1.25):
    """Richardson extrapolation and grid convergence index from the three finest grids.
    h: cells per diameter (coarse to fine), f: values. Constant refinement ratio r.
    Returns observed order p, extrapolated value, GCI on the finest grid in percent and
    whether the convergence is monotonic or oscillatory (sign change of the differences)."""
    (n3, n2, n1), (f3, f2, f1) = h[-3:], f[-3:]
    r = n1 / n2
    assert abs(r - n2 / n3) < 1e-12, "constant refinement ratio expected"
    e21, e32 = f1 - f2, f2 - f3
    if e21 == 0 or e32 == 0:
        return None
    p = abs(math.log(abs(e32 / e21))) / math.log(r)
    ext = f1 + e21 / (r ** p - 1)
    g = fs * abs(e21 / f1) / (r ** p - 1) * 100
    return dict(order=p, richardson=ext, gci_fine_pct=g, ratio=r, safety_factor=fs,
                convergence="monotonic" if e32 / e21 > 0 else "oscillatory", grids=[n3, n2, n1])


def grid_study():
    out = {}
    for case, keys in (("2D-1", ("cd", "cl", "dp")), ("2D-2", ("cd_max", "cl_max", "st", "dp_t0_half"))):
        rr = runs(f"{case}_N*_fp32")
        rr = {k: v for k, v in rr.items() if "rest" not in k}
        if not rr:
            continue
        Nd = sorted(int(re.search(r"_N(\d+)_", k).group(1)) for k in rr)
        entry = dict(Nd=Nd, files=[rr[f"{case}_N{n}_fp32"]["_dir"] + "/summary.json" for n in Nd], values={}, err_pct={}, orders={},
                     converged=[rr[f"{case}_N{n}_fp32"]["run"]["converged"] for n in Nd])
        for k in keys:
            vals = [rr[f"{case}_N{n}_fp32"]["results"][k] for n in Nd]
            errs = [rr[f"{case}_N{n}_fp32"]["error_pct"][k] for n in Nd]
            entry["values"][k] = vals; entry["err_pct"][k] = errs
            pair, fit = order(Nd, [abs(e) for e in errs])
            entry["orders"][k] = dict(pairs=pair, fit=fit)
            if len(Nd) >= 3:
                g = gci(Nd, vals)
                if g:
                    ref_k = rr[f"{case}_N{Nd[0]}_fp32"]["reference"][k]
                    g["richardson_error_pct"] = (g["richardson"] / ref_k - 1) * 100
                entry.setdefault("gci", {})[k] = g
        entry["reference"] = rr[f"{case}_N{Nd[0]}_fp32"]["reference"]
        out[case] = entry
    if not out:
        return out
    fig, axes = plt.subplots(1, len(out), figsize=(5 * len(out), 4), squeeze=False)
    for ax, (case, e) in zip(axes[0], out.items()):
        Nd = np.array(e["Nd"], float)
        for k, col in zip(e["err_pct"], (C["cd"], C["cl"], C["dp"], C["st"])):
            lab = {"cd": "cD", "cl": "cL", "dp": "Δp", "cd_max": "cD max", "cl_max": "cL max", "st": "St", "dp_t0_half": "Δp(t0 + T/2)"}[k]
            ax.loglog(Nd, np.abs(e["err_pct"][k]), "o-", color=col, lw=1.5, ms=6, label=f"{lab}, fitted order {e['orders'][k]['fit']:.2f}")
        ymax = max(abs(v) for vals in e["err_pct"].values() for v in vals)
        ax.loglog(Nd, ymax * (Nd[0] / Nd), ":", color=C["guide"], lw=1, label="slope 1 guide")
        ax.loglog(Nd, ymax * (Nd[0] / Nd) ** 2, "-.", color=C["guide"], lw=1, label="slope 2 guide")
        ax.set_xticks(Nd); ax.set_xticklabels([str(int(n)) for n in Nd]); ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("cells per diameter"); ax.set_ylabel("|error| vs reference (%)")
        ax.set_title(f"{case} grid study (FP32)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "grid_study.png")); plt.close(fig)
    json.dump(out, open(os.path.join(RES, "grid_study.json"), "w"), indent=1)
    return out


# ---------------------------------------------------------------- forces over time
def forces(gs):
    panels = []
    for case in ("2D-1", "2D-2"):
        if case in gs:
            n = gs[case]["Nd"][-1]; panels.append((case, n))
    if not panels:
        return
    fig, axes = plt.subplots(len(panels), 2, figsize=(11, 3.6 * len(panels)), squeeze=False)
    for row, (case, n) in enumerate(panels):
        d = os.path.join(RES, f"{case}_N{n}_fp32")
        f = np.genfromtxt(os.path.join(d, "forces.csv"), delimiter=",", names=True)
        s = load(os.path.join(d, "summary.json"))
        ref = s["reference"]
        for col, (k, name) in enumerate((("cd", "cD"), ("cl", "cL"))):
            ax = axes[row][col]
            ax.plot(f["t_DU"], f[k], color=C[k], lw=1.2, label=f"{name}, {case}, {n} cells/D")
            if case == "2D-1":
                ax.axhline(ref[k], color=C["ref"], ls="--", lw=1, label=f"reference {ref[k]:.5g}")
                lo, hi = s["reference"]["paper_range"][k]
                ax.axhspan(lo, hi, color="#dddddd", alpha=0.5, label="paper range")
                if k == "cd":
                    ax.set_ylim(ref[k] - 0.3, ref[k] + 0.3)
                else:
                    ax.set_ylim(-0.02, 0.04)
            else:
                for rk, lab in ((k + "_max", "max"), (k + "_min", "min")):
                    ax.axhline(ref[rk], color=C["ref"], ls="--", lw=1, label=f"reference {lab} {ref[rk]:.4f}")
                lo, hi = ref["paper_range"][k + "_max"]
                ax.axhspan(lo, hi, color="#dddddd", alpha=0.5, label="paper range (max)")
                w = s["results"]["window_DU"]
                ax.axvspan(w[0], w[1], color="#fdebd0", alpha=0.6, label="evaluation window (last 3 periods)")
                if k == "cd":
                    ax.set_ylim(ref["cd_min"] - 0.4, ref["cd_max"] + 0.4)
                else:
                    ax.set_ylim(-1.6, 1.6)
                ax.text(0.02, 0.97, "start-up transient clipped; full history in forces.csv", transform=ax.transAxes, fontsize=7, va="top", color="#666")
            ax.set_xlabel("t in D / U_mean"); ax.set_ylabel(name); ax.legend(fontsize=7, loc="best")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "forces.png")); plt.close(fig)


# ---------------------------------------------------------------- setup figure
def setup(gs):
    n = gs["2D-2"]["Nd"][-1] if "2D-2" in gs else 80
    s = load(os.path.join(RES, f"2D-2_N{n}_fp32", "summary.json"))["case"] if "2D-2" in gs else None
    fig, ax = plt.subplots(figsize=(11, 2.9))
    ax.set_aspect("equal"); ax.grid(False)
    ax.add_patch(plt.Rectangle((0, 0), 2.2, 0.41, fill=False, lw=1.5, color="#333333"))
    ax.add_patch(plt.Circle((0.2, 0.2), 0.05, color=CYL_RGB))
    ys = np.linspace(0.02, 0.39, 12)
    for y in ys:
        u = 4 * 1.0 * y * (0.41 - y) / 0.41 ** 2 * 0.085
        ax.annotate("", xy=(u, y), xytext=(0, y), arrowprops=dict(arrowstyle="->", color=C["cd"], lw=1))
    ax.text(0.015, 0.43, "inlet: u(y) = 4 Um y (H − y) / H², v = 0 (Zou and He)", fontsize=8, color=C["cd"])
    ax.text(2.2, 0.43, "outlet: p fixed (Zou and He), ρ = 1", fontsize=8, ha="right", color=C["cl"])
    ax.text(1.1, -0.045, "walls: no-slip (halfway bounce-back)", fontsize=8, ha="center")
    ax.plot([0.15, 0.25], [0.2, 0.2], "o", color=C["dp"], ms=4)
    ax.text(0.15, 0.275, "A1 (0.15, 0.2)", fontsize=7, ha="right", color=C["dp"]); ax.text(0.25, 0.275, "A2 (0.25, 0.2)", fontsize=7, ha="left", color=C["dp"])
    ax.annotate("", xy=(0.2, 0.15), xytext=(0.2, 0.0), arrowprops=dict(arrowstyle="<->", color="#666", lw=0.8)); ax.text(0.21, 0.07, "0.2", fontsize=7)
    ax.annotate("", xy=(0.2, 0.37), xytext=(0.0, 0.37), arrowprops=dict(arrowstyle="<->", color="#666", lw=0.8)); ax.text(0.1, 0.38, "0.2", fontsize=7, ha="center")
    ax.annotate("", xy=(2.25, 0.41), xytext=(2.25, 0.0), arrowprops=dict(arrowstyle="<->", color="#666", lw=0.8)); ax.text(2.27, 0.19, "H = 0.41", fontsize=7)
    ax.text(0.2, 0.2, "D = 0.1", fontsize=7, ha="center", va="center", color="white")
    ax.set_xlim(-0.05, 2.4); ax.set_ylim(-0.08, 0.5); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    title = "Schäfer and Turek flow around a cylinder, channel 2.2 m × 0.41 m, ν = 0.001 m²/s"
    if s:
        title += f"\nlattice at {n} cells per diameter: {s['Nx']} × {s['Ny']} nodes, dx = {s['dx_m']*1000:.3f} mm"
    ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "setup.png")); plt.close(fig)


# ---------------------------------------------------------------- vorticity images and GIF
def render_vort(v, types, vmax, signed, title, path):
    """v in U_mean / D units, rows y (0 at the bottom wall)."""
    Ny, Nx = v.shape
    if signed:
        rgb = plt.get_cmap("RdBu_r")((np.clip(v / vmax, -1, 1) + 1) / 2)[..., :3]
    else:
        rgb = LUT(np.clip(np.abs(v) / vmax, 0, 1))[..., :3]
    rgb[types == 4] = CYL_RGB; rgb[types == 1] = CYL_RGB
    fig = plt.figure(figsize=(12, 12 * Ny / Nx + 0.5)); ax = fig.add_axes([0.0, 0.0, 1.0, 1.0 - 0.5 / (12 * Ny / Nx + 0.5)])
    ax.imshow(rgb, origin="lower", interpolation="nearest"); ax.set_axis_off()
    fig.text(0.01, 1 - 0.3 / (12 * Ny / Nx + 0.5), title, fontsize=11, va="center")
    fig.savefig(path, dpi=110); plt.close(fig)


def vorticity(gs):
    for case in ("2D-1", "2D-2"):
        if case not in gs:
            continue
        n = gs[case]["Nd"][-1]
        d = os.path.join(RES, f"{case}_N{n}_fp32")
        if not os.path.exists(os.path.join(d, "fields.bin")):
            # the raw field dumps are not committed (gitignored, tens of MB each); a fresh
            # clone or the CI regenerates tables and line plots but keeps the committed images
            print(f"{case}: {os.path.relpath(d, ROOT)}/fields.bin not present, vorticity images left as committed")
            continue
        j, f, t = read_fields(d)
        v = f["vort_lat"] * j["vort_scale_to_U_over_D"]
        re_ = 20 if case == "2D-1" else 100
        title = f"Re = {re_}, Schäfer and Turek {case}, {n} cells per diameter, |ω| in U_mean / D (0 to 10)"
        render_vort(v, t, 10.0, False, title, os.path.join(FIG, f"vorticity_{case}.png"))
        if case == "2D-2":
            render_vort(v, t, 10.0, True, title.replace("|ω| in U_mean / D (0 to 10)", "signed ω, ±10 U_mean / D"),
                        os.path.join(FIG, "vorticity_2D-2_signed.png"))
            fb = os.path.join(d, "frames.bin")
            if os.path.exists(fb):
                fj = load(os.path.join(d, "frames.json"))
                a = np.fromfile(fb, dtype=np.float32).reshape(fj["nframes"], fj["Ny"], fj["Nx"]) * fj["vort_scale_to_U_over_D"]
                np.savez_compressed(os.path.join(d, "frames.npz"), vort_U_over_D=a, steps=np.array(fj["steps"]), period_DU=fj["period_DU"], steps_per_DU=fj["steps_per_DU"])
                scale = min(1.0, 1100 / fj["Nx"])
                frames = []
                for k in range(fj["nframes"]):
                    rgb = LUT(np.clip(np.abs(a[k]) / 10.0, 0, 1))[..., :3]
                    rgb[t == 4] = CYL_RGB; rgb[t == 1] = CYL_RGB
                    im = Image.fromarray((rgb[::-1] * 255).astype(np.uint8))
                    if scale < 1:
                        im = im.resize((int(fj["Nx"] * scale), int(fj["Ny"] * scale)), Image.LANCZOS)
                    frames.append(im.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
                frames[0].save(os.path.join(FIG, "vorticity.gif"), save_all=True, append_images=frames[1:], duration=60, loop=0, optimize=True)
                print("gif frames", len(frames), "size", frames[0].size)


# ---------------------------------------------------------------- tables, history
def fmt(v, nd=4):
    return "Unknown" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{nd}f}"


def tables(po, gs):
    L = ["# Result tables (generated by cuda/tools/report.py)", ""]
    if po:
        L += ["## Poiseuille verification", "", "| precision | H | relative L2 error | order (pairwise) | steps | source |", "|---|---|---|---|---|---|"]
        for prec, r in po.items():
            for i, h in enumerate(r["H"]):
                o = "" if i == 0 else f"{r['order_pairs'][i-1]:.2f}"
                L.append(f"| {prec} | {h} | {r['err'][i]:.3e} | {o} | {r['steps'][i]} | {r['files'][i]} |")
            L.append(f"| {prec} | fit | | {r['order_fit']:.2f} | | |")
        L.append("")
    for case, keys in (("2D-1", (("cd", "cD"), ("cl", "cL"), ("dp", "Δp"))), ("2D-2", (("cd_max", "cD max"), ("cl_max", "cL max"), ("st", "St"), ("dp_t0_half", "Δp(t0 + T/2)")))):
        if case not in gs:
            continue
        e = gs[case]; ref = e["reference"]
        L += [f"## {case} grid study (FP32)", "", "| cells per D | " + " | ".join(n for _, n in keys) + " | converged | source |", "|---|" + "---|" * (len(keys) + 2)]
        for i, n in enumerate(e["Nd"]):
            L.append(f"| {n} | " + " | ".join(f"{e['values'][k][i]:.5g} ({e['err_pct'][k][i]:+.2f} %)" for k, _ in keys) + f" | {e['converged'][i]} | {e['files'][i]} |")
        L.append("| reference | " + " | ".join(f"{ref[k]:.5g}" for k, _ in keys) + " | | " + ref["source"] + " |")
        L.append("| paper range | " + " | ".join(f"{ref['paper_range'][k.replace('_t0_half','')][0]} to {ref['paper_range'][k.replace('_t0_half','')][1]}" for k, _ in keys) + " | | Tables 3 and 4 of the paper |")
        L.append("| fitted order | " + " | ".join(f"{e['orders'][k]['fit']:.2f}" for k, _ in keys) + " | | |")
        L.append("")
        if e.get("gci"):
            L += [f"### {case}: Richardson extrapolation and grid convergence index (grids {', '.join(str(n) for n in e['Nd'][-3:])}, r = 2, safety factor 1.25)", "",
                  "| quantity | finest grid | Richardson value | its error vs reference | observed order | GCI fine | convergence |", "|---|---|---|---|---|---|---|"]
            for k, name in keys:
                g = e["gci"].get(k)
                if g:
                    L.append(f"| {name} | {e['values'][k][-1]:.5g} | {g['richardson']:.5g} | {g['richardson_error_pct']:+.2f} % | {g['order']:.2f} | {g['gci_fine_pct']:.2f} % | {g['convergence']} |")
                else:
                    L.append(f"| {name} | {e['values'][k][-1]:.5g} | Unknown | | | | no change between two grids |")
            L.append("")
    # FP32 vs FP64
    rows = []
    for case, nd, keys in (("2D-1", 40, ("cd", "cl", "dp")), ("2D-1", 80, ("cd", "cl", "dp")), ("2D-2", 80, ("cd_max", "cl_max", "st", "dp_t0_half"))):
        a = os.path.join(RES, f"{case}_N{nd}_fp32", "summary.json"); b = os.path.join(RES, f"{case}_N{nd}_fp64", "summary.json")
        if os.path.exists(a) and os.path.exists(b) and case in gs:
            sa, sb = load(a), load(b); e = gs[case]
            for k in keys:
                v32, v64 = sa["results"][k], sb["results"][k]
                i = e["Nd"].index(nd) if nd in e["Nd"] else None
                fine = abs(e["values"][k][-1] - e["values"][k][-2]) if len(e["Nd"]) >= 2 else float("nan")
                ratio = abs(v32 - v64) / fine if fine else float("nan")
                rows.append((case, nd, k, v32, v64, abs(v32 - v64), fine, ratio, os.path.relpath(b, ROOT).replace("\\", "/")))
    if rows:
        L += ["## FP32 vs FP64", "", "| case | cells per D | quantity | FP32 | FP64 | abs difference | change between two finest grids | ratio | flag | source |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.6g} | {r[4]:.6g} | {r[5]:.2e} | {r[6]:.2e} | {r[7]:.3f} | {'FP32 storage limiting' if r[7] > 0.1 else 'ok'} | {r[8]} |")
        L.append("")
    # mass drift and run facts
    L += ["## Run facts", "", "| run | steps | t end (D/U) | converged | mass drift (relative) | kernel MLUPS (events) | MLUPS incl. diagnostics | wall s | GPU at end | source |", "|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in sorted(runs("2D-*").items()):
        r = s["run"]
        L.append(f"| {k} | {r['steps']} | {r['t_end_DU']:.1f} | {r['converged']} | {r['mass_drift_rel']:.2e} | {r['mlups_kernel_events']:.0f} | {r['mlups_with_diagnostics']:.0f} | {r['wall_s']:.0f} | {r['gpu_end']} | {s['_dir']}/summary.json |")
    L.append("")
    # performance history
    hist = []
    base = load(os.path.join(RES, "baseline", "summary.json"))
    hist.append(("CuPy prototype (python/lbm_cupy.py, AoS, CUDA graph path)", "1600 x 400", base["cupy_prototype"]["solver_mlups"]["median"], base["cupy_prototype"]["solver_mlups"]["min"], base["cupy_prototype"]["solver_mlups"]["max"], "cuda/results/baseline/summary.json"))
    hist.append(("first CUDA version (karman_showcase.cu, SoA, headless)", "1600 x 400", base["cuda_first_version"]["solver_mlups"]["median"], base["cuda_first_version"]["solver_mlups"]["min"], base["cuda_first_version"]["solver_mlups"]["max"], "cuda/results/baseline/summary.json"))
    perf = os.path.join(RES, "perf", "summary.json")
    if os.path.exists(perf):
        p = load(perf)
        hist.append(("validated CUDA (cuda/lbm_bench.cu, SoA, shifted populations, kernel only)", p["grid"], p["kernel_mlups"]["median"], p["kernel_mlups"]["min"], p["kernel_mlups"]["max"], "cuda/results/perf/summary.json"))
        hist.append(("validated CUDA, including force and pressure diagnostics", p["grid"], p["diag_mlups"]["median"], p["diag_mlups"]["min"], p["diag_mlups"]["max"], "cuda/results/perf/summary.json"))
    with open(os.path.join(RES, "history.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")   # same bytes on every OS (the csv default is \r\n), so CI can compare
        w.writerow(["code", "grid", "mlups_median", "mlups_min", "mlups_max", "source"])
        for h in hist:
            w.writerow(h)
    L += ["## Performance history (same GPU, see cuda/results/MACHINE.md)", "", "| code | grid | MLUPS median | min | max | source |", "|---|---|---|---|---|---|"]
    for h in hist:
        L.append(f"| {h[0]} | {h[1]} | {h[2]:.0f} | {h[3]:.0f} | {h[4]:.0f} | {h[5]} |")
    L.append("")
    open(os.path.join(RES, "tables.md"), "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


# ---------------------------------------------------------------- roofline style performance figure
def perf_figure():
    """Effective memory bandwidth of each code (MLUPS x 73 bytes per cell update) against
    the bandwidth a plain copy kernel reaches on the same GPU and the theoretical value
    from the device attributes printed by stream_copy (memory clock and bus width)."""
    pp = os.path.join(RES, "perf", "summary.json")
    sc = os.path.join(RES, "perf", "stream_copy.log")
    if not (os.path.exists(pp) and os.path.exists(sc)):
        return
    p = load(pp)
    base = load(os.path.join(RES, "baseline", "summary.json"))
    m = re.search(r"memory clock (\d+) kHz, bus width (\d+) bit", open(sc).read())
    theo = 2 * int(m.group(1)) * 1e3 * int(m.group(2)) / 8 / 1e9    # double data rate
    peak = p["measured_peak_GBs"]
    rows = [("CuPy prototype", base["cupy_prototype"]["solver_mlups"]["median"], "#9a9a9a"),
            ("First CUDA version", base["cuda_first_version"]["solver_mlups"]["median"], "#9a9a9a"),
            ("Validated CUDA code,\nincluding diagnostics", p["diag_mlups"]["median"], C["cd"]),
            ("Validated CUDA code,\nkernel only", p["kernel_mlups"]["median"], C["cd"])]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.grid(axis="y", visible=False); ax.set_axisbelow(True)
    for i, (name, mlups, col) in enumerate(rows):
        gbs = mlups * 73 / 1e3
        ax.barh(i, gbs, height=0.55, color=col)
        ax.text(gbs - 4, i, f"measured: {mlups:,.0f} MLUPS, {gbs:.0f} GB/s, {100 * gbs / peak:.0f} %", va="center", ha="right", fontsize=8.5, color="white")
    ax.axvline(peak, color="#333333", lw=1.2, ls="--")
    ax.text(peak, len(rows) - 0.35, f"measured copy peak\n{peak:.0f} GB/s", fontsize=8, ha="right", va="bottom", color="#333333")
    ax.axvline(theo, color="#999999", lw=1, ls=":")
    ax.text(theo + 3, len(rows) - 0.35, f"theoretical\n{theo:.0f} GB/s", fontsize=8, ha="left", va="bottom", color="#666666")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
    ax.set_xlim(0, theo * 1.22); ax.set_ylim(-0.6, len(rows) + 0.35)
    ax.set_xlabel("effective memory bandwidth in GB/s (MLUPS × 73 bytes per cell update)")
    ax.set_title("How close each code gets to the memory bandwidth of the RTX 3060 Laptop GPU\nall bars measured on this GPU; percentages are of the measured copy peak", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "performance.png")); plt.close(fig)


if __name__ == "__main__":
    po = poiseuille()
    gs = grid_study()
    if gs:
        forces(gs); setup(gs); vorticity(gs)
    perf_figure()
    tables(po, gs)
