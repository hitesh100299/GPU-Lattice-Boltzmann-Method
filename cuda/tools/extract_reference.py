"""Extract the 2D-2 reference quantities from the FeatFlow 2014 official results.

The zip is not part of the repository. If it's missing, the script downloads it from the
FeatFlow 2D-2 benchmark page into cuda/reference/, checks its sha256 and unpacks it there.

Input : cuda/reference/20141124_dfg_bench2_official/logs/results_cc2d_bench2_referencenew/
        bdforces_q2_lv{L}_dt{K}    columns: timestep, time, bdc, cD, cL
        pointvalues_q2_lv{L}_dt{K} columns 7 and 12: pressure at (0.15, 0.2) and (0.25, 0.2)
Output: cuda/reference/extracted_2d2_reference.json

Method (same as the solver's own post-processing, see lbm_core.cuh, Diag):
  - periods from upward zero crossings of cL with linear interpolation; St = D f / U_mean = 0.1 / T
  - cD max, cD min, cL max, cL min: extrema over the whole file (t in [9, 10], about 3 periods)
  - Delta p at t0 + T/2 where t0 is the time of the cL maximum (parabolic interpolation),
    the paper's definition (page 4 of the paper: "initial data (t = t0) should correspond to
    the flow state with cLmax" and "pressure difference at t = t0 + 1/2f").
Run from the repository root: python cuda/tools/extract_reference.py
"""
import hashlib
import json
import os
import urllib.request
import zipfile

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "reference")
D = os.path.join(ROOT, "20141124_dfg_bench2_official", "logs", "results_cc2d_bench2_referencenew")
ZIP = os.path.join(ROOT, "20141124_dfg_bench2_official.zip")
URL = "https://wwwold.mathematik.tu-dortmund.de/~featflow/media/dfg_bench2new_2d/20141124_dfg_bench2_official.zip"
SHA256 = "08db59605b2f60efc512d2bbc79628ea5d1cc720c0986938a16920884a0215be"


def fetch_reference():
    """Download and unpack the FeatFlow zip unless the extracted logs are already there."""
    if os.path.isdir(D):
        return
    if not os.path.exists(ZIP):
        print("downloading", URL)
        urllib.request.urlretrieve(URL, ZIP)
    h = hashlib.sha256(open(ZIP, "rb").read()).hexdigest()
    if h != SHA256:
        raise SystemExit(f"{ZIP}: sha256 {h} does not match the expected {SHA256}")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(os.path.join(ROOT, "20141124_dfg_bench2_official"))  # the zip has no top folder
    if not os.path.isdir(D):
        raise SystemExit(f"{D} not found after unpacking {ZIP}")


def pinterp(t, arr, i):
    """Parabolic interpolation of an extremum around sample i."""
    if 0 < i < len(arr) - 1:
        y0, y1, y2 = arr[i - 1], arr[i], arr[i + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        return t[i] + off * (t[1] - t[0]), y1 - 0.25 * (y0 - y2) * off
    return t[i], arr[i]


def main():
    fetch_reference()
    out = {}
    for lv in (2, 3, 4, 5, 6):
        for dt in (1, 2, 4):
            f = np.loadtxt(os.path.join(D, f"bdforces_q2_lv{lv}_dt{dt}"))
            t, cd, cl = f[:, 1], f[:, 3], f[:, 4]
            p = np.loadtxt(os.path.join(D, f"pointvalues_q2_lv{lv}_dt{dt}"))
            tp, dp = p[:, 1], p[:, 6] - p[:, 11]
            zc = [t[i] - cl[i] * (t[i + 1] - t[i]) / (cl[i + 1] - cl[i])
                  for i in range(len(t) - 1) if cl[i] < 0 <= cl[i + 1]]
            per = np.diff(zc)
            T = float(per.mean()) if len(per) else float("nan")
            tmax, clmax_i = pinterp(t, cl, int(np.argmax(cl)))
            tcd, cdmax_i = pinterp(t, cd, int(np.argmax(cd)))
            thalf = tmax + T / 2
            if thalf > tp[-1]:
                thalf -= T
            out[f"lv{lv}_dt{dt}"] = dict(
                samples=int(len(t)), t_range=[float(t[0]), float(t[-1])],
                cd_max=float(cd.max()), cd_max_interp=float(cdmax_i), cd_min=float(cd.min()),
                cl_max=float(cl.max()), cl_max_interp=float(clmax_i), cl_min=float(cl.min()),
                n_periods=int(len(per)), period_mean=T,
                period_spread=float(per.max() - per.min()) if len(per) else None,
                St=0.1 / T, dp_max=float(dp.max()), dp_min=float(dp.min()),
                dp_at_t0_plus_half_period=float(np.interp(thalf, tp, dp)))
    out["_source"] = ("FeatFlow 20141124_dfg_bench2_official.zip, sha256 "
                      "08db59605b2f60efc512d2bbc79628ea5d1cc720c0986938a16920884a0215be; "
                      "primary reference = lv6_dt4")
    path = os.path.join(ROOT, "extracted_2d2_reference.json")
    json.dump(out, open(path, "w"), indent=1)
    r = out["lv6_dt4"]
    print("lv6_dt4: cD max %.5f  cD min %.5f  cL max %.5f  cL min %.5f  St %.5f  dp(t0+T/2) %.5f  T %.5f"
          % (r["cd_max"], r["cd_min"], r["cl_max"], r["cl_min"], r["St"], r["dp_at_t0_plus_half_period"], r["period_mean"]))


if __name__ == "__main__":
    main()
