"""Regression test: run two short cases with the FP32 benchmark and compare them
with stored values (tests/expected.json). Takes a few seconds of GPU time.

    python tests/run_tests.py [path to lbm_bench executable]

Called by `make -C cuda test` and `cuda\\run.bat test`. Exit code 0 when every
quantity is within the relative tolerance, 1 otherwise.
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))


def default_exe():
    base = os.path.join(HERE, "..", "cuda", "build", "lbm_bench")
    return base + ".exe" if os.path.exists(base + ".exe") else base


def main():
    # absolute path: CreateProcess on Windows rejects relative paths with forward slashes
    exe = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else default_exe())
    if not os.path.exists(exe):
        sys.exit(f"benchmark executable not found: {exe} (build it first)")
    expected = json.load(open(os.path.join(HERE, "expected.json")))
    tol = expected["relative_tolerance"]
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        for case in expected["cases"]:
            out = os.path.join(tmp, case["name"])
            run = subprocess.run([exe, *case["args"], "--out", out], capture_output=True, text=True)
            if run.returncode != 0:
                print(f"FAIL {case['name']}: benchmark exited with code {run.returncode}")
                failed += 1
                continue
            got = json.load(open(os.path.join(out, "summary.json")))[case["section"]]
            for key, ref in case["expected"].items():
                rel = abs(got[key] / ref - 1)
                ok = rel <= tol
                failed += not ok
                print(f"{'PASS' if ok else 'FAIL'} {case['name']:16s} {key:13s} {got[key]:.8g} vs {ref:.8g} (relative difference {rel:.1e}, tolerance {tol:.0e})")
    print("all tests passed" if failed == 0 else f"{failed} check(s) failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
