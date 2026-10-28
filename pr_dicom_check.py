"""Do the piglet reconstructions (FBP / ASIR / MBIR) of a dose come from the SAME acquisition?

Paired supervision (training strategies 'real' and 'joint') is only allowed if this check passes.

1) DICOM headers: the series of one dose must share study, frame of reference, acquisition time/number,
   tube current-time product, CTDIvol, kVp, slice positions and pixel spacing, and differ only in the
   reconstruction (kernel / IR level / series UID). Paths come from data/piglet/index.json.
2) Images: the low-pass difference between two reconstructions of the same dose must be much smaller
   than between two doses of the same reconstruction (different acquisitions).

    python pr_dicom_check.py --data data/piglet
    python pr_dicom_check.py --data data/piglet --root /path/to/piglet_sorted   # if the DICOM tree moved
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
from scipy import ndimage

SAME = ["StudyInstanceUID", "FrameOfReferenceUID", "AcquisitionDate", "AcquisitionTime", "AcquisitionNumber",
        "KVP", "XRayTubeCurrent", "Exposure", "ExposureTime", "CTDIvol", "SliceThickness", "PixelSpacing",
        "DataCollectionDiameter", "SpiralPitchFactor"]
INFO = ["SeriesDescription", "SeriesInstanceUID", "ConvolutionKernel", "ReconstructionDiameter",
        "SeriesTime", "ContentTime"]


def header(folder):
    import pydicom
    for f in sorted(os.listdir(folder)):
        p = os.path.join(folder, f)
        if os.path.isfile(p) and f.upper() != "DICOMDIR":
            try:
                return pydicom.dcmread(p, stop_before_pixels=True, force=True)
            except Exception:
                continue
    return None


def val(ds, k):
    v = getattr(ds, k, None) if ds is not None else None
    if v is None:
        return None
    return str(list(v)) if hasattr(v, "__iter__") and not isinstance(v, (str, bytes)) else str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/piglet")
    ap.add_argument("--root", default=None, help="replace the DICOM root stored in index.json")
    ap.add_argument("--n_slices", type=int, default=20)
    a = ap.parse_args()
    index = json.load(open(os.path.join(a.data, "index.json")))
    by_dose = defaultdict(dict)
    for k, v in index.items():
        if "alt" in k:
            continue
        by_dose[int(v["dose"])][v["recon"]] = k

    print("== 1) DICOM headers")
    verdict = {}
    for dose, recs in sorted(by_dose.items()):
        hdr = {}
        for r, k in recs.items():
            path = index[k].get("path")
            if a.root and path:
                parts = path.replace("\\", "/").split("/")
                for i in range(len(parts)):
                    cand = os.path.join(a.root, *parts[i:])
                    if os.path.isdir(cand):
                        path = cand
                        break
            hdr[r] = header(path) if path and os.path.isdir(path) else None
        print(f"\n-- dose {dose}%: {sorted(recs)}")
        if all(h is None for h in hdr.values()):
            print("   DICOM not found (use --root); header check skipped")
            verdict[dose] = "unknown"
            continue
        ok = True
        for f in SAME:
            vals = {r: val(h, f) for r, h in hdr.items()}
            uniq = {v for v in vals.values() if v is not None}
            flag = "same" if len(uniq) <= 1 else "DIFFERENT"
            if flag == "DIFFERENT" and f not in ("SliceThickness",):
                ok = False
            print(f"   {f:24s} {flag:9s} " + "  ".join(f"{r}={vals[r]}" for r in sorted(vals)))
        for f in INFO:
            vals = {r: val(h, f) for r, h in hdr.items()}
            print(f"   {f:24s} {'(info)':9s} " + "  ".join(f"{r}={vals[r]}" for r in sorted(vals)))
        zsets = {r: set(index[k]["z"]) for r, k in recs.items()}
        common = set.intersection(*zsets.values())
        print(f"   slice positions: {len(common)} common of {[len(s) for s in zsets.values()]}")
        verdict[dose] = "paired" if ok else "NOT paired"

    print("\n== 2) image check: low-pass (G3) difference std in the body, HU")
    lp = lambda x: ndimage.gaussian_filter(x, 3)
    for dose, recs in sorted(by_dose.items()):
        if dose == 100 or "FBP" not in recs:
            continue
        kf = recs["FBP"]
        if not os.path.exists(os.path.join(a.data, kf + ".npy")):
            print(f"-- dose {dose}%: {kf}.npy missing, skipped")
            continue
        zf = index[kf]["z"]
        vf = np.load(os.path.join(a.data, kf + ".npy"), mmap_mode="r")
        sel = np.linspace(len(zf) * 0.2, len(zf) * 0.8, a.n_slices).astype(int)
        other_dose = next((recs2["FBP"] for d2, recs2 in sorted(by_dose.items())
                           if d2 not in (dose, 100) and "FBP" in recs2
                           and os.path.exists(os.path.join(a.data, recs2["FBP"] + ".npy"))), None)
        def lpd(k2):
            if not os.path.exists(os.path.join(a.data, k2 + ".npy")):
                return float("nan")
            z2 = {z: i for i, z in enumerate(index[k2]["z"])}
            v2 = np.load(os.path.join(a.data, k2 + ".npy"), mmap_mode="r")
            out = []
            for i in sel:
                if zf[i] in z2:
                    x, y = np.asarray(vf[i], np.float32), np.asarray(v2[z2[zf[i]]], np.float32)
                    body = x > -500
                    out.append(np.std((lp(y) - lp(x))[body]))
            return np.median(out) if out else np.nan
        ref = lpd(other_dose) if other_dose else np.nan
        print(f"-- dose {dose}%  reference (FBP {dose}% vs {other_dose}, different acquisitions): {ref:.2f}")
        for r, k in sorted(recs.items()):
            if r != "FBP":
                v = lpd(k)
                print(f"   FBP vs {r:7s}: " + (f"{v:.2f}" if np.isfinite(v) else f"(file {k}.npy missing)"))
    print("\n== verdict (header check)")
    for d, v in sorted(verdict.items()):
        print(f"   dose {d}%: {v}")
    print("Paired supervision is allowed only for doses marked 'paired' AND whose FBP-vs-r low-pass difference\n"
          "is clearly below the between-dose reference (IR changes the low-pass image a little; a value close to\n"
          "the reference means different acquisitions).")


if __name__ == "__main__":
    main()
