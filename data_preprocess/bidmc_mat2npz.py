#!/usr/bin/env python3
import argparse, os, sys, numpy as np

def _as_obj_array(list_of_arrays):
    out = np.empty(len(list_of_arrays), dtype=object)
    for i, a in enumerate(list_of_arrays):
        a = np.asarray(a, dtype=float).ravel()
        out[i] = a
    return out

def _save_subject(seg, out_dir, i1):
    ppg = np.asarray(seg["ppg"], dtype=float)
    if ppg.ndim != 2:
        raise ValueError(f"PPG must be 2-D (n_windows, 1024). Got shape {ppg.shape}")
    ibi_cell = seg.get("ibi_ecg_ms", [])

    if ibi_cell is None:
        ibi_cell = [np.array([], float)] * ppg.shape[0]
    ibi_obj = _as_obj_array(ibi_cell)
    fname = os.path.join(out_dir, f"seg_{i1:02d}.npz")
    np.savez_compressed(fname, ppg=ppg.astype(np.float32), ibi_ecg_ms=ibi_obj)
    return fname

def _try_mat73(mat_path):
    try:
        import mat73  # type: ignore
        d = mat73.loadmat(mat_path)
        if "segs" not in d:
            return None
        segs = d["segs"]
        if isinstance(segs, dict):  # some mat73 versions return dict with numeric keys
            segs = [segs[k] for k in sorted(segs.keys(), key=lambda x: int(x))]
        # normalize into python dicts we need
        out = []
        for s in segs:
            seg = {}
            seg["ppg"] = np.asarray(s.get("ppg", np.empty((0, 0))), float)
            ibi = s.get("ibi_ecg_ms", [])
            if isinstance(ibi, (list, tuple)):
                ibi_list = [np.asarray(v, float).ravel() if v is not None else np.array([], float) for v in ibi]
            else:
                ibi_list = [np.asarray(ibi, float).ravel()]
            seg["ibi_ecg_ms"] = ibi_list
            out.append(seg)
        return out
    except Exception:
        return None

def _try_scipy(mat_path):
    try:
        from scipy.io import loadmat
        d = loadmat(mat_path, struct_as_record=False, squeeze_me=True)
        if "segs" not in d:
            return None
        segs_raw = d["segs"]
        if np.ndim(segs_raw) == 0:
            segs_raw = np.array([segs_raw], dtype=object)
        out = []
        for s in np.ravel(segs_raw):
            seg = {}
            seg["ppg"] = np.asarray(getattr(s, "ppg", np.empty((0, 0))), float)
            ibi = getattr(s, "ibi_ecg_ms", None)
            if ibi is None:
                seg["ibi_ecg_ms"] = []
            else:
                if isinstance(ibi, np.ndarray) and ibi.dtype == object:
                    seg["ibi_ecg_ms"] = [np.asarray(x, float).ravel() if x is not None else np.array([], float)
                                         for x in ibi.flat]
                else:
                    seg["ibi_ecg_ms"] = [np.asarray(ibi, float).ravel()]
            out.append(seg)
        return out
    except Exception:
        return None

def _try_h5py(mat_path):
    """
    Robust v7.3 reader for the common layout:
      /segs (Group)
        /ppg           -> Dataset of object refs, shape (N,1) or (N,)
        /ibi_ecg_ms    -> Dataset of object refs, shape (N,1) or (N,)
        /subject       -> Dataset of object refs, shape (N,1) or (N,)
    Each element is a reference to a numeric dataset (ppg) or to a cell array (ibi),
    where the cell array itself is an array of references to numeric vectors.
    """
    try:
        import h5py  # type: ignore
    except Exception as e:
        print("h5py not installed; install mat73 or scipy.", file=sys.stderr)
        return None

    def _read_ref_dataset(f, ref):
        """Dereference a Dataset/Group from an object reference."""
        obj = f[ref]
        if isinstance(obj, h5py.Dataset):
            return obj[()]
        # Group: try common patterns ('data' inside) else first dataset
        if isinstance(obj, h5py.Group):
            if "data" in obj:
                return obj["data"][()]
            # fall back: take the first dataset inside group
            for k in obj.keys():
                if isinstance(obj[k], h5py.Dataset):
                    return obj[k][()]
        return None

    def _read_string(f, ref):
        arr = _read_ref_dataset(f, ref)
        if arr is None:
            return ""
        arr = np.array(arr)
        # char arrays (uint16/uint8) or fixed strings
        if arr.dtype.kind in ("U", "S"):
            try:
                return arr.tobytes().decode(errors="ignore")
            except Exception:
                return str(arr)
        if arr.dtype.kind in ("u", "i"):
            try:
                return "".join(chr(c) for c in arr.flatten())
            except Exception:
                return str(arr.flatten())
        return str(arr)

    try:
        with h5py.File(mat_path, "r") as f:
            if "segs" not in f:
                return None
            g = f["/segs"]
            if not isinstance(g, h5py.Group):
                return None

            # Find field datasets (ppg is required; ibi_ecg_ms optional)
            if "ppg" not in g:
                return None
            ds_ppg = g["ppg"]
            n = ds_ppg.shape[0]  # subjects

            # Helper to index ref array regardless of (N,) vs (N,1)
            def _idx(ds, i):
                if ds.ndim == 1:
                    return ds[i]
                # Many MATLAB saves use (N,1)
                return ds[i, 0]

            ds_ibi = g["ibi_ecg_ms"] if "ibi_ecg_ms" in g else None
            ds_subj = g["subject"] if "subject" in g else None

            segs = []
            for i in range(n):
                # ---- PPG ----
                ref_ppg = _idx(ds_ppg, i)
                ppg_arr = _read_ref_dataset(f, ref_ppg)
                ppg_arr = np.array(ppg_arr, dtype=float)
                # Ensure 2-D (nW, 1024)
                ppg_arr = np.atleast_2d(ppg_arr)
                if ppg_arr.shape[0] == 1024 and ppg_arr.shape[1] != 1024:
                    # It might be transposed (1024 x nW)
                    ppg_arr = ppg_arr.T

                # ---- IBI (cell array) ----
                ibi_list = []
                if ds_ibi is not None:
                    ref_cell = _idx(ds_ibi, i)
                    # cell array likely returns array of object refs
                    cell = _read_ref_dataset(f, ref_cell)
                    if cell is not None:
                        cell = np.atleast_1d(cell)
                        for cref in cell.flat:
                            if isinstance(cref, (bytes, str)) or cref is None:
                                ibi_list.append(np.array([], float))
                                continue
                            try:
                                vec = _read_ref_dataset(f, cref)
                                if vec is None:
                                    ibi_list.append(np.array([], float))
                                else:
                                    v = np.asarray(vec, float).ravel()
                                    ibi_list.append(v)
                            except Exception:
                                ibi_list.append(np.array([], float))

                # ---- Subject (optional) ----
                subject = ""
                if ds_subj is not None:
                    try:
                        subject = _read_string(f, _idx(ds_subj, i))
                    except Exception:
                        subject = ""

                segs.append({"subject": subject, "ppg": ppg_arr, "ibi_ecg_ms": ibi_list})
            return segs
    except Exception as e:
        print(f"h5py reader failed: {e}", file=sys.stderr)
        return None

def _load_mat_any(mat_path):
    # Try mat73
    segs = _try_mat73(mat_path)
    if segs is not None:
        return segs
    # Try scipy (v7)
    segs = _try_scipy(mat_path)
    if segs is not None:
        return segs
    # Try robust h5py (v7.3)
    segs = _try_h5py(mat_path)
    if segs is not None:
        return segs
    raise RuntimeError(f"Could not load {mat_path} with mat73/scipy/h5py.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True, help="Path to segments_*.mat")
    ap.add_argument("--out", default="npz_out", help="Output directory for seg_*.npz")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    segs = _load_mat_any(args.mat)
    if not segs:
        print("No segments found.", file=sys.stderr); sys.exit(1)

    saved = []
    for i, seg in enumerate(segs, start=1):
        path = _save_subject(seg, args.out, i1=i)
        saved.append(path)

    print(f"Saved {len(saved)} subjects to: {args.out}")
    for p in saved[:min(5, len(saved))]:
        print("  -", p)

if __name__ == "__main__":
    main()
