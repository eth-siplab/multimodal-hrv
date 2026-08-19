# aggregate_folds_unweighted.py
import argparse
import os, glob, argparse, re, sys
import numpy as np
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from utils import print_aggregate

def _parse_ratio(dirname: str):
    m = re.match(r"^ratio_(\d+)[._]?(\d+)?$", dirname)
    if not m:
        return None
    a = m.group(1); b = m.group(2)
    return float(a) if b is None else float(f"{a}.{b}")

def _parse_exclude(s: str):
    """Parse comma/space separated IDs: '2', '2,5  9' -> [2,5,9]."""
    if not s:
        return []
    toks = re.split(r"[,\s]+", s.strip())
    ids = []
    for t in toks:
        if not t:
            continue
        try:
            ids.append(int(t))
        except:
            pass
    return sorted(set(ids))

def _read_all_summaries(root_dir: str, keep_ratio: float):
    # 1) gather any summary CSVs under fold_*/ratio_*/summary*.csv
    candidates = sorted(set(
        glob.glob(os.path.join(root_dir, "fold_*", "ratio_*", "summary*.csv"))
    ))
    if not candidates:
        raise FileNotFoundError(
            f"No summary CSVs found under {root_dir}/fold_*/ratio_*/"
        )

    # 2) filter by desired keep_ratio (accept 1e-3 tolerance)
    want = round(float(keep_ratio), 2)
    picked = []
    for p in candidates:
        parts = p.split(os.sep)
        ratio_dir = next((q for q in parts if q.startswith("ratio_")), None)
        r = _parse_ratio(ratio_dir) if ratio_dir else None
        if r is None:
            continue
        if abs(round(r, 2) - want) < 1e-3:
            picked.append(p)

    if not picked:
        found = sorted({p.split(os.sep)[-3] for p in candidates})
        raise FileNotFoundError(
            f"No summary CSVs under {root_dir}/fold_*/ratio_{want:.2f}/\n"
            f"Found ratio dirs: {found}"
        )
    return sorted(picked)

def _fold_id_from_path(path: str) -> int:
    for part in path.split(os.sep):
        if part.startswith("fold_"):
            try:
                return int(part.split("_")[1])
            except:
                pass
    return -1

def _mean(series):
    x = pd.to_numeric(series, errors="coerce").values
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

def _fmt(x, nd=3, signed=False):
    if not np.isfinite(x):
        return "--"
    return f"{x:+.{nd}f}" if signed else f"{x:.{nd}f}"

def _print_overall_summary(overall_mean: pd.Series):
    # IBI
    print("IBI:",
          f"MAE={_fmt(overall_mean.get('MAE', np.nan), 2)} ms,",
          f"RMSE={_fmt(overall_mean.get('RMSE', np.nan), 2)} ms,",
          f"r={_fmt(overall_mean.get('r', np.nan), 3)}")

    # Time-domain HRV (MAE over subjects)
    print("Time HRV MAE:",
          f"SDNN 5m={_fmt(overall_mean.get('MAE_SDNN5m', np.nan), 2)} ms,",
          f"SDNN 10m={_fmt(overall_mean.get('MAE_SDNN10m', np.nan), 2)} ms,",
          f"RMSSD 5m={_fmt(overall_mean.get('MAE_RMSSD5m', np.nan), 2)} ms,",
          f"RMSSD 10m={_fmt(overall_mean.get('MAE_RMSSD10m', np.nan), 2)} ms")

    # Frequency-domain HRV (Bland–Altman means across subjects)
    def _ba_line(tag):
        b  = overall_mean.get(f"{tag}_bias_pct_mean", np.nan)
        lo = overall_mean.get(f"{tag}_loa_lo_pct_mean", np.nan)
        hi = overall_mean.get(f"{tag}_loa_hi_pct_mean", np.nan)
        return f"{tag}: bias={_fmt(b,2,signed=True)}%  LOA[{_fmt(lo,2,signed=True)}%, {_fmt(hi,2,signed=True)}%]"

    print("Freq HRV (BA, means across folds):")
    print(" ", _ba_line("LF5"))
    print(" ", _ba_line("HF5"))
    print(" ", _ba_line("LF10"))
    print(" ", _ba_line("HF10"))

def _print_overall_summary_mean_std(overall_mean: pd.Series, overall_std: pd.Series):
    def ms(name: str, nd: int = 2, unit: str = "", signed: bool = False):
        m = overall_mean.get(name, np.nan)
        s = overall_std.get(name, np.nan)
        u = f" {unit}" if unit else ""
        return f"{name}={_fmt(m, nd, signed=signed)}±{_fmt(s, nd, signed=False)}{u}"

    print("\n=== Aggregate metrics (mean ± std across folds) ===")
    # IBI
    print("IBI:",
          ms("MAE", 2, "ms"), ",",
          ms("RMSE", 2, "ms"), ",",
          ms("r", 3))

    # Time-domain HRV (MAE over subjects)
    print("Time HRV MAE:",
          ms("MAE_SDNN5m", 2, "ms"), ",",
          ms("MAE_SDNN10m", 2, "ms"), ",",
          ms("MAE_RMSSD5m", 2, "ms"), ",",
          ms("MAE_RMSSD10m", 2, "ms"))

    # Frequency-domain HRV BA (percent)
    def ba_ms(tag: str):
        b  = ms(f"{tag}_bias_pct_mean", 2, "%", signed=True)
        lo = ms(f"{tag}_loa_lo_pct_mean", 2, "%", signed=True)
        hi = ms(f"{tag}_loa_hi_pct_mean", 2, "%", signed=True)
        return f"{tag}: bias {b}  LOA_lo {lo}  LOA_hi {hi}"

    print("Freq HRV (BA):")
    print(" ", ba_ms("LF5"))
    print(" ", ba_ms("HF5"))
    print(" ", ba_ms("LF10"))
    print(" ", ba_ms("HF10"))

def _add_delta_row(out: dict, df: pd.DataFrame, h_col: str, e_col: str, out_name: str):
    if {h_col, e_col}.issubset(df.columns):
        h = pd.to_numeric(df[h_col], errors="coerce")
        e = pd.to_numeric(df[e_col], errors="coerce")
        out[out_name] = _mean(h - e)
    else:
        out[out_name] = np.nan

def _add_mae_row(out: dict, df: pd.DataFrame, h_col: str, e_col: str, out_name: str):
    if {h_col, e_col}.issubset(df.columns):
        h = pd.to_numeric(df[h_col], errors="coerce")
        e = pd.to_numeric(df[e_col], errors="coerce")
        out[out_name] = _mean((h - e).abs())   # mean absolute error across subjects
    else:
        out[out_name] = np.nan

def _mean_numeric(series_or_vals):
    x = pd.to_numeric(series_or_vals, errors="coerce").values
    return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan

def _agg_ba_group(out: dict, df: pd.DataFrame, tag: str):
    """
    Accumulate unweighted subject means for a BA group.
    tag ∈ {"LF5","HF5","LF10","HF10"}
    Produces keys:
      {tag}_bias_pct_mean, {tag}_loa_lo_pct_mean, {tag}_loa_hi_pct_mean,
      {tag}_width_pct_mean, {tag}_n_mean
    """
    bias_col = f"{tag}_bias_pct"
    lo_col   = f"{tag}_loa_lo_pct"
    hi_col   = f"{tag}_loa_hi_pct"
    n_col    = f"{tag}_n"

    if {bias_col, lo_col, hi_col}.issubset(df.columns):
        bias_mean = _mean_numeric(df[bias_col])
        lo_mean   = _mean_numeric(df[lo_col])
        hi_mean   = _mean_numeric(df[hi_col])
        # width per-subject (need row-wise)
        width_vals = pd.to_numeric(df[hi_col], errors="coerce") - pd.to_numeric(df[lo_col], errors="coerce")
        width_mean = _mean_numeric(width_vals)
    else:
        bias_mean = lo_mean = hi_mean = width_mean = np.nan

    n_mean = _mean_numeric(df[n_col]) if n_col in df.columns else np.nan

    out[f"{tag}_bias_pct_mean"]   = bias_mean
    out[f"{tag}_loa_lo_pct_mean"] = lo_mean
    out[f"{tag}_loa_hi_pct_mean"] = hi_mean
    out[f"{tag}_width_pct_mean"]  = width_mean
    out[f"{tag}_n_mean"]          = n_mean        

def _agg_one_fold(df: pd.DataFrame) -> dict:
    out = {}
    out["n_subjects"] = int(df.shape[0])

    # basic errors
    for col in ["MAE","RMSE","Bias","r","coverage"]:
        if col in df.columns:
            out[col] = _mean(df[col])

    # totals (not used for means)
    if {"kept_T","total_T"}.issubset(df.columns):
        kept = pd.to_numeric(df["kept_T"], errors="coerce").fillna(0).sum()
        tot  = pd.to_numeric(df["total_T"], errors="coerce").fillna(0).sum()
        out["kept_T_sum"]  = int(kept)
        out["total_T_sum"] = int(tot)
        out["coverage_sum"] = float(kept / tot) if tot > 0 else np.nan

    # global HRV: signed deltas + MAE
    _add_delta_row(out, df, "SDNN_hat","SDNN_ecg","d_SDNN")
    _add_delta_row(out, df, "RMSSD_hat","RMSSD_ecg","d_RMSSD")
    _add_mae_row  (out, df, "SDNN_hat","SDNN_ecg","MAE_SDNN")
    _add_mae_row  (out, df, "RMSSD_hat","RMSSD_ecg","MAE_RMSSD")

    # 5-min means
    _add_delta_row(out, df, "SDNN5m_hat","SDNN5m_ecg","d_SDNN5m")
    _add_delta_row(out, df, "RMSSD5m_hat","RMSSD5m_ecg","d_RMSSD5m")
    _add_mae_row  (out, df, "SDNN5m_hat","SDNN5m_ecg","MAE_SDNN5m")
    _add_mae_row  (out, df, "RMSSD5m_hat","RMSSD5m_ecg","MAE_RMSSD5m")

    # 10-min means
    _add_delta_row(out, df, "SDNN10m_hat","SDNN10m_ecg","d_SDNN10m")
    _add_delta_row(out, df, "RMSSD10m_hat","RMSSD10m_ecg","d_RMSSD10m")
    _add_mae_row  (out, df, "SDNN10m_hat","SDNN10m_ecg","MAE_SDNN10m")
    _add_mae_row  (out, df, "RMSSD10m_hat","RMSSD10m_ecg","MAE_RMSSD10m")

    # frequency BA biases (percent, unweighted)
    for tag in ["LF5","HF5","LF10","HF10"]:
        _agg_ba_group(out, df, tag)

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="e.g., results/unet_dalia_cuda0_bs128_sw1024")
    ap.add_argument("--keep_ratio", type=float, default=0.80, help="e.g., 0.80 -> fold_*/ratio_0.80")
    ap.add_argument("--out_dir", default=None, help="where to write summaries (default: ratio folder)")
    ap.add_argument("--exclude_subjects", type=str, default="", help="comma/space-separated subject IDs to exclude, e.g. '2,7 15'")
    args = ap.parse_args()

    exclude_ids = _parse_exclude(args.exclude_subjects)

    csv_files = _read_all_summaries(args.root, args.keep_ratio)
    rows = []
    for f in csv_files:
        fold_id = _fold_id_from_path(f)
        df = pd.read_csv(f)

        # filter excluded subjects (if any)
        if exclude_ids and "subject_id" in df.columns:
            sid_num = pd.to_numeric(df["subject_id"], errors="coerce").astype("Int64")
            df = df[~sid_num.isin(exclude_ids)].reset_index(drop=True)

        agg = _agg_one_fold(df)
        agg["fold"] = fold_id
        agg["csv_path"] = f
        agg["excluded"] = ",".join(map(str, exclude_ids)) if exclude_ids else ""
        rows.append(agg)

    fold_df = pd.DataFrame(rows).sort_values("fold")

    # overall mean ± std across folds (numeric cols only)
    num_cols = fold_df.select_dtypes(include=[np.number]).columns
    overall_mean = fold_df[num_cols].mean().to_frame("mean").T
    overall_std  = fold_df[num_cols].std(ddof=1).to_frame("std").T
    overall = pd.concat([overall_mean, overall_std], axis=0)
    overall.insert(0, "stat", ["mean","std"])
    if exclude_ids:
        overall.insert(1, "excluded", [",".join(map(str, exclude_ids))]*2)

    ratio_tag = f"ratio_{args.keep_ratio:.2f}"
    out_dir = args.out_dir or os.path.join(args.root, ratio_tag)
    os.makedirs(out_dir, exist_ok=True)

    fold_csv = os.path.join(out_dir, "fold_summary.csv")
    overall_csv = os.path.join(out_dir, "overall_summary.csv")

    fold_df.to_csv(fold_csv, index=False, float_format="%.2f")
    overall.to_csv(overall_csv, index=False, float_format="%.2f")
    print(f"wrote {fold_csv}")
    print(f"wrote {overall_csv}")

    # Print both mean-only and mean±std summaries
    overall_mean_row = overall[overall["stat"] == "mean"].iloc[0]
    overall_std_row  = overall[overall["stat"] == "std"].iloc[0]

    print("\n=== Aggregate metrics (means across folds) ===")
    if exclude_ids:
        print(f"[excluded subjects: {','.join(map(str, exclude_ids))}]")
    _print_overall_summary(overall_mean_row)

    _print_overall_summary_mean_std(overall_mean_row, overall_std_row)

if __name__ == "__main__":
    main()
