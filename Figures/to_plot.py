# plot_rr_corr_seaborn.py
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib import font_manager as fm
import matplotlib.collections as mcoll

# Point this to your real file (you already have it)
FONT_PATH = "/local/home/bdemirel/Projects/HRV/Figures/fonts/Raleway-Regular.ttf"

# Register and get the internal family name
fm.fontManager.addfont(FONT_PATH)
FP = fm.FontProperties(fname=FONT_PATH)
FONT_NAME = FP.get_name()   # e.g., "Raleway" or "Raleway Regular"
print(f"Using local font: {FONT_NAME} from {FONT_PATH}")

# (optional) make it the default globally
mpl.rcParams["font.family"] = FONT_NAME
mpl.rcParams["svg.fonttype"] = "none"


# ---------- Config: residual outlier trimming (print-only) ----------
OUTLIER_MODE   = "mad"   # {"none","mad","iqr","pct","abs"}
OUTLIER_K      = 4.5     # "mad": 1.4826*MAD*k ; "iqr": IQR*k
OUTLIER_PCT    = 1.0     # "pct": trim % from each tail (0..50)
OUTLIER_ABS_MS = None    # "abs": keep |PPG-ECG| <= ABS (ms)

# --- Plotting-only subsample of outliers (avoid diagonal-edge look) ---
OUTLIER_PLOT_KEEP_P = 0.50   # keep this fraction of OUTLIERS in the plot
OUTLIER_PLOT_SEED   = 123    # set None for non-deterministic; else per-subject seeded
# --- Rasterization control (points-only) ---
RASTERIZE_POINTS = True
RASTER_DPI      = 600   # bitmap resolution for the points inside the SVG
RASTER_Z        = 1.5   # anything below this zorder becomes rasterized


# Set Raleway globally (must be installed on your system)
mpl.rcParams["font.family"] = "Raleway"

def _nan_filter(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]

def _fmt_cov(cov: float) -> str:
    if np.isnan(cov):
        return "N/A"
    val = cov * 100.0 if 0.0 <= cov <= 1.5 else cov
    return f"{val:.1f}%"

def load_dataframe(results_dir: str) -> pd.DataFrame:
    """
    Returns a DataFrame with columns: subject, rr_ecg, rr_hat, coverage
    Only uses frames where mask==1 and both values are finite.
    """
    rows = []
    for f in sorted(glob.glob(os.path.join(results_dir, "subject_*.npz"))):
        sid = int(os.path.splitext(os.path.basename(f))[0].split("_")[1])
        z = np.load(f, allow_pickle=True)
        rr_hat = z["rr_hat"].astype(float)
        rr_ecg = z["rr_ecg"].astype(float)
        mask   = z["mask"].astype(bool)
        hat = rr_hat[mask]
        ecg = rr_ecg[mask]
        hat, ecg = _nan_filter(hat, ecg)
        if hat.size == 0:
            continue
        cov = float(z.get("coverage", np.nan)) if "coverage" in z else np.nan
        rows.append(pd.DataFrame({
            "subject": sid,
            "rr_ecg": ecg,
            "rr_hat": hat,
            "coverage": cov
        }))
    if not rows:
        return pd.DataFrame(columns=["subject","rr_ecg","rr_hat","coverage"])
    return pd.concat(rows, ignore_index=True)

def metrics(y_hat: np.ndarray, y_true: np.ndarray):
    if y_hat.size == 0:
        return dict(MAE=np.nan, RMSE=np.nan, r=np.nan, Bias=np.nan)
    diff = y_hat - y_true
    mae  = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    bias = float(np.mean(diff))
    r    = float(np.corrcoef(y_hat, y_true)[0,1]) if y_hat.size > 1 else np.nan
    return dict(MAE=mae, RMSE=rmse, r=r, Bias=bias)

def _ba_stats(y_hat: np.ndarray, y_true: np.ndarray):
    """Return (bias, loa_lo, loa_hi, sd) for Bland–Altman."""
    if y_hat.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    diff = y_hat - y_true
    bias = float(np.mean(diff))
    sd   = float(np.std(diff, ddof=1)) if diff.size > 1 else 0.0
    loa_lo = bias - 1.96 * sd
    loa_hi = bias + 1.96 * sd
    return bias, loa_lo, loa_hi, sd

def _trim_residual_outliers(y_hat, y_true,
                            mode=OUTLIER_MODE, k=OUTLIER_K,
                            pct=OUTLIER_PCT, abs_ms=OUTLIER_ABS_MS):
    """Return (y_hat_trim, y_true_trim, n_removed, detail_str, keep_mask)."""
    y_hat = np.asarray(y_hat, float); y_true = np.asarray(y_true, float)
    y_hat, y_true = _nan_filter(y_hat, y_true)
    n0 = y_hat.size
    if n0 == 0 or mode == "none":
        keep = np.ones(0, dtype=bool) if n0 == 0 else np.ones(n0, dtype=bool)
        return y_hat, y_true, 0, "none", keep

    diff = y_hat - y_true
    keep = np.ones_like(diff, dtype=bool)
    detail = mode

    if mode == "abs" and abs_ms is not None:
        keep = np.abs(diff) <= abs_ms
        detail = f"abs<=±{abs_ms:.0f}ms"

    elif mode == "mad":
        med = np.median(diff)
        mad = np.median(np.abs(diff - med))
        if mad == 0:
            return y_hat, y_true, 0, "mad(no-op)", np.ones_like(diff, bool)
        thr = k * 1.4826 * mad
        keep = np.abs(diff - med) <= thr
        detail = f"mad×{k:g}"

    elif mode == "iqr":
        q1, q3 = np.percentile(diff, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        keep = (diff >= lo) & (diff <= hi)
        detail = f"iqr×{k:g}"

    elif mode == "pct":
        lo, hi = np.percentile(diff, [pct, 100.0 - pct])
        keep = (diff >= lo) & (diff <= hi)
        detail = f"pct {pct:.1f}% tails"

    y_ht = y_hat[keep]; y_tt = y_true[keep]
    removed = int(n0 - y_ht.size)
    return y_ht, y_tt, removed, detail, keep

def _plot_ba_subject(
    y_hat, y_true, out_dir, sid,
    dot_size=8, alpha=0.25, label_size=8, tick_size=9, title_size=11,
    # match correlation function controls
    ai_font_fallback="Arial",
    max_vector_points=2000,      # SVG keeps up to this many vector points
    subsample_seed=12345,
    also_pdf=False,               # if True, save a hybrid PDF too
    rasterize_over=200000,        # rasterize scatter in PDF if > this many points
    pdf_dpi=300
):
    os.makedirs(out_dir, exist_ok=True)

    # Keep text as text (editable) and avoid TeX
    mpl.rcParams.update({
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
    })

    sns.set_theme(style="ticks", context="talk")
    sns.set_context("talk", rc={"axes.titlesize": title_size})

    point_blue = sns.color_palette("deep")[0]
    accent_red = sns.color_palette("flare", n_colors=6)[3]

    y_hat = np.asarray(y_hat, float); y_true = np.asarray(y_true, float)
    y_hat, y_true = _nan_filter(y_hat, y_true)
    if y_hat.size == 0:
        return

    mean_vals = (y_hat + y_true) / 2.0
    diff_vals = y_hat - y_true
    bias, lo, hi, sd = _ba_stats(y_hat, y_true)

    rng = np.random.default_rng(subsample_seed)
    def _maybe_subsample(x, y, k):
        n = len(x)
        if n <= k:
            return x, y
        idx = rng.choice(n, size=k, replace=False)
        return x[idx], y[idx]

    # --- A) SVG (vector, subsampled like correlation plots) ---
    x_svg, y_svg = _maybe_subsample(mean_vals, diff_vals, max_vector_points)

    with mpl.rc_context({
        "font.family": [globals().get("FONT_NAME", ai_font_fallback),
                        ai_font_fallback, "Helvetica", "DejaVu Sans", "sans-serif"],
        "axes.unicode_minus": False,
    }):
        fig, ax = plt.subplots(figsize=(3.5, 2.8))

        # vector points (no rasterize), small alpha for density
        ax.scatter(
            x_svg, y_svg,
            c=[point_blue], s=dot_size, alpha=alpha,
            edgecolors="none", linewidths=0, zorder=1
        )

        # guide lines (vector)
        ax.axhline(bias, ls="-",  c=accent_red, lw=1.2, zorder=3)
        ax.axhline(lo,   ls="--", c=accent_red, lw=1.0, zorder=3)
        ax.axhline(hi,   ls="--", c=accent_red, lw=1.0, zorder=3)

        ax.set_xlabel("Mean RR (ms)", fontsize=label_size)
        ax.set_ylabel("RR difference (PPG − ECG) (ms)", fontsize=label_size)
        ax.tick_params(labelsize=tick_size, length=3, width=0.6)
        for s in ax.spines.values():
            s.set_linewidth(0.6)

        # figure-level annotation (separate from points, easy to edit)
        fig.text(
            0.015, 0.985,
            f"bias={bias:.0f} ms\nLOA={lo:.0f}, {hi:.0f} ms",
            ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none")
        )

        sns.despine(ax=ax)
        fig.tight_layout()

        out_svg = os.path.join(out_dir, f"ba_subject_{int(sid)}.svg")
        fig.savefig(out_svg, format="svg", bbox_inches="tight")  # no dpi for pure vector
        plt.close(fig)
        print(f"saved {out_svg} (vector SVG, {len(x_svg)} points)")

    # --- B) Optional hybrid PDF (keep all points; rasterize only if huge) ---
    if also_pdf:
        rasterize_points = (len(mean_vals) > rasterize_over)
        with mpl.rc_context({
            "font.family": [globals().get("FONT_NAME", ai_font_fallback),
                            ai_font_fallback, "Helvetica", "DejaVu Sans", "sans-serif"],
            "axes.unicode_minus": False,
        }):
            fig, ax = plt.subplots(figsize=(3.5, 2.8))

            sc = ax.scatter(
                mean_vals, diff_vals,
                c=[point_blue], s=dot_size, alpha=alpha,
                edgecolors="none", linewidths=0, zorder=1
            )
            if rasterize_points:
                sc.set_rasterized(True)

            ax.axhline(bias, ls="-",  c=accent_red, lw=1.2, zorder=3)
            ax.axhline(lo,   ls="--", c=accent_red, lw=1.0, zorder=3)
            ax.axhline(hi,   ls="--", c=accent_red, lw=1.0, zorder=3)

            ax.set_xlabel("Mean RR (ms)", fontsize=label_size)
            ax.set_ylabel("RR difference (PPG − ECG) (ms)", fontsize=label_size)
            ax.tick_params(labelsize=tick_size, length=3, width=0.6)
            for s in ax.spines.values():
                s.set_linewidth(0.6)

            fig.text(
                0.015, 0.985,
                f"bias={bias:.0f} ms\nLOA={lo:.0f}, {hi:.0f} ms",
                ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none")
            )

            sns.despine(ax=ax)
            fig.tight_layout()

            out_pdf = os.path.join(out_dir, f"ba_subject_{int(sid)}_print.pdf")
            fig.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=pdf_dpi)
            plt.close(fig)
            print(f"saved {out_pdf} ({'rasterized points' if rasterize_points else 'all vector'})")


# ---------------- Per-subject correlation (+ optional trim) ----------------
def plot_per_subject(
    df: pd.DataFrame, out_dir: str,
    dot_size=8, alpha=0.25,
    title_size=11, label_size=8, tick_size=9,
    ai_font_fallback="Arial",
    max_vector_points=2000,    
    subsample_seed=12345
):

    os.makedirs(out_dir, exist_ok=True)

    # Keep text as text
    mpl.rcParams.update({
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "text.usetex": False,
    })

    sns.set_theme(style="ticks", context="talk")
    sns.set_context("talk", rc={"axes.titlesize": title_size})

    point_blue = sns.color_palette("deep")[0]
    flare_red  = sns.color_palette("flare", n_colors=6)[3]

    rng = np.random.default_rng(subsample_seed)

    def _maybe_subsample(x, y, k):
        n = len(x)
        if n <= k:
            return x, y
        idx = rng.choice(n, size=k, replace=False)
        return x[idx], y[idx]

    for sid, d in df.groupby("subject"):
        # --- trim outliers ---
        yh_raw = d["rr_hat"].values
        yt_raw = d["rr_ecg"].values
        yh_t, yt_t, n_removed, detail, keep_mask = _trim_residual_outliers(
            yh_raw, yt_raw, OUTLIER_MODE, OUTLIER_K, OUTLIER_PCT, OUTLIER_ABS_MS
        )
        used_hat, used_ecg = (yh_t, yt_t) if yh_t.size > 0 else (yh_raw, yt_raw)
        m = metrics(used_hat, used_ecg)

        if keep_mask.size == 0:
            plot_ecg_full, plot_hat_full = used_ecg, used_hat
        else:
            in_idx  = keep_mask
            out_idx = ~keep_mask
            rng_local = np.random.default_rng(OUTLIER_PLOT_SEED + int(sid)) if OUTLIER_PLOT_SEED else np.random.default_rng()
            out_keep = rng_local.random(np.count_nonzero(out_idx)) < OUTLIER_PLOT_KEEP_P
            plot_ecg_full = np.concatenate([yt_raw[in_idx], yt_raw[out_idx][out_keep]])
            plot_hat_full = np.concatenate([yh_raw[in_idx], yh_raw[out_idx][out_keep]])

        # --- subsample for SVG ---
        plot_ecg, plot_hat = _maybe_subsample(plot_ecg_full, plot_hat_full, max_vector_points)

        mn = float(np.nanmin([np.min(plot_ecg_full), np.min(plot_hat_full)]))
        mx = float(np.nanmax([np.max(plot_ecg_full), np.max(plot_hat_full)]))

        with mpl.rc_context({
            "font.family": [globals().get("FONT_NAME", ai_font_fallback),
                            ai_font_fallback, "Helvetica", "DejaVu Sans", "sans-serif"],
            "axes.unicode_minus": False,
        }):
            fig, ax = plt.subplots(figsize=(3.5, 2.8))
            ax.scatter(
                plot_ecg, plot_hat,
                c=[point_blue], s=dot_size, alpha=alpha,
                edgecolors="none", linewidths=0, zorder=1
            )
            ax.plot([mn, mx], [mn, mx], ls="--", c=flare_red, lw=1.2, zorder=3)
            ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)

            ax.set_xlabel("ECG RR (ms)", fontsize=label_size)
            ax.set_ylabel("Estimated RR (ms)", fontsize=label_size)
            ax.tick_params(labelsize=tick_size, length=3, width=0.6)
            for spine in ax.spines.values():
                spine.set_linewidth(0.6)

            fig.text(
                0.015, 0.985,
                f"r={m['r']:.2f}, MAE={m['MAE']:.0f} ms\nRMSE={m['RMSE']:.0f} ms",
                ha="left", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, edgecolor="none")
            )

            sns.despine(ax=ax)
            fig.tight_layout()

            out_svg = os.path.join(out_dir, f"corr_subject_{int(sid)}.svg")
            fig.savefig(out_svg, format="svg", bbox_inches="tight")
            plt.close(fig)
            print(f"saved {out_svg} ({len(plot_ecg)} points kept for SVG)")

        # ---- Bland–Altman per-subject (use EXACTLY the same points as correlation) ----
        _plot_ba_subject(plot_hat_full, plot_ecg_full, out_dir, int(sid),
                        dot_size=dot_size, alpha=alpha, label_size=label_size, tick_size=tick_size)

def main(results_dir: str, out_dir: str = None, per_subject: bool = True):
    out_dir = out_dir or results_dir
    os.makedirs(out_dir, exist_ok=True)
    df = load_dataframe(results_dir)
    if df.empty:
        print("No valid data found. Are your NPZs in:", results_dir, "?")
        return
    if per_subject:
        plot_per_subject(df, out_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True,
                    help="Folder with subject_*.npz from save_subject_results")
    ap.add_argument("--out_dir", type=str, default=None,
                    help="Folder to write plots (defaults to results_dir)")
    ap.add_argument("--no_per_subject", action="store_true",
                    help="Skip per-subject plots")

    # Outlier gate (optional CLI overrides)
    ap.add_argument("--outlier_mode", type=str, default=OUTLIER_MODE,
                    choices=["none","mad","iqr","pct","abs"])
    ap.add_argument("--outlier_k", type=float, default=OUTLIER_K)
    ap.add_argument("--outlier_pct", type=float, default=OUTLIER_PCT)
    ap.add_argument("--outlier_abs_ms", type=float, default=-1)

    args = ap.parse_args()

    # Apply CLI overrides
    OUTLIER_MODE = args.outlier_mode
    OUTLIER_K    = args.outlier_k
    OUTLIER_PCT  = args.outlier_pct
    OUTLIER_ABS_MS = None if args.outlier_abs_ms < 0 else float(args.outlier_abs_ms)

    main(args.results_dir, args.out_dir, per_subject=(not args.no_per_subject))
