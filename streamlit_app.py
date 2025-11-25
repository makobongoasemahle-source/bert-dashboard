from __future__ import annotations
from pathlib import Path
import json
import io
import zipfile
import mimetypes
import streamlit as st

# ---------------- basics ----------------
ROOT = Path(__file__).resolve().parent
SEARCH_ROOTS = [ROOT, ROOT.parent]  # project root + one level up

MAX_INLINE_BYTES = 25 * 1024 * 1024   # 25MB safety limit for inline download_button
MAX_ZIP_BYTES    = 80 * 1024 * 1024   # 80MB safety limit for in-memory ZIPs

st.set_page_config(page_title="BERT Dashboard", layout="wide")
st.title("BERT Dashboard")

st.sidebar.markdown("### About")
st.sidebar.write(
    "• Select a run on the left\n"
    "• Preview metrics, plots, reports\n"
    "• Download light artifacts safely (large files are blocked to avoid memory errors)"
)

def find_run_dirs():
    runs = []
    for base in SEARCH_ROOTS:
        for top in base.glob("bert_runs*"):
            if top.is_dir():
                runs += [d for d in top.rglob("run_*") if d.is_dir()]
    return sorted(set(runs), key=lambda p: p.stat().st_mtime, reverse=True)

run_dirs = find_run_dirs()

with st.expander("Debug info", expanded=False):
    st.write("SEARCH_ROOTS:", [str(x) for x in SEARCH_ROOTS])
    st.write("Found run dirs:", [str(p) for p in run_dirs])

if not run_dirs:
    st.warning("No run_* folders found. After training you should have e.g. bert_runs_weighted/run_YYYYMMDD_HHMM")
    st.stop()

labels = [str(p.relative_to(ROOT)) for p in run_dirs]
choice = st.sidebar.selectbox("Select a run", labels, index=0)
run_dir = run_dirs[labels.index(choice)]
st.subheader(f"Selected: {choice}")

# --------------- helpers ----------------
def safe_download_button(label: str, fp: Path, mime: str | None = None):
    """Only serve if file is reasonably small to avoid MemoryError."""
    if not fp.exists():
        return
    size = fp.stat().st_size
    st.caption(f"{fp.name} — {size/1024:.1f} KB")
    if size <= MAX_INLINE_BYTES:
        st.download_button(
            label,
            data=fp.read_bytes(),
            file_name=fp.name,
            mime=mime or mimetypes.guess_type(fp.name)[0] or "application/octet-stream",
        )
    else:
        st.warning(
            f"File is large ({size/1024/1024:.1f} MB). For stability, "
            f"host it externally (Drive/Dropbox) or shrink it before exposing here."
        )

def show_json_file(fp: Path, title: str | None = None):
    if fp.exists():
        st.markdown(f"### {title or fp.name}")
        try:
            with fp.open("r", encoding="utf-8") as f:
                st.json(json.load(f))
        except Exception:
            st.code(fp.read_text(encoding="utf-8"))
        safe_download_button(f"Download {fp.name}", fp, "application/json")

def show_text_file(fp: Path, title: str | None = None):
    if fp.exists():
        st.markdown(f"### {title or fp.name}")
        st.code(fp.read_text(encoding="utf-8"))
        safe_download_button(f"Download {fp.name}", fp, "text/plain")

def show_image_file(fp: Path, caption: str | None = None):
    if fp.exists():
        st.image(str(fp), caption=caption or fp.name, width="stretch")
        safe_download_button(f"Download {fp.name}", fp)

def show_csv_preview(fp: Path, title: str | None = None, max_rows: int = 300):
    import pandas as pd
    if fp.exists():
        st.markdown(f"### {title or fp.name}")
        df = pd.read_csv(fp)
        total = len(df)
        if total > max_rows:
            st.info(f"Showing first {max_rows} of {total} rows.")
            df = df.head(max_rows)
        st.dataframe(df, use_container_width=True)
        safe_download_button(f"Download {fp.name}", fp, "text/csv")

def list_files(folder: Path, patterns: tuple[str, ...]):
    out = []
    if folder.exists():
        for pat in patterns:
            out += list(folder.glob(pat))
    return sorted(out, key=lambda p: p.name.lower())

# ---------------- run-level: JSON metrics ----------------
for name in ["metrics.json", "test_metrics.json"]:
    show_json_file(run_dir / name, title=name)

# ---------------- run-level: images ----------------
for img in ["confusion_matrix.png", "val_accuracy_curve.png", "val_f1_curve.png", "loss_curve.png"]:
    show_image_file(run_dir / img, caption=img)

# ---------------- run-level: classification report ----------------
show_text_file(run_dir / "classification_report.txt", title="classification_report.txt")

# ---------------- run-level: predictions table ----------------
show_csv_preview(run_dir / "preds_test.csv", title="preds_test.csv")

# ---------------- run-level: other json/txt ----------------
with st.expander("Other JSON/TXT artifacts", expanded=False):
    exclude = {"metrics.json", "test_metrics.json", "classification_report.txt"}
    others = [p for p in list_files(run_dir, ("*.json","*.txt")) if p.name not in exclude]
    if not others:
        st.caption("No other JSON/TXT files found in this run.")
    else:
        for fp in others:
            if fp.suffix.lower() == ".json":
                show_json_file(fp)
            else:
                show_text_file(fp)

# ---------------- download this run as ZIP (no weights) ----------------
st.markdown("### Download this run (ZIP, excludes model weights)")
if st.button("Build ZIP for current run"):
    # Build a temporary in-memory zip without hf_outputs/ or large blobs
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        total = 0
        for p in run_dir.rglob("*"):
            if p.is_dir():
                continue
            # Skip HuggingFace weights / checkpoints
            if "hf_outputs" in p.parts:
                continue
            rel = p.relative_to(run_dir)
            data = p.read_bytes()
            total += len(data)
            if total > MAX_ZIP_BYTES:
                st.error(
                    f"Archive exceeded {MAX_ZIP_BYTES/1024/1024:.0f} MB while adding {rel}. "
                    "Aborting to avoid memory errors. Remove large files or host externally."
                )
                z.close()
                buf.close()
                buf = None
                break
            z.writestr(str(rel), data)
    if buf:
        st.success("ZIP ready.")
        st.download_button(
            "Download run ZIP",
            data=buf.getvalue(),
            file_name=f"{run_dir.name}_light.zip",
            mime="application/zip",
        )

# ================================================================
#                     PROJECT-WIDE ARTIFACTS
# ================================================================
st.header("Project-wide artifacts (root)")

col1, col2 = st.columns(2)

# ----- HTML reports -----
with col1:
    st.subheader("HTML reports")
    htmls = list_files(ROOT, ("report_*.html",))
    if not htmls:
        st.caption("No HTML reports found (report_*.html).")
    else:
        import streamlit.components.v1 as components
        pick = st.selectbox("Preview report", [h.name for h in htmls], key="html_pick")
        html_path = ROOT / pick
        safe_download_button(f"Download {pick}", html_path, "text/html")
        # Inline preview
        try:
            components.html(html_path.read_text("utf-8"), height=600, scrolling=True)
        except Exception:
            st.info("HTML too large or contains external refs. Download instead.")

# ----- ZIP reports -----
with col2:
    st.subheader("Zipped reports")
    zips = list_files(ROOT, ("evaluation_report_*.zip",))
    if not zips:
        st.caption("No ZIPs found (evaluation_report_*.zip).")
    else:
        pick_zip = st.selectbox("Select ZIP", [z.name for z in zips], key="zip_pick")
        zp = ROOT / pick_zip
        safe_download_button(f"Download {pick_zip}", zp, "application/zip")
        with st.expander("View ZIP contents"):
            try:
                with zipfile.ZipFile(io.BytesIO(zp.read_bytes()), "r") as zf:
                    names = sorted(zf.namelist())
                st.write("\n".join(names[:300] if len(names) > 300 else names))
                if len(names) > 300:
                    st.caption(f"... and {len(names)-300} more")
            except Exception:
                st.info("ZIP too large to preview in-memory; download instead.")

# ----- Root CSV previews -----
st.subheader("Dataset & evaluation CSVs")
root_csvs = list_files(
    ROOT,
    (
        "test_with_preds*.csv",
        "test_with_probs*.csv",
        "pred_counts*.csv",
        "confusion_from_csv*.csv",
        "new_with_preds*.csv",
        "new_with_probs*.csv",
        "test.csv",
        "val.csv",
        "train.csv",
    ),
)
if not root_csvs:
    st.caption("No matching CSVs found.")
else:
    pick_csv = st.selectbox("Pick a CSV to preview", [c.name for c in root_csvs], key="csv_pick")
    show_csv_preview(ROOT / pick_csv, title=pick_csv)

# ----- Plot galleries -----
st.subheader("Plot galleries")
plot_dirs = [d for d in ROOT.iterdir() if d.is_dir() and d.name.startswith(("plots_", "eval_plots_"))]
if not plot_dirs:
    st.caption("No plot folders (plots_* / eval_plots_*) found.")
else:
    pick_plot_dir = st.selectbox("Pick a plot folder", [d.name for d in plot_dirs], key="plot_dir_pick")
    pdir = ROOT / pick_plot_dir
    imgs = list_files(pdir, ("*.png","*.jpg","*.jpeg"))
    if not imgs:
        st.caption("No images in selected folder.")
    else:
        cols = st.columns(3)
        for i, img in enumerate(imgs):
            with cols[i % 3]:
                show_image_file(img)

# ----- Quick folder browser (advanced) -----
st.subheader("Browse any folder")
all_dirs = [ROOT] + [d for d in ROOT.iterdir() if d.is_dir()]
pick_browse = st.selectbox("Folder", [str(d.relative_to(ROOT)) for d in all_dirs], key="browse_pick")
browse_dir = ROOT / pick_browse

with st.expander(f"Listing: {browse_dir}", expanded=False):
    files = sorted([p for p in browse_dir.iterdir() if p.is_file()], key=lambda p: p.name.lower())
    if not files:
        st.caption("No files in this folder.")
    for fp in files:
        ext = fp.suffix.lower()
        if ext in {".png", ".jpg", ".jpeg"}:
            show_image_file(fp)
        elif ext == ".csv":
            show_csv_preview(fp)
        elif ext == ".json":
            show_json_file(fp)
        elif ext in {".txt", ".log"}:
            show_text_file(fp)
        elif ext == ".html":
            import streamlit.components.v1 as components
            st.markdown(f"#### {fp.name}")
            try:
                components.html(fp.read_text("utf-8"), height=450, scrolling=True)
            except Exception:
                st.info("HTML too large to preview; download instead.")
            safe_download_button(f"Download {fp.name}", fp, "text/html")
        elif ext == ".zip":
            st.markdown(f"#### {fp.name}")
            safe_download_button(f"Download {fp.name}", fp, "application/zip")
        else:
            st.write(f"• {fp.name} ({ext or 'no extension'})")
