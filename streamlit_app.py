from __future__ import annotations
from pathlib import Path
import json, io, zipfile, mimetypes, hashlib
import streamlit as st

# ---------------- basics ----------------
ROOT = Path(__file__).resolve().parent
# search both the repo root and one level up (covers things inside bert_project_starter/)
SEARCH_ROOTS = [ROOT, ROOT.parent]

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

# ---------- small helpers for keys & listing ----------
def _key_for(fp: Path, tag: str = "dl") -> str:
    """Stable unique key per file *and* section."""
    h = hashlib.md5(str(fp.resolve()).encode("utf-8")).hexdigest()[:10]
    return f"{tag}_{h}"

def _list_many(patterns: tuple[str, ...], bases: list[Path]) -> list[Path]:
    """Recursive, de-duplicated listing across bases."""
    seen = {}
    for base in bases:
        for pat in patterns:
            for p in base.rglob(pat):
                seen[str(p.resolve()).lower()] = p
    return sorted(seen.values(), key=lambda p: p.as_posix().lower())

# ---------------- discover runs ----------------
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

# --------------- display helpers ----------------
def safe_download_button(label: str, fp: Path, mime: str | None = None, key_tag: str = "dl"):
    """Serve only if small enough; always provide unique key to avoid DuplicateElementId."""
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
            key=_key_for(fp, key_tag),
        )
    else:
        st.warning(
            f"File is large ({size/1024/1024:.1f} MB). For stability, "
            f"host it externally (Drive/Dropbox) or shrink it before exposing here."
        )

def show_json_file(fp: Path, title: str | None = None, key_tag: str = "json"):
    if fp.exists():
        st.markdown(f"### {title or fp.name}")
        try:
            with fp.open("r", encoding="utf-8") as f:
                st.json(json.load(f))
        except Exception:
            st.code(fp.read_text(encoding="utf-8"))
        safe_download_button(f"Download {fp.name}", fp, "application/json", key_tag=key_tag)

def show_text_file(fp: Path, title: str | None = None, key_tag: str = "txt"):
    if fp.exists():
        st.markdown(f"### {title or fp.name}")
        st.code(fp.read_text(encoding="utf-8"))
        safe_download_button(f"Download {fp.name}", fp, "text/plain", key_tag=key_tag)

def show_image_file(fp: Path, caption: str | None = None, key_tag: str = "img"):
    if fp.exists():
        st.image(str(fp), caption=caption or fp.name, width="stretch")
        safe_download_button(f"Download {fp.name}", fp, key_tag=key_tag)

def show_csv_preview(fp: Path, title: str | None = None, max_rows: int = 300, key_tag: str = "csv"):
    import pandas as pd
    if fp.exists():
        st.markdown(f"### {title or fp.name}")
        df = pd.read_csv(fp)
        total = len(df)
        if total > max_rows:
            st.info(f"Showing first {max_rows} of {total} rows.")
            df = df.head(max_rows)
        try:
            st.dataframe(df, width="stretch")
        except TypeError:
            st.dataframe(df, use_container_width=True)
        safe_download_button(f"Download {fp.name}", fp, "text/csv", key_tag=key_tag)

def list_files(folder: Path, patterns: tuple[str, ...]):
    out = []
    if folder.exists():
        for pat in patterns:
            out += list(folder.glob(pat))
    return sorted(out, key=lambda p: p.name.lower())

# ---------------- run-level: JSON metrics ----------------
for name in ["metrics.json", "test_metrics.json"]:
    show_json_file(run_dir / name, title=name, key_tag="json_run")

# ---------------- run-level: images ----------------
for img in ["confusion_matrix.png", "val_accuracy_curve.png", "val_f1_curve.png", "loss_curve.png"]:
    show_image_file(run_dir / img, caption=img, key_tag="img_run")

# ---------------- run-level: classification report ----------------
show_text_file(run_dir / "classification_report.txt", title="classification_report.txt", key_tag="txt_run")

# ---------------- run-level: predictions table ----------------
show_csv_preview(run_dir / "preds_test.csv", title="preds_test.csv", key_tag="csv_run")

# ---------------- run-level: other json/txt ----------------
with st.expander("Other JSON/TXT artifacts", expanded=False):
    exclude = {"metrics.json", "test_metrics.json", "classification_report.txt"}
    others = [p for p in list_files(run_dir, ("*.json","*.txt")) if p.name not in exclude]
    if not others:
        st.caption("No other JSON/TXT files found in this run.")
    else:
        for fp in others:
            if fp.suffix.lower() == ".json":
                show_json_file(fp, key_tag="json_run_other")
            else:
                show_text_file(fp, key_tag="txt_run_other")

# ---------------- download this run as ZIP (no weights) ----------------
st.markdown("### Download this run (ZIP, excludes model weights)")
if st.button("Build ZIP for current run", key="zip_btn"):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        total = 0
        for p in run_dir.rglob("*"):
            if p.is_dir():
                continue
            # skip heavy HF weights/checkpoints
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
            key="zip_download",
        )

# ================================================================
#                     PROJECT-WIDE ARTIFACTS
# ================================================================
st.header("Project-wide artifacts (root)")

col1, col2 = st.columns(2)

# ----- HTML reports (recursive) -----
with col1:
    st.subheader("HTML reports")
    htmls = _list_many(("report_*.html",), SEARCH_ROOTS)
    if not htmls:
        st.caption("No HTML reports found (report_*.html).")
    else:
        import streamlit.components.v1 as components
        labels_html = [str(p.relative_to(ROOT)) if (p == ROOT or ROOT in p.parents) else p.name for p in htmls]
        pick = st.selectbox("Preview report", labels_html, key="html_pick")
        html_path = htmls[labels_html.index(pick)]
        safe_download_button(f"Download {html_path.name}", html_path, "text/html", key_tag="html_root")
        try:
            components.html(html_path.read_text("utf-8"), height=600, scrolling=True)
        except Exception:
            st.info("HTML too large or contains external refs. Download instead.")

# ----- ZIP reports (recursive) -----
with col2:
    st.subheader("Zipped reports")
    zips = _list_many(("evaluation_report_*.zip",), SEARCH_ROOTS)
    if not zips:
        st.caption("No ZIPs found (evaluation_report_*.zip).")
    else:
        labels_zip = [str(p.relative_to(ROOT)) if (p == ROOT or ROOT in p.parents) else p.name for p in zips]
        pick_zip = st.selectbox("Select ZIP", labels_zip, key="zip_pick")
        zp = zips[labels_zip.index(pick_zip)]
        safe_download_button(f"Download {zp.name}", zp, "application/zip", key_tag="zip_root")
        with st.expander("View ZIP contents"):
            try:
                with zipfile.ZipFile(io.BytesIO(zp.read_bytes()), "r") as zf:
                    names = sorted(zf.namelist())
                st.write("\n".join(names[:300] if len(names) > 300 else names))
                if len(names) > 300:
                    st.caption(f"... and {len(names)-300} more")
            except Exception:
                st.info("ZIP too large to preview in-memory; download instead.")

# ----- Root CSV previews (root-only) -----
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
    show_csv_preview(ROOT / pick_csv, title=pick_csv, key_tag="csv_root")

# ----- Plot galleries (recursive; only dirs that actually contain images) -----
st.subheader("Plot galleries")
all_plot_dirs = _list_many(("plots_*", "eval_plots_*"), SEARCH_ROOTS)
plot_dirs = [
    d for d in all_plot_dirs
    if d.is_dir() and (any(d.glob("*.png")) or any(d.glob("*.jpg")) or any(d.glob("*.jpeg")))
]
if not plot_dirs:
    st.caption("No plot folders (plots_* / eval_plots_*) found.")
else:
    labels_plot = [str(d.relative_to(ROOT)) if (d == ROOT or ROOT in d.parents) else d.name for d in plot_dirs]
    pick_plot_dir = st.selectbox("Pick a plot folder", labels_plot, key="plot_dir_pick")
    pdir = plot_dirs[labels_plot.index(pick_plot_dir)]
    imgs = list_files(pdir, ("*.png","*.jpg","*.jpeg"))
    if not imgs:
        st.caption("No images in selected folder.")
    else:
        cols = st.columns(3)
        for i, img in enumerate(imgs):
            with cols[i % 3]:
                show_image_file(img, key_tag="img_plot")

# ----- Quick folder browser (recursive) -----
st.subheader("Browse any folder")
all_dirs = [ROOT] + [d for d in ROOT.rglob("*") if d.is_dir()]
label_dirs = [str(d.relative_to(ROOT)) for d in all_dirs]
pick_browse = st.selectbox("Folder", label_dirs, key="browse_pick")
browse_dir = all_dirs[label_dirs.index(pick_browse)]

with st.expander(f"Listing: {browse_dir}", expanded=False):
    files = sorted([p for p in browse_dir.iterdir() if p.is_file()], key=lambda p: p.name.lower())
    if not files:
        st.caption("No files in this folder.")
    for fp in files:
        ext = fp.suffix.lower()
        if ext in {".png", ".jpg", ".jpeg"}:
            show_image_file(fp, key_tag="img_browse")
        elif ext == ".csv":
            show_csv_preview(fp, key_tag="csv_browse")
        elif ext == ".json":
            show_json_file(fp, key_tag="json_browse")
        elif ext in {".txt", ".log"}:
            show_text_file(fp, key_tag="txt_browse")
        elif ext == ".html":
            import streamlit.components.v1 as components
            st.markdown(f"#### {fp.name}")
            try:
                components.html(fp.read_text("utf-8"), height=450, scrolling=True)
            except Exception:
                st.info("HTML too large to preview; download instead.")
            safe_download_button(f"Download {fp.name}", fp, "text/html", key_tag="html_browse")
        elif ext == ".zip":
            st.markdown(f"#### {fp.name}")
            safe_download_button(f"Download {fp.name}", fp, "application/zip", key_tag="zip_browse")
        else:
            st.write(f"• {fp.name} ({ext or 'no extension'})")
