from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Optional, Sequence

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

# Allow `from pai_cnn...` when run as `uvicorn ml.pai.web_predict_pai:app`
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pai_cnn.common import build_model, build_transforms  # noqa: E402


app = FastAPI(title="HemiPy Web Predictor", description="PAI / Clumping inference via browser")


MODEL_CACHE: dict[tuple[Path, str], tuple[torch.nn.Module, int]] = {}


@app.on_event("startup")
def _log_startup() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = os.environ.get("PORT", "8000")
    display_host = "localhost" if host in {"0.0.0.0", "::"} else host
    print(f"HemiPy Web Predictor ready: http://{display_host}:{port}/")


def _load_model(ckpt_path: Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    key = (ckpt_path.resolve(), device.type)
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    if not ckpt_path.exists():
        raise HTTPException(status_code=400, detail=f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    img_size = int(cfg.get("img_size", 224))
    pretrained = bool(cfg.get("pretrained", False))

    model = build_model(pretrained=pretrained)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.to(device)
    model.eval()

    MODEL_CACHE[key] = (model, img_size)
    return model, img_size


def _predict_single(model: torch.nn.Module, img_size: int, file_bytes: bytes, device: torch.device) -> float:
    pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    tf = build_transforms(img_size, train=False)
    x = tf(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).squeeze().detach().cpu().item()
    return float(pred)


def _find_checkpoints(root: Path) -> list[Path]:
    """Return sorted list of candidate checkpoints under root/ml/runs."""
    runs_root = root / "ml" / "runs"
    if not runs_root.exists():
        return []

    patterns = ["**/model_best*.pt", "**/model_last.pt", "**/model.pt"]
    found: set[Path] = set()
    for pat in patterns:
        for p in runs_root.glob(pat):
            if p.is_file():
                found.add(p.resolve())
    return sorted(found)


def _render_html_form(message: str = "", *, ckpts: Sequence[Path] = ()) -> HTMLResponse:
    note = ("Provide checkpoint paths that are accessible on the server. "
            "Upload one or more images; predictions return as a table and a CSV download.")

    options = "".join(f"<option value='{p}'>" for p in ckpts)
    html = f"""
    <html><body>
    <h2>HemiPy Web Predictor</h2>
    <p>{note}</p>
    <form action="/predict" method="post" enctype="multipart/form-data">
      <label>PAI checkpoint (.pt, required):</label><br>
      <input list="pai_ckpts" type="text" name="pai_checkpoint" size="80" required>
      <datalist id="pai_ckpts">{options}</datalist><br><br>
      <label>Clumping checkpoint (.pt, optional):</label><br>
      <input list="clump_ckpts" type="text" name="clumping_checkpoint" size="80">
      <datalist id="clump_ckpts">{options}</datalist><br><br>
      <label>Images (png/jpg/tif):</label><br>
      <input type="file" name="images" multiple required><br><br>
      <input type="submit" value="Run predictions">
    </form>
    <p style="color:red;">{message}</p>
    </body></html>
    """
    return HTMLResponse(content=html)


def _html_results(rows: list[dict], csv_content: str) -> HTMLResponse:
    # Build data URI for CSV download
    b64 = base64.b64encode(csv_content.encode("utf-8")).decode("ascii")
    link = f"data:text/csv;base64,{b64}"

    def _fmt_clumping(val: Optional[float]) -> str:
        return "" if val is None else f"{val:.4f}"

    table_rows = "".join(
        f"<tr><td>{r['filename']}</td><td>{r['pred_pai']:.4f}</td><td>{_fmt_clumping(r['pred_clumping'])}</td></tr>"
        for r in rows
    )
    html = f"""
    <html><body>
    <h3>Predictions</h3>
    <p><a download="predictions.csv" href="{link}">Download CSV</a></p>
    <table border="1" cellpadding="6" cellspacing="0">
      <tr><th>Filename</th><th>Pred PAI</th><th>Pred Clumping</th></tr>
      {table_rows}
    </table>
    <p><a href="/">Back</a></p>
    </body></html>
    """
    return HTMLResponse(content=html)


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    ckpts = _find_checkpoints(Path(__file__).resolve().parents[2])
    return _render_html_form(ckpts=ckpts)


@app.post("/predict", response_class=HTMLResponse)
def predict(
    pai_checkpoint: str = Form(...),
    clumping_checkpoint: Optional[str] = Form(None),
    images: list[UploadFile] = File(...),
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpts = _find_checkpoints(Path(__file__).resolve().parents[2])

    try:
        pai_model, pai_img_size = _load_model(Path(pai_checkpoint), device)
    except Exception as exc:  # pragma: no cover
        return _render_html_form(message=f"Failed to load PAI checkpoint: {exc}", ckpts=ckpts)

    clumping_model = None
    clumping_img_size = None
    if clumping_checkpoint:
        try:
            clumping_model, clumping_img_size = _load_model(Path(clumping_checkpoint), device)
        except Exception as exc:  # pragma: no cover
            return _render_html_form(message=f"Failed to load clumping checkpoint: {exc}", ckpts=ckpts)

    rows = []
    for uf in images:
        data = uf.file.read()
        if not data:
            continue
        try:
            pred_pai = _predict_single(pai_model, pai_img_size, data, device)
            pred_clumping = None
            if clumping_model is not None and clumping_img_size is not None:
                pred_clumping = _predict_single(clumping_model, clumping_img_size, data, device)
            rows.append({"filename": uf.filename, "pred_pai": pred_pai, "pred_clumping": pred_clumping})
        except Exception as exc:  # pragma: no cover
            return _render_html_form(message=f"Failed on {uf.filename}: {exc}", ckpts=ckpts)

    if not rows:
        return _render_html_form(message="No images processed.", ckpts=ckpts)

    # Build CSV content
    csv_lines = ["filename,pred_pai,pred_clumping"]
    for r in rows:
        csv_lines.append(f"{r['filename']},{r['pred_pai']},{'' if r['pred_clumping'] is None else r['pred_clumping']}")
    csv_content = "\n".join(csv_lines)

    return _html_results(rows, csv_content)


if __name__ == "__main__":
    # Convenience for local debugging: `python ml/pai/web_predict_pai.py`
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("ml.pai.web_predict_pai:app", host=host, port=port, reload=False)
