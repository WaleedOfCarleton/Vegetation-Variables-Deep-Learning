from __future__ import annotations

import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from tkinter import filedialog, messagebox

import torch
from PIL import Image, ImageTk

# Allow `from pai_cnn...` when invoked as `python ml/pai/gui_predict_pai.py`
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pai_cnn.common import build_model, build_transforms  # noqa: E402


@dataclass
class ModelState:
    checkpoint_path: Path | None = None
    model: torch.nn.Module | None = None
    device: torch.device | None = None
    img_size: int = 224
    pretrained: bool = False
    target_name: str = "PAI"


class PaiGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("HemiPy — CNN Predictor")

        # .../repo/ml/pai/gui_predict_pai.py
        self.repo_root = Path(__file__).resolve().parents[2]

        self.state = ModelState()
        self._preview_imgtk: ImageTk.PhotoImage | None = None
        self.selected_image_path: Path | None = None
        self.selected_folder_path: Path | None = None
        self._is_predicting = False

        self._build_ui()

    def _build_ui(self) -> None:
        frm = tk.Frame(self.root, padx=12, pady=12)
        frm.pack(fill=tk.BOTH, expand=True)

        title = tk.Label(frm, text="CNN prediction from image(s)", font=("Segoe UI", 14, "bold"))
        title.grid(row=0, column=0, columnspan=3, sticky="w")

        self.lbl_target = tk.Label(frm, text="Target: (not loaded)")
        self.lbl_target.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self.lbl_device = tk.Label(frm, text="Device: (not loaded)")
        self.lbl_device.grid(row=2, column=0, columnspan=3, sticky="w", pady=(6, 8))

        btn_ckpt = tk.Button(frm, text="Load checkpoint (.pt)", command=self.on_load_checkpoint)
        btn_ckpt.grid(row=3, column=0, sticky="w")

        self.lbl_ckpt = tk.Label(frm, text="Checkpoint: (none)")
        self.lbl_ckpt.grid(row=3, column=1, columnspan=2, sticky="w")

        btn_img = tk.Button(frm, text="Choose image", command=self.on_choose_image)
        btn_img.grid(row=4, column=0, sticky="w", pady=(10, 0))

        btn_dir = tk.Button(frm, text="Choose folder (average)", command=self.on_choose_folder)
        btn_dir.grid(row=4, column=1, sticky="w", pady=(10, 0), padx=(10, 0))

        self.btn_predict = tk.Button(frm, text="Predict", command=self.on_predict)
        self.btn_predict.grid(row=4, column=2, sticky="e", pady=(10, 0))

        self.lbl_img = tk.Label(frm, text="Image: (none)")
        self.lbl_img.grid(row=5, column=0, columnspan=3, sticky="w", pady=(10, 0))

        self.lbl_status = tk.Label(frm, text="Status: idle")
        self.lbl_status.grid(row=6, column=0, columnspan=3, sticky="w", pady=(12, 0))

        self.lbl_pred = tk.Label(frm, text="Prediction: (none)", font=("Segoe UI", 12, "bold"))
        self.lbl_pred.grid(row=7, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self.preview = tk.Label(frm)
        self.preview.grid(row=8, column=0, columnspan=3, sticky="w", pady=(10, 0))

        note = (
            "Notes: Load a checkpoint produced by the CNN training scripts (PAI or clumping). "
            "Prediction is a single-image estimate; for best stability, use the folder option to average multiple images."
        )
        self.lbl_note = tk.Label(frm, text=note, wraplength=700, justify="left", fg="#444")
        self.lbl_note.grid(row=9, column=0, columnspan=3, sticky="w", pady=(10, 0))

        hint = (
            "Tip: The folder option works for a single case or a parent folder containing multiple Case */ subfolders; "
            "images are averaged per case and also across all cases."
        )
        self.lbl_hint = tk.Label(frm, text=hint, wraplength=700, justify="left", fg="#444")
        self.lbl_hint.grid(row=10, column=0, columnspan=3, sticky="w", pady=(6, 0))

        frm.grid_columnconfigure(1, weight=1)
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        if self._is_predicting:
            self.btn_predict.config(state=tk.DISABLED)
            return

        can_predict = self._model_loaded() and (
            self.selected_image_path is not None or self.selected_folder_path is not None
        )
        self.btn_predict.config(state=(tk.NORMAL if can_predict else tk.DISABLED))

    def _set_status(self, text: str) -> None:
        self.lbl_status.config(text=f"Status: {text}")

    def _set_prediction_text(self, text: str) -> None:
        self.lbl_pred.config(text=text)

    @staticmethod
    def _target_name_from_checkpoint_config(cfg: dict) -> str:
        # PAI checkpoints: target_col=truth_PAI
        # Clumping checkpoints (truth): target_col=truth_Clumping
        # Hinge-clumping checkpoints: target=omega_hinge
        target_col = cfg.get("target_col")
        if target_col == "truth_PAI":
            return "PAI"
        if target_col == "truth_Clumping":
            return "Clumping"

        target = cfg.get("target")
        if target == "omega_hinge":
            return "Clumping (hinge)"

        return "Value"

    def _set_preview_from_path(self, img_path: Path) -> None:
        pil = Image.open(img_path).convert("RGB")
        preview = pil.copy()
        preview.thumbnail((512, 512))
        self._preview_imgtk = ImageTk.PhotoImage(preview)
        self.preview.config(image=self._preview_imgtk)

    @staticmethod
    def _list_image_files(folder: Path) -> list[Path]:
        exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        files: list[Path] = []
        for p in folder.rglob("*"):
            if not p.is_file():
                continue
            if p.name.lower() == "thumbs.db":
                continue
            if p.suffix.lower() in exts:
                files.append(p)
        return sorted(files)

    @staticmethod
    def _group_images_by_case(folder: Path) -> dict[str, list[Path]]:
        """Treat each immediate subfolder as a case; gather images inside it."""

        groups: dict[str, list[Path]] = {}
        for sub in sorted(folder.iterdir()):
            if not sub.is_dir():
                continue
            imgs = PaiGui._list_image_files(sub)
            if imgs:
                groups[sub.name] = imgs
        return groups

    def _predict_pil(self, pil: Image.Image) -> float:
        tf = build_transforms(self.state.img_size, train=False)
        x = tf(pil).unsqueeze(0).to(self.state.device)
        with torch.no_grad():
            pred = self.state.model(x).squeeze().detach().cpu().item()
        return float(pred)

    def _model_loaded(self) -> bool:
        return self.state.model is not None and self.state.device is not None

    def on_load_checkpoint(self) -> None:
        initial = self.repo_root / "ml" / "runs"
        path = filedialog.askopenfilename(
            title="Select model checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*")],
            initialdir=str(initial) if initial.exists() else None,
        )
        if not path:
            return

        ckpt_path = Path(path)
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            cfg = ckpt.get("config", {})
            img_size = int(cfg.get("img_size", 224))
            pretrained = bool(cfg.get("pretrained", False))
            target_name = self._target_name_from_checkpoint_config(cfg)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = build_model(pretrained=pretrained)
            model.load_state_dict(ckpt["model_state"], strict=True)
            model.to(device)
            model.eval()

            self.state = ModelState(
                checkpoint_path=ckpt_path,
                model=model,
                device=device,
                img_size=img_size,
                pretrained=pretrained,
                target_name=target_name,
            )

            dev_str = "cuda" if device.type == "cuda" else "cpu"
            if device.type == "cuda":
                dev_str += f" ({torch.cuda.get_device_name(0)})"

            self.lbl_device.config(text=f"Device: {dev_str}")
            self.lbl_target.config(text=f"Target: {target_name}")
            self.lbl_ckpt.config(text=f"Checkpoint: {ckpt_path}")
            self._set_status("checkpoint loaded")
            self._set_prediction_text(f"Prediction ({target_name}): (select image/folder, then click Predict)")
            self._refresh_controls()

        except Exception as exc:
            messagebox.showerror("Failed to load checkpoint", str(exc))

    def on_choose_image(self) -> None:
        initial = self.repo_root / "Simulations"
        path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("All files", "*.*"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("JPEG", "*.jpeg"),
                ("TIFF", "*.tif"),
                ("TIFF", "*.tiff"),
                ("Bitmap", "*.bmp"),
            ],
            initialdir=str(initial) if initial.exists() else None,
        )
        if not path:
            return

        img_path = Path(path)
        self.selected_image_path = img_path
        self.selected_folder_path = None
        self.lbl_img.config(text=f"Image: {img_path}")

        if not self._model_loaded():
            self._set_status("select a checkpoint")
            self._set_prediction_text("Prediction: load a checkpoint, then click Predict")
        else:
            self._set_status("ready")
            self._set_prediction_text(f"Prediction ({self.state.target_name}): click Predict to compute")
            self._set_preview_from_path(img_path)

        self._refresh_controls()

    def on_choose_folder(self) -> None:
        initial = self.repo_root / "Simulations"
        path = filedialog.askdirectory(
            title="Select a folder of images",
            initialdir=str(initial) if initial.exists() else None,
        )
        if not path:
            return

        folder = Path(path)
        self.selected_folder_path = folder
        self.selected_image_path = None

        if not self._model_loaded():
            self.lbl_img.config(text=f"Folder: {folder} (load checkpoint, then Predict)")
            self._set_status("select a checkpoint")
            self._set_prediction_text("Prediction: load a checkpoint, then click Predict")
        else:
            self.lbl_img.config(text=f"Folder: {folder}")
            self._set_status("ready")
            self._set_prediction_text(f"Prediction ({self.state.target_name}): click Predict to compute")

        self._refresh_controls()

    def on_predict(self) -> None:
        if self._is_predicting:
            return

        if not self._model_loaded():
            messagebox.showwarning("No model loaded", "Load a checkpoint first.")
            return

        if self.selected_image_path is None and self.selected_folder_path is None:
            messagebox.showwarning("Nothing selected", "Select an image or folder first.")
            return

        self._is_predicting = True
        self._set_status("predicting…")
        self._set_prediction_text(f"Prediction ({self.state.target_name}): (running…)")
        self._refresh_controls()
        self.root.update_idletasks()

        img_path = self.selected_image_path
        folder_path = self.selected_folder_path

        def worker() -> None:
            try:
                if img_path is not None:
                    pil = Image.open(img_path).convert("RGB")
                    pred = self._predict_pil(pil)
                    result_text = f"Prediction ({self.state.target_name}): {pred:.4f}"
                    message_text = result_text
                    preview_path = img_path
                else:
                    assert folder_path is not None
                    case_groups = self._group_images_by_case(folder_path)
                    use_case_groups = len(case_groups) >= 2

                    if not use_case_groups:
                        files = self._list_image_files(folder_path)
                        if not files:
                            raise RuntimeError("No supported images found in selected folder (png/jpg/tif/bmp).")

                        preds: list[float] = []
                        for i, fp in enumerate(files, start=1):
                            pil = Image.open(fp).convert("RGB")
                            preds.append(self._predict_pil(pil))
                            if i % 50 == 0:
                                self.root.after(0, lambda i=i, n=len(files): self._set_status(f"predicting… {i}/{n}"))

                        mu = mean(preds)
                        sigma = pstdev(preds) if len(preds) > 1 else 0.0
                        result_text = (
                            f"Prediction ({self.state.target_name}): mean={mu:.4f} | std={sigma:.4f} | n_images={len(preds)}"
                        )
                        message_text = result_text
                        preview_path = files[0]
                    else:
                        case_preds: dict[str, list[float]] = {}
                        total_imgs = 0
                        for case, imgs in case_groups.items():
                            preds: list[float] = []
                            for i, fp in enumerate(imgs, start=1):
                                pil = Image.open(fp).convert("RGB")
                                preds.append(self._predict_pil(pil))
                                total_imgs += 1
                                if total_imgs % 50 == 0:
                                    self.root.after(
                                        0,
                                        lambda i=total_imgs, n=sum(len(v) for v in case_groups.values()): self._set_status(
                                            f"predicting… {i}/{n}"
                                        ),
                                    )
                            case_preds[case] = preds

                        case_means = {c: mean(ps) for c, ps in case_preds.items()}
                        mu_cases = mean(case_means.values())
                        sigma_cases = pstdev(case_means.values()) if len(case_means) > 1 else 0.0
                        result_text = (
                            f"Prediction ({self.state.target_name}): mean_case={mu_cases:.4f} | std_case={sigma_cases:.4f} "
                            f"| n_cases={len(case_means)} | n_images={total_imgs}"
                        )

                        detail_lines: list[str] = []
                        for case in list(sorted(case_means.keys()))[:10]:
                            detail_lines.append(
                                f"{case}: mean={case_means[case]:.4f} (n={len(case_preds[case])})"
                            )
                        if len(case_means) > 10:
                            detail_lines.append(f"... (+{len(case_means) - 10} more cases)")

                        message_text = result_text + ("\n\n" + "\n".join(detail_lines) if detail_lines else "")
                        first_case = next(iter(case_preds.keys()))
                        preview_path = case_groups[first_case][0]

                def done() -> None:
                    self._set_preview_from_path(preview_path)
                    self._set_prediction_text(result_text)
                    self._set_status("done")
                    self._is_predicting = False
                    self._refresh_controls()
                    messagebox.showinfo("Prediction", message_text)

                self.root.after(0, done)

            except Exception as exc:
                def fail() -> None:
                    self._is_predicting = False
                    self._set_status("error")
                    self._set_prediction_text("Prediction: (failed)")
                    self._refresh_controls()
                    messagebox.showerror("Prediction failed", str(exc))

                self.root.after(0, fail)

        threading.Thread(target=worker, daemon=True).start()


def main() -> int:
    root = tk.Tk()
    _app = PaiGui(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
