import glob
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd
import torch
from DISTS_pytorch import DISTS
from PIL import Image
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


# ---------------- MATLAB NIQE helpers (Windows-friendly) ----------------


class MatlabNIQE:
    """
    Batch-only NIQE via MATLAB.
    - Prefers MATLAB Engine (one session, one call).
    - Otherwise, runs a single CLI batch (one process, one call).
    Requires Image Processing Toolbox (function `niqe`).
    """
    def __init__(self,
                 matlab_cmd=r"C:\Program Files\MATLAB\R2025a\bin\matlab.exe",
                 prefer_engine=True,
                 launch_timeout=300):
        self.mode = None
        self.matlab_cmd = matlab_cmd
        self.launch_timeout = int(launch_timeout)
        self.eng = None

        if prefer_engine:
            try:
                import matlab.engine  # type: ignore
                self.eng = matlab.engine.start_matlab()
                # check `niqe` exists
                if int(self.eng.eval("exist('niqe','file')", nargout=1)) > 0:
                    self.mode = "engine"
                    return
            except Exception:
                self.mode = None

        # CLI batch mode: smoke test quickly
        try:
            proc = subprocess.run(
                [self.matlab_cmd, "-batch", "disp(version)"],
                capture_output=True, text=True, timeout=self.launch_timeout
            )
            if proc.returncode == 0:
                self.mode = "cli-batch"
            else:
                raise RuntimeError(proc.stderr or "matlab -batch failed")
        except Exception as e:
            raise RuntimeError(
                "Could not initialize MATLAB (engine or CLI). "
                "Set matlab_cmd to your matlab.exe and ensure Image Processing Toolbox is installed."
            ) from e

    def batch(self, img_paths):
        """
        Compute NIQE for all img_paths in one MATLAB run.
        Returns list of floats aligned to img_paths (NaN on failure).
        """
        if not img_paths:
            return []

        # normalize paths for MATLAB
        norm = [p.replace("\\", "/").replace("'", "''") for p in img_paths]

        if self.mode == "engine":
            # define a small MATLAB func and call it
            self.eng.eval("""
                function scores = _niqe_batch(paths)
                    n = numel(paths); scores = NaN(n,1);
                    for i = 1:n
                        try
                            img = imread(paths{i});
                            scores(i) = niqe(img);
                        catch
                            scores(i) = NaN;
                        end
                    end
                end
            """, nargout=0)
            scores = self.eng._niqe_batch(norm, nargout=1)
            return [float(s) for s in list(scores)]

        # CLI batch: write one script, run once, read CSV
        with tempfile.TemporaryDirectory() as td:
            mfile = os.path.join(td, "run_batch.m")
            td_run = td.replace("\\", "/")
            with open(mfile, "w", encoding="utf-8") as f:
                f.write("paths = {\n")
                for i, p in enumerate(norm):
                    f.write(f"'{p}'")
                    f.write(",\n" if i < len(norm)-1 else "\n")
                f.write("};\n")
                f.write("""
                    n = numel(paths); scores = NaN(n,1);
                    for i=1:n
                        try
                            img = imread(paths{i});
                            scores(i) = niqe(img);
                        catch
                            scores(i) = NaN;
                        end
                    end
                    fid = fopen('niqe_out.csv','w');
                    for i=1:n, fprintf(fid,'%.15g\\n',scores(i)); end
                    fclose(fid);
                """)
            proc = subprocess.run(
                [self.matlab_cmd, "-batch",
                 f"cd('{td_run}'); run('run_batch.m')"],
                capture_output=True, text=True, timeout=max(self.launch_timeout, 900)
            )
            if proc.returncode != 0:
                tail = (proc.stdout or "") + "\n" + (proc.stderr or "")
                raise RuntimeError("MATLAB batch NIQE failed:\n" + tail[-1000:])

            out_csv = os.path.join(td, "niqe_out.csv")
            with open(out_csv, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f.readlines()]

            def to_float(x):
                try: return float(x)
                except: return float('nan')

            return [to_float(x) for x in lines]

def _load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

def _resize_like(img, w, h):
    return np.asarray(
        Image.fromarray((img*255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
        dtype=np.float32
    )/255.0

def evaluate_metrics(hq_dir, gt_dir, iteration=None, ext="png", save_csv=None):
    print(f"Evaluating metrics for: \nHQ: {hq_dir} \nGT: {gt_dir}")
    gt_paths = sorted(glob.glob(os.path.join(gt_dir, f"*.{ext}")))
    print(f"Found {len(gt_paths)} GT files.")
    if not gt_paths:
        raise FileNotFoundError(f"No GT files like in {gt_dir}")

    # Find processed paths first (so we can batch NIQE)
    items = []
    for gt_path in gt_paths:
        name = os.path.basename(gt_path).rsplit(".", 1)[0].replace("_gt", "")
        # adjust to your layout; here it's per-image subfolder:
        image_name = f"{name}_{iteration}.{ext}" if iteration else f"{name}.{ext}"
        hq_path = os.path.join(hq_dir, name, image_name) if iteration else os.path.join(hq_dir, image_name)
        if not os.path.isfile(hq_path):
            # recursive fallback; comment out if not needed
            cands = glob.glob(os.path.join(hq_dir, "**", image_name), recursive=True)
            hq_path = cands[0] if cands else None
        if hq_path:
            items.append((name, gt_path, hq_path))

    if not items:
        raise RuntimeError("No processed images matched; check iteration/layout.")

    # ---- Batch NIQE once
    niqe_matlab = MatlabNIQE(
        matlab_cmd=r"C:\Program Files\MATLAB\R2025a\bin\matlab.exe",
        prefer_engine=True,
        launch_timeout=300
    )
    proc_paths = [hq_path for (_, _, hq_path) in items]
    print(f"Computing NIQE in MATLAB for {len(proc_paths)} images (single batch)...")
    niqe_vals = niqe_matlab.batch(proc_paths)
    niqe_by_path = dict(zip(proc_paths, niqe_vals))

    # ---- Other metrics
    dists_model = DISTS().eval()
    rows = []
    for name, gt_path, hq_path in items:
        gt = _load_rgb(gt_path)
        proc = _load_rgb(hq_path)
        if proc.shape != gt.shape:
            proc = _resize_like(proc, gt.shape[1], gt.shape[0])

        psnr = float(peak_signal_noise_ratio(gt, proc, data_range=1.0))
        try:
            ssim = float(structural_similarity(gt, proc, channel_axis=2, data_range=1.0))
        except TypeError:
            ssim = float(structural_similarity(gt, proc, multichannel=True, data_range=1.0))

        with torch.no_grad():
            t1 = torch.from_numpy(gt.transpose(2, 0, 1)).unsqueeze(0).float()
            t2 = torch.from_numpy(proc.transpose(2, 0, 1)).unsqueeze(0).float()
            dists = float(dists_model(t1, t2).item())

        rows.append({
            "image": name,
            "psnr": psnr,
            "ssim": ssim,
            "dists": dists,
            "niqe": float(niqe_by_path.get(hq_path, float('nan')))
        })

    print(f"Done. Evaluated {len(rows)} images.")

    df = pd.DataFrame(rows).sort_values(["image"])
    avg_metrics = df[["psnr", "ssim", "dists", "niqe"]].mean(numeric_only=True)
    avg_row = {"image": "AVERAGE",
               "psnr": avg_metrics.get("psnr", np.nan),
               "ssim": avg_metrics.get("ssim", np.nan),
               "dists": avg_metrics.get("dists", np.nan),
               "niqe": avg_metrics.get("niqe", np.nan)}

    out = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

    if save_csv:
        out.to_csv(save_csv, index=False)
    return out

table = evaluate_metrics(hq_dir=r"",
                         gt_dir=r"",
                         save_csv=r"")
print(table)
