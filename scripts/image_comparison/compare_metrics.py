import pandas as pd
from pathlib import Path

root = Path("hq")  # contains subfolders with same-named CSVs
out  = root / "comparison"
out.mkdir(exist_ok=True)
metrics = ["psnr", "ssim", "dists", "niqe"]

files_by_name = {}

for sub in [p for p in root.iterdir() if p.is_dir()]:
    for csv_path in sub.glob("*.csv"):
        files_by_name.setdefault(csv_path.name, []).append((sub.name, csv_path))

for name, entries in files_by_name.items():
    dfs = []
    averages = {}
    for parent, csv_path in entries:
        df = pd.read_csv(csv_path)
        df = df.set_index("image")
        df.columns = [f"{c}_{parent}" for c in df.columns]
        dfs.append(df)
    merged = pd.concat(dfs, axis=1).reset_index()  # outer join by default
    merged.to_csv(out / name, index=False)
    for metric in metrics:
        metric_df = merged[[col for col in merged.columns if metric in col or col == "image"]]
        metric_df.to_csv(out / f"{metric}.csv", index=False)
        for col in merged.columns:
            if col == "image":
                continue
            if col.startswith(metric + "_"):
                method = col[len(metric) + 1:]
                averages.setdefault(method, {})[metric] = merged.loc[merged["image"].eq("AVERAGE"), col].iloc[0]
    averages = pd.DataFrame.from_dict(averages, orient="index").reindex(columns=metrics)
    averages.to_csv(out / f"averages.csv", index=True)