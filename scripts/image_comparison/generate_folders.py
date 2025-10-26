from pathlib import Path
import shutil

# INPUTS
pth = Path(r"")
iteration_number = 100000
out_dir = Path(r"")  # <- change this

# Create output dirs
gt_dir = out_dir / "gt"
lq_dir = out_dir / "lq"
gt_dir.mkdir(parents=True, exist_ok=True)
lq_dir.mkdir(parents=True, exist_ok=True)

gt_files = list(pth.rglob(f"*PANSHARP*_{iteration_number}.png"))
lq_files = list(pth.rglob("*PANSHARP*_lq.png"))

def next_free_path(dest: Path) -> Path:
    """If dest exists, append _1, _2, ... before the suffix to avoid overwrite."""
    if not dest.exists():
        return dest
    stem, suf = dest.stem, dest.suffix
    i = 1
    while True:
        candidate = dest.with_name(f"{stem}_{i}{suf}")
        if not candidate.exists():
            return candidate
        i += 1

# Copy + rename GT: remove `_q{iteration_number}` (preferred) or fallback `_{iteration_number}`
for src in gt_files:
    if not src.is_file():
        continue
    stem, suf = src.stem, src.suffix
    new_stem = stem.replace(f"_q{iteration_number}", "")
    if new_stem == stem:
        new_stem = stem.replace(f"_{iteration_number}", "")  # fallback
    dest = next_free_path(gt_dir / f"{new_stem}{suf}")
    shutil.copy2(src, dest)

# Copy + rename LQ: remove `_lq`
for src in lq_files:
    if not src.is_file():
        continue
    stem, suf = src.stem, src.suffix
    new_stem = stem.replace("_lq", "")
    dest = next_free_path(lq_dir / f"{new_stem}{suf}")
    shutil.copy2(src, dest)

print(f"Searched: {pth.resolve()}")
print(f"GT copied: {len(gt_files)} -> {gt_dir}")
print(f"LQ copied: {len(lq_files)} -> {lq_dir}")
