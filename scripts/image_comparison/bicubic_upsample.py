from pathlib import Path
from PIL import Image

input_path = Path(r"")
output_path = Path(r"")

for p in input_path.rglob("*.png"):
    img = Image.open(p)
    out = img.resize((img.width * 2, img.height * 2), resample=Image.BICUBIC)
    out.save(output_path / p.name)
    print(f"Saved {p.name} to {output_path / p.name} with size {out.width}x{out.height}")
