import os, re
from PIL import Image

FOLDER = r""
ROWS = 5
COLS = 5
CELL_W = 500
CELL_H = 500
PADDING = 8
BG_HEX = "#1e1e1e"
OUTPUT_PATH = "mosaic.jpg"
USE_ALPHA = False

def natural_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def list_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in exts], key=natural_key)

def resize_to_fit(img, cell_w, cell_h):
    w, h = img.size
    scale = min(cell_w / w, cell_h / h)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.LANCZOS)

def hex_to_tuple(h):
    h = h.lstrip("#")
    if len(h) == 6:
        return tuple(int(h[i:i+2], 16) for i in (0,2,4))
    elif len(h) == 8:
        return tuple(int(h[i:i+2], 16) for i in (0,2,4,6))
    raise ValueError("BG_HEX must be 6 or 8 hex digits")

def make_mosaic(folder, rows, cols, cell_w, cell_h, padding, bg_color, output_path, use_alpha=False):
    images = list_images(folder)
    if not images:
        raise ValueError(f"No images found in: {folder}")
    max_cells = rows * cols
    paths = images[:max_cells]
    mode = "RGBA" if use_alpha else "RGB"
    mosaic_w = cols * cell_w + (cols - 1) * padding
    mosaic_h = rows * cell_h + (rows - 1) * padding
    canvas = Image.new(mode, (mosaic_w, mosaic_h), bg_color)
    for idx, p in enumerate(paths):
        try:
            with Image.open(p) as im:
                im = im.convert("RGBA") if use_alpha else im.convert("RGB")
                fitted = resize_to_fit(im, cell_w, cell_h)
                x_cell = idx % cols
                y_cell = idx // cols
                x0 = x_cell * (cell_w + padding) + (cell_w - fitted.width) // 2
                y0 = y_cell * (cell_h + padding) + (cell_h - fitted.height) // 2
                if use_alpha:
                    canvas.alpha_composite(fitted, dest=(x0, y0))
                else:
                    canvas.paste(fitted, (x0, y0))
        except Exception as e:
            print(f"Skipping '{p}': {e}")
    ext = os.path.splitext(output_path)[1].lower()
    if use_alpha and ext not in {".png", ".webp"}:
        output_path = os.path.splitext(output_path)[0] + ".png"
    canvas.save(output_path)
    return output_path

bg = hex_to_tuple(BG_HEX)
make_mosaic(FOLDER, ROWS, COLS, CELL_W, CELL_H, PADDING, bg, OUTPUT_PATH, USE_ALPHA)