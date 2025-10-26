import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# ==== INPUT & OUTPUT PATHS ====
input_path = r""
output_path = r""

# ==== TASK FUNCTION ====
def tiff_to_png(tiff_path, src_root, dst_root):
    try:
        relative_path = os.path.relpath(os.path.dirname(tiff_path), src_root)
        target_folder = os.path.join(dst_root, relative_path)
        os.makedirs(target_folder, exist_ok=True)

        png_filename = os.path.splitext(os.path.basename(tiff_path))[0] + '.png'
        png_path = os.path.join(target_folder, png_filename)

        with Image.open(tiff_path) as img:
            img.save(png_path, format='PNG')

        return f"Converted: {tiff_path} -> {png_path}"
    except Exception as e:
        return f"Failed: {tiff_path} - Error: {e}"

# ==== MAIN EXECUTION ====
def get_all_tiff_paths(src_root):
    tiff_paths = []
    for foldername, _, filenames in os.walk(src_root):
        for filename in filenames:
            if filename.lower().endswith(('.tiff', '.tif')):
                tiff_paths.append(os.path.join(foldername, filename))
    return tiff_paths

tiff_files = get_all_tiff_paths(input_path)

with ThreadPoolExecutor() as executor:
    futures = [executor.submit(tiff_to_png, path, input_path, output_path) for path in tiff_files]

    for future in as_completed(futures):
        print(future.result())

