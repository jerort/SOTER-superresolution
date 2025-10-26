import os
from src.preprocessing.cropping import ConventionalImageCropper, GeoTIFFImageCropper
from concurrent.futures import ThreadPoolExecutor


def process_dataset(croppers, source_path, _size, max_workers=8):
    files = [f for f in os.listdir(source_path) if os.path.isfile(os.path.join(source_path, f))]
    tasks = []

    for filename in files:
        filepath = os.path.join(source_path, filename)
        for cropper in croppers:
            if cropper.is_supported(filename):
                tasks.append((cropper, filepath, filename))
                break
        else:
            print(f"[Unsupported File] Skipping: {filename}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(lambda args: args[0].process_image(args[1], args[2], _size), tasks)

# Instantiate and run
input_path = r""
output_path= r""
size = 250

cv_cropper = ConventionalImageCropper(input_path, output_path)
shapefile = r""
geo_cropper = GeoTIFFImageCropper(input_path, output_path, shapefile_path=shapefile)

process_dataset([geo_cropper, cv_cropper], input_path, size)
