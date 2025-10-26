import os
from abc import ABC, abstractmethod
import cv2
import geopandas as gpd
from shapely.geometry import box, Polygon
import rasterio
from rasterio.windows import Window
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image


class ImageCropperBase(ABC):
    def __init__(self, source_path, target_path, max_workers=8):
        self.source_path = source_path
        self.target_path = target_path
        self.max_workers = max_workers
        os.makedirs(self.target_path, exist_ok=True)

    @abstractmethod
    def is_supported(self, filename):
        pass

    @abstractmethod
    def process_image(self, filepath, filename, size):
        pass


class ConventionalImageCropper(ImageCropperBase):
    def is_supported(self, filename):
        return os.path.splitext(filename)[1].lower() in {'.png', '.jpg', '.jpeg'}

    def _crop_and_save(self, image, base_name, ext, x, y, row, col, size):
        crop = image[y:y + size, x:x + size]
        crop_name = f"{base_name}__R{row}__C{col}{ext}"
        out_path = os.path.join(self.target_path, crop_name)
        cv2.imwrite(out_path, crop)

    def process_image(self, filepath, filename, size):
        image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"[Invalid Image] Skipping: {filename}")
            return

        height, width = image.shape[:2]
        base_name, ext = os.path.splitext(filename)

        tasks = []
        for row, y in enumerate(range(0, height - size + 1, size)):
            for col, x in enumerate(range(0, width - size + 1, size)):
                tasks.append((image, base_name, ext, x, y, row, col, size))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            executor.map(lambda args: self._crop_and_save(*args), tasks)


class GeoTIFFImageCropper(ImageCropperBase):
    def __init__(self, source_path, target_path, shapefile_path=None, max_workers=8):
        super().__init__(source_path, target_path, max_workers)
        self.shapefile_path = shapefile_path

    def is_supported(self, filename):
        return os.path.splitext(filename)[1].lower() in {'.tif', '.tiff'}

    def _crop_and_save(self, data, meta, transform, base_name, row, col, size):
        out_name = f"{base_name}__R{row // size}__C{col // size}.tif"
        out_path = os.path.join(self.target_path, out_name)

        meta = meta.copy()
        meta.update({
            "height": size,
            "width": size,
            "transform": transform
        })
        with rasterio.Env(GDAL_PAM_ENABLED=False):
            with rasterio.open(out_path, "w", **meta) as dst:
                dst.write(data)

    def process_image(self, filepath, filename, size):
        if self.shapefile_path is None:
            raise ValueError("shapefile_path must be provided for polygon-based filtering.")

        gdf = gpd.read_file(self.shapefile_path)
        with rasterio.open(filepath) as src:
            if src.crs != gdf.crs:
                gdf = gdf.to_crs(src.crs)
            polygon = gdf.unary_union  # Merge all geometries

            width, height = src.width, src.height
            base_name, _ = os.path.splitext(filename)
            meta = src.meta.copy()
            tasks = []
            included = 0
            for row in range(0, height - size + 1, size):
                for col in range(0, width - size + 1, size):
                    window = Window(col, row, size, size)
                    transform = src.window_transform(window)
                    bounds = rasterio.transform.array_bounds(size, size, transform)
                    crop_poly = box(*bounds)

                    if polygon.contains(crop_poly):
                        data = src.read(window=window)
                        tasks.append((data, meta, transform, base_name, row, col, size))
                        included += 1

            if not tasks:
                print(f"[No Tiles] Skipped: {filename} — No tiles matched the polygon")
                return

            print(f"[{filename}] Accepted {included} tile(s) inside polygon.")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                executor.map(lambda args: self._crop_and_save(*args), tasks)
