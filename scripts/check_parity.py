import os

# === CONFIGURATION ===
SOURCE_DIR = r""
TARGET_DIR = r""

def get_all_relative_file_paths(base_path):
    relative_paths = set()
    for root, _, files in os.walk(base_path):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, base_path)
            relative_paths.add(rel_path)
    return relative_paths

def compare_folders(src, tgt):
    src_files = get_all_relative_file_paths(src)
    tgt_files = get_all_relative_file_paths(tgt)

    only_in_src = sorted(src_files - tgt_files)
    only_in_tgt = sorted(tgt_files - src_files)

    print("\nFiles only in SOURCE folder:")
    for f in only_in_src:
        print(f"  {f}")

    print("\nFiles only in TARGET folder:")
    for f in only_in_tgt:
        print(f"  {f}")

compare_folders(SOURCE_DIR, TARGET_DIR)
