import os

def rename_all_within(root_path):
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        # 重命名文件
        for filename in filenames:
            if ' ' in filename:
                old_file = os.path.join(dirpath, filename)
                new_filename = filename.replace(' ', '_')
                new_file = os.path.join(dirpath, new_filename)
                os.rename(old_file, new_file)
                print(f"[文件] {old_file} -> {new_file}")
        # 重命名目录
        for dirname in dirnames:
            if ' ' in dirname:
                old_dir = os.path.join(dirpath, dirname)
                new_dirname = dirname.replace(' ', '_')
                new_dir = os.path.join(dirpath, new_dirname)
                os.rename(old_dir, new_dir)
                print(f"[目录] {old_dir} -> {new_dir}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help="根目录路径")
    args = parser.parse_args()

    rename_all_within(args.path)

