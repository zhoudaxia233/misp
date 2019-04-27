import os
import shutil
from pathlib import Path


def copy_dir_tree(src: str, dst: str, ignore_files: bool=False, symlinks: bool=False):
    src = Path(src)
    dst = Path(dst)

    if os.path.isdir(dst):
        os.rmdir(dst)

    if ignore_files:
        def ignore_func(folder, folder_contents):
            return [f for f in folder_contents if not os.path.isdir(os.path.join(folder, f))]
        shutil.copytree(src, dst, symlinks=symlinks, ignore=ignore_func)

    else:
        shutil.copytree(src, dst, symlinks=symlinks)

    print('Done.')
