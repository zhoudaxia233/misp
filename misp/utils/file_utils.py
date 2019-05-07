import os
import shutil
from pathlib import Path

__all__ = ['copy_dir_tree']

def copy_dir_tree(src: str, dst: str, ignore_files: bool=False, symlinks: bool=False):
    src = Path(src)
    dst = Path(dst)

    print('Start copying directory tree ...')

    if os.path.isdir(dst):
        shutil.rmtree(dst)

    if ignore_files:
        def ignore_func(folder, folder_contents):
            return [f for f in folder_contents if not os.path.isdir(os.path.join(folder, f))]
        shutil.copytree(src, dst, symlinks=symlinks, ignore=ignore_func)

    else:
        shutil.copytree(src, dst, symlinks=symlinks)

    print('Copying directory tree is done.')
