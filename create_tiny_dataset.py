import os
import shutil
import random
from pathlib import Path
import argparse
from tqdm import tqdm

def create_tiny_dataset(src: str, dst: str, num: float, file_format: str):
    if num <= 0:
        print('"num" must be greater than zero!')
        return
    
    src = Path(src)
    dst = Path(dst)

    if not os.path.isdir(dst):
        os.mkdir(dst)

    candidates = os.listdir(src)
    if num < 1:
        # sample by percentage
        filenames = random.sample(candidates, int(len(candidates) * num))
    else:
        # sample by number
        filenames = random.sample(candidates, int(num))
    
    for filename in tqdm(filenames):
        if filename.endswith(file_format):
            filepath = src / filename
            shutil.copy(filepath, dst)
    print('Done.')

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', help="path of the source folder", type=str)
    parser.add_argument('-d', '--dst', help="path of the destination folder", type=str)
    parser.add_argument('-n', '--num', help="the percentage or number of needed files in the source folder",
                        type=float, default=0.2)
    parser.add_argument('-ff', '--file_format', help='the file format we need to use in the source folder',
                        type=str, default='.tif')
    args = parser.parse_args()
    return args.src, args.dst, args.num, args.file_format

def main():
    src, dst, num, file_format = init()
    # -- Section starts --
    # customize here for your own folder structures
    create_tiny_dataset(src, dst, num, file_format)
    # -- Section ends --


if __name__ == '__main__':
    main()
