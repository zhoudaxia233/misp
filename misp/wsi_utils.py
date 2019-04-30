import os
import openslide
from openslide.deepzoom import DeepZoomGenerator
from tqdm import tqdm

__all__ = ['WSI', 'validate_mpp', 'stitch_tiles']

class WSI():
    def __init__(self, path: str, tile_size: int=224):
        self.path = path
        self.tile_size = tile_size
        self.slide = openslide.OpenSlide(self.path)
        self.deepzoom_gen = DeepZoomGenerator(self.slide, tile_size=self.tile_size, overlap=0, limit_bounds=False)
        self.mpp = (float(self.slide.properties[openslide.PROPERTY_NAME_MPP_X]), float(self.slide.properties[openslide.PROPERTY_NAME_MPP_Y]))
        self.level_count = self.deepzoom_gen.level_count
        self.level_dimensions = self.deepzoom_gen.level_dimensions
        self.level_tiles = self.deepzoom_gen.level_tiles

    def get_info(self):
        '''Get basic information of the wsi.
        '''
        print(f'Num of levels: {self.level_count}')
        print('Dimensions of levels:')
        for level_nr in range(self.level_count):
            print(f'- level {level_nr}: {self.level_dimensions[level_nr]}')
        print(f'MPP: {self.mpp}')
        print()
    
    def get_tile(self, row: int, col: int):
        return self.deepzoom_gen.get_tile(level=self.level_count-1, address=(col, row))

    def make_tiles(self, tiles_folder_path: str):
        if not os.path.isdir(tiles_folder_path):
            os.mkdir(tiles_folder_path)
        level = self.level_count - 1
        cols, rows = self.level_tiles[level]
        for row in tqdm(range(rows)):
            for col in range(cols):
                self.get_tile(row, col).save(tiles_folder_path + f'{row}_{col}.tif')
        return

def validate_mpp(path: str, mpp: float=0.5, thresh: float=0.015) -> bool:
    '''Validate if the input WSI has the same MPP as our training data.
    '''
    wsi = openslide.OpenSlide(path)
    x_diff = float(wsi.properties[openslide.PROPERTY_NAME_MPP_X]) - mpp
    y_diff = float(wsi.properties[openslide.PROPERTY_NAME_MPP_Y]) - mpp
    return (abs(x_diff) < thresh) and (abs(y_diff) < thresh)

def stitch_tiles(patches_folder_path: str, target_folder_path: str):
    pass
