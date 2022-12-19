import os
import gdown
from typing import Optional
from matplotlib import pyplot as plt


def rename_and_filter_images(dirname: str, min_height: int = 256,
        max_height: Optional[int] = None, min_width: int = 256, 
        max_width: Optional[int] = None) -> None:
    dirname = './landscapes/'

    if max_height is None:
        max_height = float('inf')
    if max_width is None:
        max_width = float('inf')

    for i, filename in enumerate(os.listdir(dirname)):
        full_filename = dirname + filename
        new_filename = dirname + f'{i+1}.jpg'
        
        os.rename(full_filename, new_filename)

        img = plt.imread(new_filename)
        height, width = img.shape[:2]
        
        if min_height <= height <= max_height and \
                min_width <= width <= max_width:
            continue
        else:
            os.remove(new_filename)
