from itertools import product
from peony.db import download_gee_composite
import os
import sys

CITIES = ['accra', 'barcelona', 'hongkong', 'lagos', 'london', 'luanda', 'munich', 'singapore']
METHODS = ['medoid']
#CITIES = ['hongkong', 'hanoi']
#METHODS = ['median', 'q-mosaic', 'mosaic']

if __name__ == '__main__':
    for city, method in product(CITIES, METHODS):
        print(f"Trying the {method} method for the area of {city}")
        download_gee_composite(os.path.join('cities', f"{city}.json"), os.path.join(sys.argv[1], f"{city}_{method}.tiff"), mosaic=method, cloudless_portion=0.0, max_tile_size=8, start_date='2019-01-01', end_date='2020-01-01', project_name='ee-orbitfold')

