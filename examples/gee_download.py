from itertools import product
from peony.db import download_gee_composite
import os
import sys

#CITIES = ['accra', 'barcelona', 'hongkong', 'lagos', 'london', 'luanda', 'munich', 'singapore']
CITIES = ['hongkong', 'hanoi']
METHODS = ['median', 'q-mosaic', 'mosaic']

if __name__ == '__main__':
    for city, method in product(CITIES, METHODS):
        try:
            download_gee_composite(os.path.join('cities', f"{city}.json"), os.path.join(sys.argv[1], f"{city}_{method}.tiff"), mosaic=method, cloudless_portion=0.0)
        except:
            pass
