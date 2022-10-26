from itertools import product
from peony.db import download_gee_composite
import os
import sys

CITIES = ['accra.json', 'barcelona.json', 'hanoi.json', 'hongkong.json', 'lagos.json', 'london.json', 'luanda.json', 'munich.json', 'singapore.json']
METHODS = ['median', 'q-mosaic', 'mosaic']
CLOUDLESS = [0.25, 0.5, 0.75]

if __name__ == '__main__':
    for city, method, cloudless in product(CITIES, METHODS, CLOUDLESS):
        cloudint = int(cloudless * 100)
        download_gee_composite(os.path.join('cities', city), os.path.join(sys.argv[1], f"{city}_{method}_{cloudint}.tiff"), mosaic=method, cloudless_portion=cloudless)
