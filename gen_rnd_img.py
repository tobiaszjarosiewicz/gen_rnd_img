"""
Tobiasz Jarosiewicz
"""

import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt

def gen_rnd_ar(w, h):
    # Give the matrix/array dimmensions
    #w, h = 512, 512

    # Fill the matrix with zeros
    data = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Check dimmensions:
    print(data.ndim)
    # check data size (w*h*3) because every element has RGB val
    print(data.size)
    
    # Fill the matrix with random color values
    for i in range(0, w, int(w/50)):
        for j in range(0, h, int(h/50)):
            r = random.randint(0, 50)
            g = random.randint(0, 100)
            b = random.randint(0, 255)
            data[i][j] = [r, g, b]
    return data

def show_img(data):
    plt.imshow(data)
    plt.title("random")
    plt.show(block=False)
    plt.pause(1e-3)


img1 = gen_rnd_ar(128, 128)
show_img(img1)

for i in range(10):
    img1 = gen_rnd_ar(128, 128)
    show_img(img1)
