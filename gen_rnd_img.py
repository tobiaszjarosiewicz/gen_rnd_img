"""
Tobiasz Jarosiewicz
"""

import numpy as np
import random
from PIL import Image

# Give the matrix/array dimmensions
w, h = 512, 512
# Fill the matrix with zeros
data = np.zeros((h, w, 3), dtype=np.uint8)

# Check dimmensions:
print(data.ndim)
# check data size (w*h*3) because every element has RGB val
print(data.size)

# Fill the matrix with random color values
for i in range(0, w):
    for j in range(0, h):
        r = random.randint(0, 50)
        g = random.randint(0, 100)
        b = random.randint(0, 255)
        data[i][j] = [r, g, b]

img = Image.fromarray(data, 'RGB')
# Uncomment to save the image to a file.
#img.save('asd.png')
img.show()
