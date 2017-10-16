"""
Tobiasz Jarosiewicz

Code to learn basics of how to use multiprocessing.
The idea is to have some time-consuming function (like adding noise pixel by 
pixel with ar_rgb_sel() function) run in parallel. 
Currently the 'image' is split in 4 quarters but is should be divided in as 
many parts as needed. 
"""

import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import random
import time

def gen_rnd_sparse(w, h):
    """
    Create an empty array and fill it with random data but with many zero-values.
    """
    # Give the matrix/array dimmensions
    #w, h = 512, 512

    # Creathe the matrix
    data = np.zeros((w, h, 3), dtype=np.uint8)
    
    # Fill the matrix with random color values
    for i in range(0, w, int(w/50)):
        for j in range(0, h, int(h/50)):
            r = random.randint(0, 50)
            g = random.randint(0, 100)
            b = random.randint(0, 255)
            data[i][j] = [r, g, b]
    return data

def gen_rnd_ar(w, h):
    """
    Same as above but whole matrix is populated.
    """
    data = np.zeros((w, h, 3), dtype=np.uint8)
    for i in range(0, w):
        for j in range(0, h):
            r = random.randint(0, 50)
            g = random.randint(0, 100)
            b = random.randint(0, 255)
            data[i][j] = [r, g, b]
    return data


def fill_rgb(in_ar):
    """
    For debug mostly - fill each quarter of the matrix with different colour.
    """
    col_val = 142
    for x in range(0, 63):
        for y in range(0, 63):
            in_ar[x][y] = [col_val, 0, 0]
    for x in range(0, 63):
        for y in range(64, 127):
            in_ar[x][y] = [0, col_val, 0]
    for x in range(64, 127):
        for y in range(0, 63):
            in_ar[x][y] = [0, 0, col_val]
    for x in range(63, 127):
        for y in range(63, 127):
            in_ar[x][y] = [col_val, col_val, col_val]

def show_img(data):
    """
    Display the image using matplotlib.
    """
    plt.imshow(data)
    plt.title("random")
    plt.show(block=False)
    plt.pause(1e-3)

def ar_brigh(in_array):
    rgbi = 0
    
    #w = len(in_array)
    #h = len(in_array[0])
    #print(nxy, w, h)
    print("begin")
    """
    for x in range(0, w-1):
        for y in range(0, h-1):
            if np.array_equal(in_array[x][y], [0, 0, 0]) == True:
    """
    for it1 in in_array:
        for it2 in it1:
            for rgbv in it2:
                eps = random.randint(0, 2)
                it2[rgbi] += eps
                #print(rgbv, eps, asdx)
                rgbi += 1
            rgbi = 0

def ar_rgb_sel(in_array, col):
    """
    Select a channel from rgb as an argument.
    Iterate through each element and add a random
    value to each element increasing the chosen channel.
    """
    #w = len(in_array)
    #h = len(in_array[0])
    #print(nxy, w, h)
    
    for i in range(40):
        chval = 1
    
        if col == "r":
            for it1 in in_array:
                for it2 in it1:
                    for rgbv in it2:
                        eps = random.randint(0, chval)
                        it2[0] += eps
        elif col == "g":
            for it1 in in_array:
                for it2 in it1:
                    for rgbv in it2:
                        eps = random.randint(0, chval)
                        it2[1] += eps
        elif col == "b":
            for it1 in in_array:
                for it2 in it1:
                    for rgbv in it2:
                        eps = random.randint(0, chval)
                        it2[2] += eps
        elif col == "w":
            for it1 in in_array:
                for it2 in it1:
                    for rgbv in it2:
                        eps = random.randint(0, chval)
                        it2[0] += eps
                        it2[1] += eps
                        it2[2] += eps
        else:
            pass
    return in_array

def split_ar(in_array):
    """
    Split the input array in 4 equal parts. This is done in 2 steps - first 
    the 0 axis is split creating 2 rectangle matrices, then these 2 matrices 
    are divided into 2 leaving 4 equal parts. Probably will fail with odd sized 
    ndarrays or not square.
    """
    out_ar = []
    # Divide the array in 2 along the x axis:
    n1 = np.split(in_array, 2, axis = 0)
    for i in n1:
        # Divide along the y axis:
        n2 = np.split(i, 2, axis = 1)
        for j in n2:
            # The actual "small" arrays:
            #show_img(j)
            out_ar.append(j)
    return out_ar

def join_ar(array_list):
    """
    Join 4 parts of the matrix split by split_ar() function. The function 
    will work on any 4 ndarrays but is written to restore the original image 
    of matrix before splitting.
    """
    ay_join1 = np.concatenate((array_list[0], array_list[1]), axis = 0)
    ay_join2 = np.concatenate((array_list[2], array_list[3]), axis = 0)
    ay_fin = np.concatenate((ay_join1, ay_join2), axis = 1)
    #np.transpose(ay_fin, (0, 2, 1))
    #a = np.transpose(ay_fin)
    a = np.transpose(ay_fin, (1, 0, 2))
    #print(a.shape)
    return a

t_s = time.clock()

x = 128
y = 128
#img1 = gen_rnd_ar(x, y)
img1 = np.zeros((128, 128, 3), dtype=np.uint8)
#show_img(img1)

fill_rgb(img1)
#print("original")
show_img(img1)
#print(img1.shape)

"""
for i in range(2):
    #img1 = gen_rnd_ar(x, y)
    show_img(img1)
"""

#ar_rgb_sel(img1, 'r')
"""
for i in range(10):
    #ar_brigh(img1)
    ar_rgb_sel(img1, 'b')
    show_img(img1)
"""


ar_s = split_ar(img1)
a0 = ar_s[0]
a1 = ar_s[1]
a2 = ar_s[2]
a3 = ar_s[3]


# Multiprocessing part:

number_processes = multiprocessing.cpu_count()
pool = multiprocessing.Pool(number_processes)
total_tasks = 4
tasks = range(total_tasks)

args = [(a0, "r"), (a1, "g"), (a2, 'b'), (a3, 'w')]
    
results = pool.starmap(ar_rgb_sel, args)
pool.close()
pool.join()

out_ar = []
for elem in results:
    out_ar.append(elem)


#out_ar.append(a0)
#out_ar.append(a1)
#out_ar.append(a2)
#out_ar.append(a3)



final_array = join_ar(out_ar)
show_img(final_array)
#show_img(img1)

t_f = time.clock()
print(t_f-t_s)

