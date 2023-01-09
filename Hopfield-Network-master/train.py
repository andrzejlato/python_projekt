# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 08:40:49 2018

@author: user
"""

import numpy as np
np.random.seed(1)
from matplotlib import pyplot as plt
import skimage.data
from skimage.color import rgb2gray
from skimage.filters import threshold_mean
from skimage.transform import resize
import network
from skimage import io as io
import random


# Utils
def get_corrupted_input(input, corruption_level):
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=len(input))
    for i, v in enumerate(input):
        if inv[i]:
            corrupted[i] = -1 * v
    return corrupted


def get_corrupted_input_line(input, corruption_level):
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=128)
    x = 127
    for i in range(0, x):
        if inv[i]:
            for n in range(0, x):
                corrupted[i*(x+1) + n] = -1
    return corrupted


def get_corrupted_input_cover(input, corruption_level, noise_size):
    corrupted = np.copy(input)
    inv = np.random.binomial(n=1, p=corruption_level, size=128)
    x = 128
    for i in range(0, x - 1):
        if inv[i]:
            k = noise_size
            l = random.randint(0, x - k - 1)
            for n in range(i, i + k - 1):
                for m in range(l, l + k - 1):
                        corrupted[(x - 1)*m + n + i] = -1
    return corrupted

def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data

def plot(data, test, predicted, figsize=(5, 6)):
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]

    fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
    for i in range(len(data)):
        if i==0:
            axarr[i, 0].set_title('Train data')
            axarr[i, 1].set_title("Input data")
            axarr[i, 2].set_title('Output data')

        axarr[i, 0].imshow(data[i])
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(test[i])
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted[i])
        axarr[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig("result.png")
    plt.show()

def preprocessing(img, w=128, h=128):
    # Resize image
    img = resize(img, (w,h), mode='reflect')

    # Thresholding
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2*(binary*1)-1 # Boolian to int

    # Reshape
    flatten = np.reshape(shift, (w*h))
    return flatten

def main():
    # Load data
    palac = rgb2gray(io.imread("C:/Users/Grzegorz/Desktop/projekt/Hopfield-Network-master/sources/palac.png"))
    kotek = rgb2gray(skimage.io.imread("C:/Users/Grzegorz/Desktop/projekt/Hopfield-Network-master/sources/kotek.png"))
    student = rgb2gray(skimage.io.imread("C:/Users/Grzegorz/Desktop/projekt/Hopfield-Network-master/sources/student.png"))
    kebab = rgb2gray(skimage.io.imread("C:/Users/Grzegorz/Desktop/projekt/Hopfield-Network-master/sources/kebab.png"))

    # Marge data
    data = [palac, kotek, student, kebab]

    # Preprocessing
    print("Start to data preprocessing...")
    data = [preprocessing(d) for d in data]

    # Create Hopfield Network Model
    model = network.HopfieldNetwork()
    model.train_weights(data)

    # Generate testset
    #test = [get_corrupted_input(d, 0.48) for d in data] #0.3, 0.5, 0.4, 0.45, 0.48
    #test = [get_corrupted_input_cover(d, 0.02, 68) for d in data] #60, 65 ,68, 70
    test = [get_corrupted_input_line(d, 0.3) for d in data]
    predicted = model.predict(test, threshold=0, asyn=False)
    print("Show prediction results...")
    plot(data, test, predicted)
    print("Show network weights matrix...")
    #model.plot_weights()

if __name__ == '__main__':
    main()
