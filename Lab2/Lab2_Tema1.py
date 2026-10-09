import numpy as np
import cv2

img = cv2.imread('Image.jpg', 0)
hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
fdp = hist / (img.shape[0] * img.shape[1])
