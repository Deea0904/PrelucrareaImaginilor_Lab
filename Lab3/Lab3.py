import cv2
import numpy as np

image_path = r'food.jpg'
# Image directory
directory = r'./Imagini'

img = cv2.imread(image_path)
negative_img = 255 - img
contrast_img = cv2.convertScaleAbs(img, alpha=2.0, beta=0)
gamma=3
invGamma = 1.0 / gamma
table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
gamma_img = cv2.LUT(img, table)
brightness_img = cv2.convertScaleAbs(img, alpha=1, beta=60)
cv2.imshow("Imaginea originala", img)
key = cv2.waitKey()
cv2.imshow('Negativarea imaginii', negative_img)
key = cv2.waitKey()
cv2.imshow('Modificarea contrastului', contrast_img)
key = cv2.waitKey()
cv2.imshow('Corectia gamma', gamma_img)
key = cv2.waitKey()
cv2.imshow('Modificarea luminozitatii', brightness_img)

cv2.waitKey(0)
cv2.destroyAllWindows()