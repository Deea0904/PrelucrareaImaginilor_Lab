import cv2
import numpy as np
import os
import sys
from pathlib import Path


# Lab 8 - Operatii morfologice pe imagini binare

# Image path
image_path = r'./forma plina.jpg'


OBIECT, FUNDAL = 0, 255
B = np.ones((3, 3), dtype=bool)

# Binarizare
def binarizare(gray):
    _, binara = cv2.threshold(gray, 0, FUNDAL, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binara == OBIECT

# Eroziune
def eroziune(A, B):
    k = B.shape[0] // 2
    extinsa = np.pad(A, k, constant_values=False)
    M, N = A.shape
    rezultat = A.copy()
    for u, v in zip(*np.nonzero(B)):
        rezultat &= extinsa[u:u + M, v:v + N]
    return rezultat

# Extragerea conturului
def extragere_contur(A, B):
    return A & ~eroziune(A, B)


# masca booleana -> imagine (obiect negru pe fundal alb)
def la_imagine(masca):
    return np.where(masca, OBIECT, FUNDAL).astype(np.uint8)


os.chdir(Path(__file__).resolve().parent)

img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

A = binarizare(gray)
erodata = eroziune(A, B)
contur = extragere_contur(A, B)
print(f'Pixeli obiect: {A.sum()}, dupa eroziune: {erodata.sum()}, pe contur: {contur.sum()}')

cv2.imshow("Imagine originala", gray)
cv2.waitKey(0)
cv2.imshow("Imagine binara A", la_imagine(A))
cv2.waitKey(0)
cv2.imshow("Eroziune A (-) B", la_imagine(erodata))
cv2.waitKey(0)
cv2.imshow("Contur beta(A)", la_imagine(contur))
cv2.waitKey(0)
cv2.destroyAllWindows()