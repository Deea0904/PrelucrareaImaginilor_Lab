import cv2
import numpy as np
import os
from collections import deque
from pathlib import Path

# Tema 9 - Etichetarea componentelor conexe
# Algoritm 2 - Doua treceri cu clase de echivalenta
# TREBUIA AFISATA SI IMAGINEA INTERMEDIARA

image_path = r'./forma_binara.jpg'

OBIECT, FUNDAL = 0, 255

# Vecinii anteriori pentru parcurgere sus-jos, stanga-dreapta
NP = [(0, -1), (-1, -1), (-1, 0), (-1, 1)]


# Binarizare: obiect = 0 (negru), fundal = 255 (alb)
def binarizare(gray):
    _, binara = cv2.threshold(gray, 0, FUNDAL, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binara


# Algoritm 2 - Doua treceri cu clase de echivalenta
def etichetare_doua_treceri(img):
    h, w = img.shape
    label = 0
    labels = np.zeros((h, w), dtype=np.int32)
    edges = {}                                            # graful echivalentelor

    # Prima trecere
    for i in range(h):
        for j in range(w):
            if img[i, j] == OBIECT and labels[i, j] == 0:
                L = []
                for di, dj in NP:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w and labels[ni, nj] > 0:
                        L.append(labels[ni, nj])
                if len(L) == 0:                           # eticheta noua
                    label += 1
                    labels[i, j] = label
                    edges[label] = []
                else:                                     # cel mai mic vecin
                    x = min(L)
                    labels[i, j] = x
                    for y in L:
                        if y != x:
                            edges[x].append(y)
                            edges[y].append(x)

    # BFS pe graful etichetelor (clase de echivalenta)
    newlabel = 0
    newlabels = np.zeros(label + 1, dtype=np.int32)
    for i in range(1, label + 1):
        if newlabels[i] == 0:
            newlabel += 1
            Q = deque()
            newlabels[i] = newlabel
            Q.append(i)
            while Q:
                x = Q.popleft()
                for y in edges[x]:
                    if newlabels[y] == 0:
                        newlabels[y] = newlabel
                        Q.append(y)

    # A doua trecere: reetichetare
    for i in range(h):
        for j in range(w):
            labels[i, j] = newlabels[labels[i, j]]
    return labels, newlabel, label


# matrice de etichete -> imagine color (fundal alb, cate o culoare pe obiect)
def colorare(labels, n):
    rng = np.random.default_rng(1)
    culori = rng.integers(0, 200, size=(n + 1, 3), dtype=np.uint8)
    culori[0] = (255, 255, 255)
    return culori[labels]


os.chdir(Path(__file__).resolve().parent)

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Nu pot citi imaginea: {image_path}")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binara = binarizare(gray)

labels, n, etichete_initiale = etichetare_doua_treceri(binara)
print(f'Algoritm 2: {etichete_initiale} etichete initiale -> {n} obiecte')

cv2.imshow("Imagine binara", binara)
cv2.waitKey(0)
cv2.imshow("Etichetare doua treceri (Algoritm 2)", colorare(labels, n))
cv2.waitKey(0)
cv2.destroyAllWindows()