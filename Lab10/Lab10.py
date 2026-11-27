import cv2
import numpy as np
import os
from collections import deque
from pathlib import Path

# Lab 10 - Algoritmul de urmarire a conturului (vecinatate de 8)

image_path = r'./forme_simple.jpg'

OBIECT, FUNDAL = 0, 255

# Directiile 0..7 pentru vecinatatea de 8 (slide 9), ca deplasari (di, dj) = (linie, coloana)
#   3 2 1
#   4 x 0
#   5 6 7
DI = [0, -1, -1, -1, 0, 1, 1, 1]
DJ = [1, 1, 0, -1, -1, -1, 0, 1]

# Vecinatatea de 8 pentru etichetare
N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# Binarizare: obiect = 0 (negru), fundal = 255 (alb)
def binarizare(gray):
    _, binara = cv2.threshold(gray, 0, FUNDAL, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binara


# Etichetarea prealabila (Laborator 9, BFS) - algoritmul de contur presupune imagine etichetata
def etichetare_bfs(img):
    h, w = img.shape
    label = 0
    labels = np.zeros((h, w), dtype=np.int32)
    for i in range(h):
        for j in range(w):
            if img[i, j] == OBIECT and labels[i, j] == 0:
                label += 1
                Q = deque()
                labels[i, j] = label
                Q.append((i, j))
                while Q:
                    qi, qj = Q.popleft()
                    for di, dj in N8:
                        ni, nj = qi + di, qj + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            if img[ni, nj] == OBIECT and labels[ni, nj] == 0:
                                labels[ni, nj] = label
                                Q.append((ni, nj))
    return labels, label


# Algoritmul de urmarire a conturului (slide 3-5), vecinatate de 8
def urmarire_contur(labels, start):
    h, w = labels.shape
    val = labels[start]                      # "aceeasi valoare ca pixelul curent" = aceeasi eticheta
    P = [start]                              # P0 = pixelul de start
    dir_ = 7                                 # pasul 1: dir = 7 pentru vecinatate de 8
    cur = start
    while True:
        # Pasul 2: pozitia de la care incepe cautarea (sens invers acelor de ceasornic)
        if dir_ % 2 == 0:
            s = (dir_ + 7) % 8
        else:
            s = (dir_ + 6) % 8
        gasit = False
        for k in range(8):
            d = (s + k) % 8
            ni, nj = cur[0] + DI[d], cur[1] + DJ[d]
            if 0 <= ni < h and 0 <= nj < w and labels[ni, nj] == val:
                gasit = True
                break
        if not gasit:                        # regiune formata dintr-un singur pixel (caz trivial)
            return P
        cur = (ni, nj)                       # noul element P_n
        dir_ = d                             # se actualizeaza dir
        P.append(cur)
        # Pasul 3: P_n = P_1 si P_(n-1) = P_0 -> stop
        if len(P) >= 3 and P[-1] == P[1] and P[-2] == P[0]:
            break
    return P[:-2]                            # Pasul 4: conturul este P_0 ... P_(n-2)


os.chdir(Path(__file__).resolve().parent)

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Nu pot citi imaginea: {image_path}")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binara = binarizare(gray)

labels, n = etichetare_bfs(binara)
print(f'Obiecte etichetate: {n}')

rng = np.random.default_rng(1)
culori = rng.integers(80, 256, size=(n + 1, 3), dtype=np.uint8)
contururi = np.zeros((*binara.shape, 3), dtype=np.uint8)         # fundal negru, ca in slide
suprapus = cv2.cvtColor(binara, cv2.COLOR_GRAY2BGR)

for eticheta in range(1, n + 1):
    # Pasul 1: primul pixel al regiunii la scanarea din stanga sus
    start = tuple(int(x) for x in np.argwhere(labels == eticheta)[0])
    contur = urmarire_contur(labels, start)
    print(f'Obiect {eticheta}: start = {start}, pixeli pe contur = {len(contur)}')
    for (i, j) in contur:
        contururi[i, j] = culori[eticheta]
        suprapus[i, j] = (0, 0, 255)

cv2.imshow("Imagine binara", binara)
cv2.waitKey(0)
cv2.imshow("Contururi obiecte (urmarire contur)", contururi)
cv2.waitKey(0)
cv2.imshow("Contur (rosu) peste obiecte", suprapus)
cv2.waitKey(0)
cv2.destroyAllWindows()