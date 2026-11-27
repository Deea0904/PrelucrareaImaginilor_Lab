import cv2
import numpy as np
import os
from collections import deque
from pathlib import Path

# Lab 9 - Etichetarea componentelor conexe
# Algoritm 1 - Traversare in latime (BFS)

image_path = r'./forma_binara.jpg'

OBIECT, FUNDAL = 0, 255

# Vecinatatea de 8 (slide 3): toate pozitiile cu diferenta <= 1 pe fiecare coordonata
N8 = [(-1, -1), (-1, 0), (-1, 1),
      (0, -1),           (0, 1),
      (1, -1),  (1, 0),  (1, 1)]


# Binarizare: obiect = 0 (negru), fundal = 255 (alb)
def binarizare(gray):
    _, binara = cv2.threshold(gray, 0, FUNDAL, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binara


# Algoritm 1 - Traversare in latime (BFS)
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
                    qi, qj = Q.popleft()                  # FIFO -> latime
                    for di, dj in N8:
                        ni, nj = qi + di, qj + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            if img[ni, nj] == OBIECT and labels[ni, nj] == 0:
                                labels[ni, nj] = label
                                Q.append((ni, nj))
    return labels, label


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

labels, n = etichetare_bfs(binara)
print(f'Algoritm 1 (BFS): {n} obiecte')

cv2.imshow("Imagine binara", binara)
cv2.waitKey(0)
cv2.imshow("Etichetare BFS (Algoritm 1)", colorare(labels, n))
cv2.waitKey(0)
cv2.destroyAllWindows()