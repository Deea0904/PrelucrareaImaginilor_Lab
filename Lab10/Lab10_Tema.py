import cv2
import numpy as np
import os
from collections import deque
from pathlib import Path

# Tema 10 - Extragerea codului inlantuit pentru un obiect dintr-o imagine binara (vecinatate de 8)

image_path = r'./forme_simple.jpg'
ETICHETA_OBIECT = 1          # obiectul (in ordinea scanarii stanga-sus) pentru care se extrage codul

OBIECT, FUNDAL = 0, 255

# Directiile 0..7 (slide 9), ca deplasari (di, dj) = (linie, coloana)
#   3 2 1
#   4 x 0
#   5 6 7
DI = [0, -1, -1, -1, 0, 1, 1, 1]
DJ = [1, 1, 0, -1, -1, -1, 0, 1]
DIRECTIE = {(DI[d], DJ[d]): d for d in range(8)}      # (di, dj) -> cod 0..7

N8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# Binarizare: obiect = 0 (negru), fundal = 255 (alb)
def binarizare(gray):
    _, binara = cv2.threshold(gray, 0, FUNDAL, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binara


# Etichetare BFS (Laborator 9)
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


# Urmarirea conturului (Laborator 10)
def urmarire_contur(labels, start):
    h, w = labels.shape
    val = labels[start]
    P = [start]
    dir_ = 7
    cur = start
    while True:
        s = (dir_ + 7) % 8 if dir_ % 2 == 0 else (dir_ + 6) % 8
        gasit = False
        for k in range(8):
            d = (s + k) % 8
            ni, nj = cur[0] + DI[d], cur[1] + DJ[d]
            if 0 <= ni < h and 0 <= nj < w and labels[ni, nj] == val:
                gasit = True
                break
        if not gasit:
            return P
        cur = (ni, nj)
        dir_ = d
        P.append(cur)
        if len(P) >= 3 and P[-1] == P[1] and P[-2] == P[0]:
            break
    return P[:-2]


# Codul inlantuit: directia de trecere de la fiecare pixel al conturului la urmatorul (contur inchis)
def cod_inlantuit(contur):
    m = len(contur)
    cod = []
    for k in range(m):
        a = contur[k]
        b = contur[(k + 1) % m]
        cod.append(DIRECTIE[(b[0] - a[0], b[1] - a[1])])
    return cod


# Derivata codului inlantuit: diferenta modulo 8 intre directii consecutive (slide 13)
def derivata(cod):
    n = len(cod)
    return [(cod[(k + 1) % n] - cod[k]) % 8 for k in range(n)]


# Normalizare: alegerea punctului de start astfel incat secventa sa formeze numarul minim (slide 12)
def normalizare(cod):
    n = len(cod)
    return min((cod[k:] + cod[:k] for k in range(n)))


# Reconstruirea conturului din punctul de referinta si codul inlantuit (slide 12)
def reconstruire(start, cod):
    puncte = [start]
    i, j = start
    for c in cod[:-1]:
        i, j = i + DI[c], j + DJ[c]
        puncte.append((i, j))
    return puncte


os.chdir(Path(__file__).resolve().parent)

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Nu pot citi imaginea: {image_path}")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binara = binarizare(gray)
labels, n = etichetare_bfs(binara)
if not 1 <= ETICHETA_OBIECT <= n:
    raise ValueError(f"ETICHETA_OBIECT trebuie sa fie intre 1 si {n}")

# Punctul de referinta: cel mai de sus, din stanga pixel al obiectului (slide 14)
start = tuple(int(x) for x in np.argwhere(labels == ETICHETA_OBIECT)[0])
contur = urmarire_contur(labels, start)
cod = cod_inlantuit(contur)
der = derivata(cod)
norm = normalizare(cod)

print(f'Obiect {ETICHETA_OBIECT}: punct de referinta (linie, coloana) = {start}')
print(f'Lungimea codului inlantuit: {len(cod)}')
print('Cod inlantuit:     ', ''.join(map(str, cod)))
print('Derivata codului:  ', ''.join(map(str, der)))
print('Cod normalizat:    ', ''.join(map(str, norm)))
print('Colturi (schimbari de directie):', sum(1 for k in range(len(cod)) if cod[k] != cod[(k + 1) % len(cod)]))

# Verificare: conturul reconstruit din (start + cod) trebuie sa fie identic cu cel urmarit
rec = reconstruire(start, cod)
print('Reconstruire identica cu conturul original:', rec == contur)

contur_img = np.zeros((*binara.shape, 3), dtype=np.uint8)
for (i, j) in contur:
    contur_img[i, j] = (255, 255, 255)
contur_img[start] = (0, 255, 0)                                   # punctul de start, verde
rec_img = np.zeros((*binara.shape, 3), dtype=np.uint8)
for (i, j) in rec:
    rec_img[i, j] = (255, 255, 255)

cv2.imshow("Imagine binara", binara)
cv2.waitKey(0)
cv2.imshow("Contur obiect (start = verde)", contur_img)
cv2.waitKey(0)
cv2.imshow("Contur reconstruit din codul inlantuit", rec_img)
cv2.waitKey(0)
cv2.destroyAllWindows()