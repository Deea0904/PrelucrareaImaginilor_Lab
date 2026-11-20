import cv2
import numpy as np
import os
from pathlib import Path

# Lab 8 - Operatii morfologice pe imagini binare

image_path = r'./forma plina.jpg'

OBIECT, FUNDAL = 0, 255
B = np.ones((3, 3), dtype=bool)                          # element structural 8 vecini (eroziune)
B_CRUCE = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]], dtype=bool)              # element structural 4 vecini (umplere)


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


# Dilatare: pixelii acoperiti de B devin obiect (SAU intre versiunile deplasate)
def dilatare(A, B):
    k = B.shape[0] // 2
    extinsa = np.pad(A, k, constant_values=False)
    M, N = A.shape
    rezultat = np.zeros_like(A)
    for u, v in zip(*np.nonzero(B)):
        rezultat |= extinsa[u:u + M, v:v + N]
    return rezultat


# Extragerea conturului: beta(A) = A - (A (-) B)
def extragere_contur(A, B):
    return A & ~eroziune(A, B)


# Umplerea regiunilor: X_k = (X_{k-1} (+) B) ∩ A^C, pornind de la X_0 = p
def umplere_regiuni(A, B, p):
    Ac = ~A                                   # complementul lui A
    X = np.zeros_like(A)
    X[p] = True                               # X_0 = p
    iteratii = 0
    while True:
        X_nou = dilatare(X, B) & Ac           # X_k
        iteratii += 1
        if np.array_equal(X_nou, X):          # X_k = X_{k-1} -> stop
            break
        X = X_nou
    return X | A, iteratii                    # X_k ∪ A


# masca booleana -> imagine (obiect negru pe fundal alb)
def la_imagine(masca):
    return np.where(masca, OBIECT, FUNDAL).astype(np.uint8)


os.chdir(Path(__file__).resolve().parent)

img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Nu pot citi imaginea: {image_path}")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Etapa 1: conturul
A = binarizare(gray)
erodata = eroziune(A, B)
contur = extragere_contur(A, B)

# Etapa 5: punct p din interiorul regiunii (un pixel din interiorul erodat)
interior = np.argwhere(erodata)
if len(interior) == 0:
    raise ValueError("Obiectul e prea subtire: nu exista pixeli interiori pentru p.")
p = tuple(interior[0])

# Etapele 2-7: umplerea pornind DOAR de la contur
umplut, iteratii = umplere_regiuni(contur, B_CRUCE, p)

print(f'p = {p}, iteratii: {iteratii}')
print(f'Pixeli contur: {contur.sum()}, dupa umplere: {umplut.sum()}, obiect original: {A.sum()}')

cv2.imshow("Imagine originala", gray)
cv2.waitKey(0)
cv2.imshow("Contur beta(A)", la_imagine(contur))
cv2.waitKey(0)
cv2.imshow("Umplere regiune (X_k U A)", la_imagine(umplut))
cv2.waitKey(0)
cv2.destroyAllWindows()