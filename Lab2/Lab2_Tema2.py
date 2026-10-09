import numpy as np
import cv2


# 1. Citirea imaginii in tonuri de gri si calculul FDP
img = cv2.imread('Image.jpg', 0)
hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
M = img.shape[0] * img.shape[1]
fdp = hist / M



# 2. Functiile la fel ca la tema 1
def gaseste_maxime(fdp, WH=5, TH=0.0003):
    maxime = [0]
    for k in range(WH, 255 - WH + 1):
        fereastra = fdp[k - WH: k + WH + 1]
        v = fereastra.mean()
        if fdp[k] > v + TH and fdp[k] >= fereastra.max():
            maxime.append(k)
    maxime.append(255)
    return maxime


def calc_praguri(maxime):
    praguri = []
    for i in range(len(maxime) - 1):
        praguri.append((maxime[i] + maxime[i + 1]) / 2)
    return praguri


def cel_mai_apropiat_maxim(valoare, maxime):
    cel_mai_apropiat = maxime[0]
    dist_min = abs(valoare - maxime[0])
    for m in maxime:
        d = abs(valoare - m)
        if d < dist_min:
            dist_min = d
            cel_mai_apropiat = m
    return cel_mai_apropiat


def reduce_niveluri(img, maxime):
    h, w = img.shape
    rezultat = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            rezultat[y, x] = cel_mai_apropiat_maxim(int(img[y, x]), maxime)
    return rezultat


#_______________________________________________________________________________
def floyd_steinberg(img, maxime):
    h, w = img.shape
    pixel = img.astype(np.float32)

    for y in range(h):
        for x in range(w):
            pixel_vechi = pixel[y, x]
            pixel_nou = cel_mai_apropiat_maxim(pixel_vechi, maxime)
            pixel[y, x] = pixel_nou
            eroare = pixel_vechi - pixel_nou

            if x + 1 < w:
                pixel[y, x + 1] = pixel[y, x + 1] + 7 * eroare / 16
            if y + 1 < h:
                if x - 1 >= 0:
                    pixel[y + 1, x - 1] = pixel[y + 1, x - 1] + 3 * eroare / 16
                pixel[y + 1, x] = pixel[y + 1, x] + 5 * eroare / 16
                if x + 1 < w:
                    pixel[y + 1, x + 1] = pixel[y + 1, x + 1] + eroare / 16

    return pixel.astype(np.uint8)



# 4. Rulare si afisare

maxime = gaseste_maxime(fdp)
praguri = calc_praguri(maxime)

print("Maxime:", maxime)
print("Praguri:", praguri)

img_cuantizata = reduce_niveluri(img, maxime)
img_fs = floyd_steinberg(img, maxime)

cv2.imshow('Imaginea originala', img)
cv2.imshow('Imaginea cuantizata (praguri multiple)', img_cuantizata)
cv2.imshow('Floyd-Steinberg', img_fs)
cv2.waitKey()
cv2.destroyAllWindows()