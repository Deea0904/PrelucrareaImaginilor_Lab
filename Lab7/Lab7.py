
import time

import cv2
import numpy as np

image_path = r'flori.jpg'

SIGMA = 0.8  # deviatia standard a filtrului gaussian
SIGMA_ZGOMOT = 15  # deviatia standard a zgomotului gaussian adaugat


def dimensiune_nucleu(sigma):
    """
    Dimensiunea filtrului: w = 6 * sigma, rotunjita la cel mai apropiat numar impar
    (exemplu: sigma = 0.8 => w = 4.8 ~ 5)
    """
    # 1. w = 6 * sigma, rotunjit in sus
    w = int(np.ceil(6 * sigma))

    # 2. Nucleul trebuie sa aiba un centru, deci w impar
    if w % 2 == 0:
        w += 1
    return w


def nucleu_gaussian_2d(sigma):
    """
    G(x,y) = 1 / (2*pi*sigma^2) * exp( -((x-x0)^2 + (y-y0)^2) / (2*sigma^2) )
    unde (x0, y0) = linia si coloana centrala a nucleului
    """
    # 1. Dimensiunea si centrul nucleului
    w = dimensiune_nucleu(sigma)
    x0 = y0 = w // 2

    # 2. Calculam fiecare element al nucleului
    G = np.zeros((w, w), dtype=np.float64)
    for y in range(w):
        for x in range(w):
            G[y, x] = (1.0 / (2 * np.pi * sigma ** 2)) * \
                      np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # 3. Normalizare: suma elementelor = 1 (pastreaza luminozitatea imaginii)
    return G / G.sum()


def adauga_zgomot_gaussian(img_gri, mu, sigma):
    """
    Model de zgomot aditiv: f(i,j) = s(i,j) + n(i,j), n ~ N(mu, sigma^2)
    """
    # 1. Generam zgomotul gaussian
    n = np.random.normal(mu, sigma, img_gri.shape)

    # 2. Adunam zgomotul si ramanem in intervalul [0, 255]
    return np.clip(img_gri.astype(np.float64) + n, 0, 255).astype(np.uint8)


def convolutie_2d(img_gri, G):
    """
    I_D = G * I_S   (w*w inmultiri pentru fiecare pixel)
    Nucleul gaussian este simetric, deci convolutia coincide cu corelatia.
    """
    # 1. Dimensiuni imagine si nucleu
    h, wid = img_gri.shape
    w = G.shape[0]
    r = w // 2

    # 2. Extindem marginile prin replicarea pixelilor de la frontiera
    img_ext = np.pad(img_gri.astype(np.float64), r, mode='edge')

    # 3. Acumulam contributia fiecarui element G[i, j] al nucleului
    rezultat = np.zeros((h, wid), dtype=np.float64)
    for i in range(w):
        for j in range(w):
            rezultat += G[i, j] * img_ext[i:i + h, j:j + wid]

    # 4. Revenim la uint8
    return np.clip(np.rint(rezultat), 0, 255).astype(np.uint8)


if __name__ == "__main__":
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Nu pot citi imaginea: {image_path}")

    # Lucram pe tonuri de gri
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Imaginea afectata de zgomot gaussian
    zgomotoasa = adauga_zgomot_gaussian(gri, 0, SIGMA_ZGOMOT)

    # Nucleul gaussian bidimensional
    G = nucleu_gaussian_2d(SIGMA)

    # Filtrare + masurarea timpului de procesare
    t0 = time.perf_counter()
    filtrata = convolutie_2d(zgomotoasa, G)
    timp = (time.perf_counter() - t0) * 1000

    print(f"sigma = {SIGMA}, dimensiunea nucleului w = {G.shape[0]}")
    print("Nucleul gaussian 2D:")
    print(np.round(G, 4))
    print(f"Timp de procesare (nucleu 2D): {timp:.3f} ms")

    cv2.imshow("Imaginea originala", gri)
    cv2.waitKey(0)
    cv2.imshow("Imaginea cu zgomot gaussian", zgomotoasa)
    cv2.waitKey(0)
    cv2.imshow("Imaginea filtrata - nucleu gaussian 2D", filtrata)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


