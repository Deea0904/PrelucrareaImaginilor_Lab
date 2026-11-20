import time

import cv2
import numpy as np

image_path = r'flori.jpg'

SIGMA = 0.8          # deviatia standard a filtrului gaussian
SIGMA_ZGOMOT = 15    # deviatia standard a zgomotului gaussian adaugat


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


def nucleu_gaussian_1d(sigma):
    """
    G(x) = 1 / (sqrt(2*pi)*sigma) * exp( -(x-x0)^2 / (2*sigma^2) )
    = vectorul liniei (Gx) / coloanei (Gy) centrale ale nucleului bidimensional
    """
    # 1. Dimensiunea si centrul nucleului
    w = dimensiune_nucleu(sigma)
    x0 = w // 2

    # 2. Calculam fiecare element al vectorului
    G = np.zeros(w, dtype=np.float64)
    for x in range(w):
        G[x] = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * \
            np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

    # 3. Normalizare: suma elementelor = 1
    return G / G.sum()


def adauga_zgomot_gaussian(img_gri, mu, sigma):
    """
    Model de zgomot aditiv: f(i,j) = s(i,j) + n(i,j), n ~ N(mu, sigma^2)
    """
    # 1. Generam zgomotul gaussian
    n = np.random.normal(mu, sigma, img_gri.shape)

    # 2. Adunam zgomotul si ramanem in intervalul [0, 255]
    return np.clip(img_gri.astype(np.float64) + n, 0, 255).astype(np.uint8)


def convolutie_separabila(img_gri, Gx, Gy):
    """
    Separabilitatea functiei gaussiene: G(x,y) = G(x) * G(y)
    I_D = (Gx Gy) * I_S = Gx * (Gy * I_S)
    => doua convolutii 1D, cu w inmultiri pentru fiecare pixel la fiecare parcurgere
    """
    # 1. Dimensiuni imagine si nucleu
    h, wid = img_gri.shape
    w = len(Gy)
    r = w // 2

    # 2. Extindem marginile prin replicarea pixelilor de la frontiera
    img_ext = np.pad(img_gri.astype(np.float64), r, mode='edge')

    # 3. Prima parcurgere: pe verticala, cu Gy
    temp = np.zeros((h, wid + 2 * r), dtype=np.float64)
    for i in range(w):
        temp += Gy[i] * img_ext[i:i + h, :]

    # 4. A doua parcurgere: pe orizontala, cu Gx
    rezultat = np.zeros((h, wid), dtype=np.float64)
    for j in range(w):
        rezultat += Gx[j] * temp[:, j:j + wid]

    # 5. Revenim la uint8
    return np.clip(np.rint(rezultat), 0, 255).astype(np.uint8)


if __name__ == "__main__":
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Nu pot citi imaginea: {image_path}")

    # Lucram pe tonuri de gri
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Imaginea afectata de zgomot gaussian
    zgomotoasa = adauga_zgomot_gaussian(gri, 0, SIGMA_ZGOMOT)

    # Vectorii gaussieni 1D (linia si coloana centrala a nucleului 2D)
    Gx = nucleu_gaussian_1d(SIGMA)
    Gy = nucleu_gaussian_1d(SIGMA)

    # Filtrare + masurarea timpului de procesare
    t0 = time.perf_counter()
    filtrata = convolutie_separabila(zgomotoasa, Gx, Gy)
    timp = (time.perf_counter() - t0) * 1000

    print(f"sigma = {SIGMA}, dimensiunea nucleului w = {len(Gx)}")
    print(f"Gx = Gy = {np.round(Gx, 4)}")
    print(f"Timp de procesare (nucleu separabil): {timp:.3f} ms")

    cv2.imshow("Imaginea originala", gri)
    cv2.waitKey(0)
    cv2.imshow("Imaginea cu zgomot gaussian", zgomotoasa)
    cv2.waitKey(0)
    cv2.imshow("Imaginea filtrata - nucleu gaussian separabil", filtrata)
    cv2.waitKey(0)
    cv2.destroyAllWindows()