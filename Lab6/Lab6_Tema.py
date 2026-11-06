import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r'flori.jpg'

L = 255  # nivelul maxim de gri


def histograma(img_gri):
    """
    Pasul 1. h[g] = numarul de pixeli care au nivelul de gri g, g = 0..L
    """
    # 1. Numaram aparitiile fiecarui nivel de gri
    return np.bincount(img_gri.ravel(), minlength=L + 1).astype(np.int64)


def histograma_cumulativa(h):
    """
    Pasul 2. hc[0] = h[0];  hc[i] = hc[i-1] + h[i]
    """
    # 1. Suma cumulata a histogramei
    return np.cumsum(h)


def transformare_egalizare(hc, M):
    """
    Pasul 3. Transformarea y = T(x):
    T(x) = int[ (hc[x] - hc_min) / (M - hc_min) * L + 0.5 ]
    Intoarce tabelul T (look-up table) cu cate o valoare pentru fiecare nivel x.
    """
    # 1. hc_min = valoarea minima (nenula) a histogramei cumulative
    #    (hc la primul nivel de gri care apare efectiv in imagine)
    hc_min = hc[hc > 0].min()

    # 2. Noile nivele de gri pentru fiecare nivel vechi x = 0..L
    T = (hc - hc_min) / (M - hc_min) * L + 0.5

    # 3. Partea intreaga, cu valorile pastrate in [0, 255]
    return np.clip(T.astype(np.int32), 0, L).astype(np.uint8)


def egalizare_histograma(img_gri):
    """
    Egalizarea histogramei: fiecare pixel primeste noul nivel T(nivel_vechi).
    """
    # 1. Numarul de pixeli, histograma si histograma cumulativa
    M = img_gri.size
    h = histograma(img_gri)
    hc = histograma_cumulativa(h)

    # 2. Transformarea T, apoi aplicarea ei pe toti pixelii
    T = transformare_egalizare(hc, M)
    return T[img_gri]


if __name__ == "__main__":
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Nu pot citi imaginea: {image_path}")

    # Egalizarea lucreaza pe tonuri de gri
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    egalizata = egalizare_histograma(gri)

    cv2.imshow("Imaginea originala", gri)
    cv2.waitKey(0)
    cv2.imshow("Imaginea dupa egalizare", egalizata)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Histogramele corespunzatoare celor doua imagini
    titluri = ["Histograma imaginii originale", "Histograma dupa egalizare"]
    histograme = [histograma(gri), histograma(egalizata)]
    plt.figure(figsize=(12, 4))
    for idx, (titlu, h) in enumerate(zip(titluri, histograme), start=1):
        plt.subplot(1, 2, idx)
        plt.bar(np.arange(L + 1), h, width=1.0, color='black')
        plt.title(titlu)
        plt.xlabel("Nivel de gri")
        plt.ylabel("Numar de pixeli")
        plt.xlim(0, L)
    plt.tight_layout()
    plt.show()