import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r'buchet.jpg'

L = 255  # nivelul maxim de gri

# Nuclee de convolutie de tip trece-jos
H_MEDIE_3 = np.ones((3, 3), dtype=np.int32)
H_MEDIE_5 = np.ones((5, 5), dtype=np.int32)
H_GAUSS_3 = np.array([[1, 2, 1],
                      [2, 4, 2],
                      [1, 2, 1]], dtype=np.int32)


def convolutie(img_gri, H):
    """
    I_D(i,j) = sum_u sum_v H(u,v) * I_S(i+u-k, j+v-k),   w = 2k+1
    Se ignora primele si ultimele k linii si coloane (raman copiate din sursa).
    Rezultatul este int32 (cu semn), FARA scalare.
    """
    # 1a. Dimensiunile imaginii si ale nucleului
    h_img, w_img = img_gri.shape
    w = H.shape[0]
    k = w // 2

    # 1b. Lucram pe intregi cu semn, ca sa nu apara depasiri uint8
    sursa = img_gri.astype(np.int32)
    rez = sursa.copy()
    rez[k:h_img - k, k:w_img - k] = 0

    # 2. Pentru fiecare element H(u,v) adunam imaginea sursa translatata
    for u in range(w):
        for v in range(w):
            rez[k:h_img - k, k:w_img - k] += \
                H[u, v] * sursa[u:h_img - w + 1 + u, v:w_img - w + 1 + v]
    return rez


def filtru_trece_jos(img_gri, H):
    """
    Filtru trece-jos: coeficienti pozitivi, rezultatul se imparte la
    c = suma(H). Impartirea este ULTIMA operatie (erori de rotunjire minime).
    """
    # 1. Suma coeficientilor nucleului
    c = int(H.sum())

    # 2. Convolutie (intregi), apoi impartire intreaga la final
    rez = convolutie(img_gri, H) // c

    # 3. Ne asiguram ca valorile sunt in [0, 255]
    return np.clip(rez, 0, L).astype(np.uint8)


if __name__ == "__main__":
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Nu pot citi imaginea: {image_path}")

    # Filtrarea lucreaza pe tonuri de gri
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    medie3 = filtru_trece_jos(gri, H_MEDIE_3)
    medie5 = filtru_trece_jos(gri, H_MEDIE_5)
    gauss3 = filtru_trece_jos(gri, H_GAUSS_3)

    cv2.imshow("Imaginea originala", gri)
    cv2.waitKey(0)
    cv2.imshow("Filtru medie aritmetica 3x3", medie3)
    cv2.waitKey(0)
    cv2.imshow("Filtru medie aritmetica 5x5", medie5)
    cv2.waitKey(0)
    cv2.imshow("Filtru gaussian 3x3", gauss3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # Comparatie intre rezultate
    # titluri = ["Originala", "Medie 3x3", "Medie 5x5", "Gaussian 3x3"]
    # imagini = [gri, medie3, medie5, gauss3]
    # plt.figure(figsize=(12, 4))
    # for idx, (titlu, im) in enumerate(zip(titluri, imagini), start=1):
    #     plt.subplot(1, 4, idx)
    #     plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    #     plt.title(titlu)
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.show()