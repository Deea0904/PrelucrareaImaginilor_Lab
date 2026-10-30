import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r'buchet.jpg'

L = 255  # nivelul maxim de gri

# Nuclee de convolutie de tip trece-sus
H_LAPLACE_1 = np.array([[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]], dtype=np.int32)
H_LAPLACE_2 = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]], dtype=np.int32)

H_HIGHPASS_1 = np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]], dtype=np.int32)
H_HIGHPASS_2 = np.array([[-1, -1, -1],
                         [-1, 9, -1],
                         [-1, -1, -1]], dtype=np.int32)

# Nucleu trece-jos folosit doar pentru netezirea prealabila (Laplace pe imagine netezita)
H_MEDIE_3 = np.ones((3, 3), dtype=np.int32)


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


def scalare_metoda1(conv, H):
    """
    Metoda 1: S+ = suma coeficientilor pozitivi, S- = suma modulelor celor negativi
              S = 1 / (2 * max(S+, S-))      I_D = S * (F * I_S) + floor(L/2)
    Rezultatul din [-L/2, L/2] este mutat in [0, L].
    """
    s_plus = int(H[H > 0].sum())
    s_minus = int(-H[H < 0].sum())
    S = 1.0 / (2 * max(s_plus, s_minus))
    rez = S * conv + (L // 2)
    return np.clip(np.round(rez), 0, L).astype(np.uint8)


def scalare_metoda2(conv):
    """
    Metoda 2: transformare liniara pe [min, max]:  D = L * (S - min) / (max - min)
    """
    I_min, I_max = int(conv.min()), int(conv.max())
    if I_max == I_min:          # rezultat constant
        return np.zeros(conv.shape, dtype=np.uint8)
    rez = L * (conv - I_min) / (I_max - I_min)
    return np.round(rez).astype(np.uint8)


def scalare_metoda3(conv):
    """
    Metoda 3: magnitudinea rezultatului, cu saturare la nivelul maxim L.
    """
    return np.clip(np.abs(conv), 0, L).astype(np.uint8)


def filtru_trece_sus(img_gri, H, metoda=3):
    """
    Filtru trece-sus: coeficienti pozitivi si negativi.
    Se face convolutia pe intregi cu semn, apoi rezultatul este adus
    in [0, 255] cu una dintre cele 3 metode.
    """
    # 1. Convolutie (intregi cu semn)
    conv = convolutie(img_gri, H)

    # 2. Aducerea rezultatului in [0, 255]
    if metoda == 1:
        return scalare_metoda1(conv, H)
    if metoda == 2:
        return scalare_metoda2(conv)
    return scalare_metoda3(conv)


if __name__ == "__main__":
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Nu pot citi imaginea: {image_path}")

    # Filtrarea lucreaza pe tonuri de gri
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Laplace (suma = 0) -> muchii pe fond negru (metoda 3)
    laplace1 = filtru_trece_sus(gri, H_LAPLACE_1, metoda=3)
    laplace2 = filtru_trece_sus(gri, H_LAPLACE_2, metoda=3)

    # Laplace pe imaginea netezita in prealabil cu media aritmetica 3x3
    netezita = np.clip(convolutie(gri, H_MEDIE_3) // 9, 0, L).astype(np.uint8)
    laplace_netezit = filtru_trece_sus(netezita, H_LAPLACE_2, metoda=3)

    # High-pass cu nucleul (4) (suma = 1) -> imagine cu muchii accentuate
    highpass1 = filtru_trece_sus(gri, H_HIGHPASS_1, metoda=3)
    highpass2 = filtru_trece_sus(gri, H_HIGHPASS_2, metoda=3)

    # Aceeasi imagine Laplace, cu cele 3 metode de scalare
    laplace_m1 = filtru_trece_sus(gri, H_LAPLACE_2, metoda=1)
    laplace_m2 = filtru_trece_sus(gri, H_LAPLACE_2, metoda=2)
    laplace_m3 = laplace2

    cv2.imshow("Imaginea originala", gri)
    cv2.waitKey(0)
    cv2.imshow("Laplace (nucleu cu 4)", laplace1)
    cv2.waitKey(0)
    cv2.imshow("Laplace (nucleu cu 8)", laplace2)
    cv2.waitKey(0)
    # cv2.imshow("Laplace pe imagine netezita", laplace_netezit)
    # cv2.waitKey(0)
    cv2.imshow("High-pass (nucleu cu 5)", highpass1)
    cv2.waitKey(0)
    cv2.imshow("High-pass (nucleu cu 9)", highpass2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # Comparatie intre filtre
    # titluri = ["Originala", "Laplace (4)", "Laplace (8)",
    #            "Laplace pe netezita", "High-pass (5)", "High-pass (9)"]
    # imagini = [gri, laplace1, laplace2, laplace_netezit, highpass1, highpass2]
    # plt.figure(figsize=(12, 7))
    # for idx, (titlu, im) in enumerate(zip(titluri, imagini), start=1):
    #     plt.subplot(2, 3, idx)
    #     plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    #     plt.title(titlu)
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.show()

    # # Comparatie intre cele 3 metode de scalare (Laplace cu nucleul cu 8)
    # titluri = ["Metoda 1 (S, +L/2)", "Metoda 2 (min-max)", "Metoda 3 (modul + saturare)"]
    # imagini = [laplace_m1, laplace_m2, laplace_m3]
    # plt.figure(figsize=(12, 4))
    # for idx, (titlu, im) in enumerate(zip(titluri, imagini), start=1):
    #     plt.subplot(1, 3, idx)
    #     plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    #     plt.title(titlu)
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.show()