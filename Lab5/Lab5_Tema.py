import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r'flori.jpg'

L = 255  # nivelul maxim de gri

R = 30   # raza filtrului ideal (in pixeli de frecventa)
A = 30   # parametrul filtrului gaussian (A ~ 1/sigma din domeniul spatial)


def centrare(img):
    """
    Transformarea de centrare: x(u,v) <- (-1)^(u+v) * x(u,v)
    Aplicata de doua ori, revine la imaginea initiala.
    """
    # 1. Indicii liniilor (u) si ai coloanelor (v)
    h_img, w_img = img.shape
    u, v = np.indices((h_img, w_img))

    # 2. Semnul (-1)^(u+v) pentru fiecare pixel
    semn = np.where((u + v) % 2 == 0, 1, -1)

    return img * semn


def dft(img_gri):
    """
    Pasii a) si b): centrarea imaginii originale, apoi DFT 2D (prin FFT).
    Componentele de frecventa joasa ajung in mijlocul spectrului.
    """
    # 1. Lucram pe numere reale (float), ca sa putem inmulti cu +-1
    centrata = centrare(img_gri.astype(np.float64))

    # 2. Transformata Fourier discreta bidimensionala
    return np.fft.fft2(centrata)


def idft(X):
    """
    Pasii d) si e): IDFT 2D, apoi centrarea (anuleaza efectul primei centrari).
    """
    # 1. Inversa transformatei Fourier, partea reala
    imagine = np.real(np.fft.ifft2(X))

    # 2. Anulam centrarea
    return centrare(imagine)


def spectru(X):
    """
    Spectrul de magnitudine, in scara logaritmica, adus in [0, 255]:
    S = log(1 + |X|)
    """
    # 1. Logaritmul magnitudinii (altfel componenta continua acopera restul)
    s = np.log1p(np.abs(X))

    # 2. Normalizare in [0, 255] pentru afisare
    s = (s - s.min()) / (s.max() - s.min()) * L
    return s.astype(np.uint8)


def normalizare(img):
    """
    Dupa filtrarea trece-sus componenta continua dispare, deci media devine ~0
    si apar valori negative. Aducem rezultatul in [0, 255] pentru afisare.
    """
    return ((img - img.min()) / (img.max() - img.min()) * L).astype(np.uint8)


def distanta_patrat(shape):
    """
    D^2(u,v) = (H/2 - u)^2 + (W/2 - v)^2   (fata de centrul imaginii)
    """
    h_img, w_img = shape
    u, v = np.indices((h_img, w_img))
    return (h_img / 2 - u) ** 2 + (w_img / 2 - v) ** 2


def filtru_ideal_trece_sus(img_gri, raza):
    """
    X'(u,v) = X(u,v)  daca D^2 > R^2, altfel 0
    """
    # 1. Centrare + DFT
    X = dft(img_gri)

    # 2. Masca: eliminam frecventele din interiorul cercului de raza R
    masca = distanta_patrat(img_gri.shape) > raza ** 2
    X_filtrat = X * masca

    # 3. IDFT + centrare, apoi normalizare pentru afisare
    rez = idft(X_filtrat)
    return normalizare(rez), X_filtrat


def filtru_gaussian_trece_sus(img_gri, a):
    """
    X'(u,v) = X(u,v) * (1 - exp( -D^2 / A^2 ))
    Suprimare neteda a frecventelor joase, fara efectul de unde circulare.
    """
    # 1. Centrare + DFT
    X = dft(img_gri)

    # 2. Atenuare gaussiana a frecventelor apropiate de centru
    H = 1 - np.exp(-distanta_patrat(img_gri.shape) / a ** 2)
    X_filtrat = X * H

    # 3. IDFT + centrare, apoi normalizare pentru afisare
    rez = idft(X_filtrat)
    return normalizare(rez), X_filtrat


if __name__ == "__main__":
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Nu pot citi imaginea: {image_path}")

    # Filtrarea lucreaza pe tonuri de gri
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ideal, X_ideal = filtru_ideal_trece_sus(gri, R)
    gauss, X_gauss = filtru_gaussian_trece_sus(gri, A)

    cv2.imshow("Imaginea originala", gri)
    cv2.waitKey(0)
    cv2.imshow("Spectrul imaginii originale", spectru(dft(gri)))
    cv2.waitKey(0)
    cv2.imshow("Filtru ideal trece-sus", ideal)
    cv2.waitKey(0)
    cv2.imshow("Spectrul dupa filtrul ideal", spectru(X_ideal))
    cv2.waitKey(0)
    cv2.imshow("Filtru gaussian trece-sus", gauss)
    cv2.waitKey(0)
    cv2.imshow("Spectrul dupa filtrul gaussian", spectru(X_gauss))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # Comparatie intre rezultate
    # titluri = ["Originala", "Ideal trece-sus", "Gaussian trece-sus"]
    # imagini = [gri, ideal, gauss]
    # spectre = [spectru(dft(gri)), spectru(X_ideal), spectru(X_gauss)]
    # plt.figure(figsize=(12, 8))
    # for idx, (titlu, im, sp) in enumerate(zip(titluri, imagini, spectre), start=1):
    #     plt.subplot(2, 3, idx)
    #     plt.imshow(im, cmap='gray', vmin=0, vmax=255)
    #     plt.title(titlu)
    #     plt.axis('off')
    #     plt.subplot(2, 3, idx + 3)
    #     plt.imshow(sp, cmap='gray', vmin=0, vmax=255)
    #     plt.title("Spectru")
    #     plt.axis('off')
    # plt.tight_layout()
    # plt.show()