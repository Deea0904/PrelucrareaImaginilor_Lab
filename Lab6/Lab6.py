import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r'flori.jpg'

L = 255  # nivelul maxim de gri


def histograma(img_gri):
    """
    h(g) = numarul de pixeli care au nivelul de gri g, g = 0..L
    """
    # 1. Numaram aparitiile fiecarui nivel de gri (parcurgere a intregii imagini)
    return np.bincount(img_gri.ravel(), minlength=L + 1).astype(np.int64)


def histograma_cumulativa(h):
    """
    hc[0] = h[0];  hc[i] = hc[i-1] + h[i]
    """
    # 1. Suma cumulata a histogramei
    return np.cumsum(h)


def valoare_medie(h, M):
    """
    Media nivelurilor de intensitate, calculata prin histograma:
    mu = (1/M) * sum_g g * h(g)
    """
    # 1. Nivelurile de gri g = 0..L
    g = np.arange(L + 1)

    # 2. Suma ponderata, impartita la numarul de pixeli
    return np.sum(g * h) / M


def deviatie_standard(h, mu, M):
    """
    Deviatia standard, calculata prin distributia statistica a nivelelor de gri:
    sigma = sqrt( sum_g (g - mu)^2 * p(g) ),  p(g) = h(g) / M
    """
    # 1. Nivelurile de gri si functia de densitate de probabilitate
    g = np.arange(L + 1)
    p = h / M

    # 2. Dispersia ponderata, apoi radacina patrata
    return np.sqrt(np.sum((g - mu) ** 2 * p))


def valoare_medie_directa(img_gri):
    """
    Calcul spatial direct: mu = (1/M) * sum_i sum_j I(i,j)
    """
    return img_gri.astype(np.float64).sum() / img_gri.size


def deviatie_standard_directa(img_gri, mu):
    """
    Calcul spatial direct: sigma = sqrt( (1/M) * sum_i sum_j (I(i,j) - mu)^2 )
    """
    return np.sqrt(((img_gri.astype(np.float64) - mu) ** 2).sum() / img_gri.size)


if __name__ == "__main__":
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Nu pot citi imaginea: {image_path}")

    # Lucram pe tonuri de gri
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    M = gri.size  # M = H * W

    h = histograma(gri)
    hc = histograma_cumulativa(h)

    mu = valoare_medie(h, M)
    sigma = deviatie_standard(h, mu, M)
    mu_d = valoare_medie_directa(gri)
    sigma_d = deviatie_standard_directa(gri, mu_d)

    print(f"Numar de pixeli M = {M}")
    print(f"Verificare normare: sum p(g) = {np.sum(h / M):.4f}")
    print(f"Media (prin histograma)        : {mu:.4f}")
    print(f"Media (calcul direct)          : {mu_d:.4f}")
    print(f"Dev. standard (prin histograma): {sigma:.4f}")
    print(f"Dev. standard (calcul direct)  : {sigma_d:.4f}")

    cv2.imshow("Imaginea originala", gri)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Histograma si histograma cumulativa
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.bar(np.arange(L + 1), h, width=1.0, color='black')
    plt.axvline(mu, color='red', label=f'media = {mu:.2f}')
    plt.axvspan(mu - sigma, mu + sigma, color='red', alpha=0.15,
                label=f'±sigma (sigma = {sigma:.2f})')
    plt.title("Histograma")
    plt.xlabel("Nivel de gri")
    plt.ylabel("Numar de pixeli")
    plt.xlim(0, L)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(L + 1), hc, color='blue')
    plt.title("Histograma cumulativa")
    plt.xlabel("Nivel de gri")
    plt.ylabel("Numar cumulat de pixeli")
    plt.xlim(0, L)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()