import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = r'food.jpg'


def binarizare_automata_globala(img_gri, eroare=0.1):

    # 1a. Histograma h(f), f = 0..255
    h = np.bincount(img_gri.ravel(), minlength=256).astype(np.float64)
    nivele = np.arange(256, dtype=np.float64)

    # 1b. Intensitatea minima si maxima
    I_min = int(img_gri.min())
    I_max = int(img_gri.max())

    # 1c. Valoarea initiala a pragului
    T = (I_max + I_min) / 2.0

    while True:
        t = int(T)

        # 2. G1: f in [I_min, T]   |   G2: f in [T+1, I_max]
        h1, f1 = h[I_min:t + 1], nivele[I_min:t + 1]
        h2, f2 = h[t + 1:I_max + 1], nivele[t + 1:I_max + 1]

        N1, N2 = h1.sum(), h2.sum()
        if N1 == 0 or N2 == 0:      # imagine (aproape) constanta
            break

        mu1 = (f1 * h1).sum() / N1  # media clasei G1
        mu2 = (f2 * h2).sum() / N2  # media clasei G2

        # 3. Actualizare prag
        T_nou = (mu1 + mu2) / 2.0

        # 4. Conditia de oprire: |T_k - T_(k-1)| < eroare
        if abs(T_nou - T) < eroare:
            T = T_nou
            break
        T = T_nou

    # 5. Binarizare cu pragul final
    img_bin = np.where(img_gri > T, 255, 0).astype(np.uint8)
    return T, img_bin


def construieste_histograma(img_gri, T, inaltime=300):
    """Deseneaza histograma imaginii (bare gri) si pragul T (linie rosie)."""
    h = np.bincount(img_gri.ravel(), minlength=256).astype(np.float64)
    latime_bin = 2
    x0 = 10  # margine stanga
    y_baza = inaltime - 25  # linia de baza a histogramei
    h_norm = h / h.max() * (y_baza - 35)  # scalare pe inaltime

    canvas = np.full((inaltime, 256 * latime_bin + 2 * x0, 3), 255, np.uint8)

    for f in range(256):
        x = x0 + f * latime_bin
        cv2.rectangle(canvas, (x, y_baza - int(h_norm[f])),
                      (x + latime_bin - 1, y_baza), (150, 150, 150), -1)

    # Axa si etichetele nivelurilor de gri
    cv2.line(canvas, (x0, y_baza), (x0 + 256 * latime_bin, y_baza), (0, 0, 0), 1)
    for f in (0, 128, 255):
        cv2.putText(canvas, str(f), (x0 + f * latime_bin - 8, inaltime - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Pragul final T (linie rosie)
    xT = x0 + int(round(T)) * latime_bin
    cv2.line(canvas, (xT, 10), (xT, y_baza), (0, 0, 255), 2)
    cv2.putText(canvas, f"T = {T:.2f}", (xT + 6, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return canvas

if __name__ == "__main__":
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Nu pot citi imaginea: {image_path}")

    # Binarizarea lucreaza pe tonuri de gri
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    T, binara = binarizare_automata_globala(gri, eroare=0.1)
    print(f"Pragul final calculat: T = {T:.2f}")

    # Histograma (ca la Lab2) + pragul final T


    cv2.imshow("Imaginea originala", gri)
    cv2.waitKey(0)
    cv2.imshow(f"Imaginea binarizata (T = {T:.2f})", binara)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    plt.hist(gri.ravel(), 256, [0, 256])
    plt.axvline(T, color='r', linestyle='--', linewidth=2, label=f'Prag T = {T:.2f}')
    plt.title("Histograma imaginii in tonuri de gri")
    plt.legend()
    plt.show()



