import cv2
import numpy as np

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


if __name__ == "__main__":
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Nu pot citi imaginea: {image_path}")

    # Binarizarea lucreaza pe tonuri de gri
    gri = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    T, binara = binarizare_automata_globala(gri, eroare=0.1)
    print(f"Pragul final calculat: T = {T:.2f}")

    cv2.imshow("Imaginea originala", gri)
    key=cv2.waitKey(0)
    cv2.imshow(f"Imaginea binarizata (T = {T:.2f})", binara)

    cv2.waitKey(0)
    cv2.destroyAllWindows()