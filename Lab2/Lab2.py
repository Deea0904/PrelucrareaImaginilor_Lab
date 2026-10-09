import cv2
import matplotlib.pyplot as plt
img = cv2.imread('Image.jpg')
hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('Original', img)
cv2.imshow('HSV image', hsvImage)
cv2.waitKey()


imgGray = cv2.imread('Image.jpg', 0)   # 0 = grayscale
plt.hist(imgGray.ravel(), 256, [0, 256])
plt.title("Histograma imaginii in tonuri de gri")
plt.show()

color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.title("Histograma imaginii color")
plt.show()