import cv2
import os

# Image path
image_path = r'Flower.jpg'
# Image directory
directory = r'./Imagini'

# Image read
img = cv2.imread(image_path)
if img is None:
    print("Image not exist")
    exit()

cv2.imshow('Image', img)
key = cv2.waitKey(0)
if key == 27:  # if ESC is pressed, exit
    cv2.destroyAllWindows()

# Image print
print("Before saving image:")
print(os.listdir(directory))

# Save new image
filename = 'SavedFlower.jpg'
cv2.imwrite(os.path.join(directory, filename), img)
print("After saving image:")
print(os.listdir(directory))
print('Successfully saved')

#Color Image to GrayScale Image
gray =cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray image',gray )
key = cv2.waitKey()

filename = 'SavedFlowerGray.jpg'
cv2.imwrite(os.path.join(directory, filename), gray)
print("After saving image:")
print(os.listdir(directory))
print('Successfully saved')


#Black&white Image
(thresh,BlackAndWhiteImage) = cv2.threshold (gray,150,255,cv2.THRESH_BINARY)
cv2.imshow( 'Black white image ', BlackAndWhiteImage)
key = cv2.waitKey()

filename = 'SavedFlowerBlackAndWhite.jpg'
cv2.imwrite(os.path.join(directory, filename), BlackAndWhiteImage)
print("After saving image:")
print(os.listdir(directory))
print('Successfully saved')