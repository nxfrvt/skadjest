import cv2

image = cv2.imread("data/tablica1.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv2.imshow("Treshold_img", threshold_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
