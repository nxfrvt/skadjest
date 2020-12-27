import numpy as np
import cv2
import imutils

img = cv2.imread('image.png')
img = cv2.resize(img, (620, 480))  # resizing so the resolution is always the same

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting to gray, just getting rid of nasty RGB values
img_gray = cv2.bilateralFilter(img_gray, 13, 35, 35)  # blurring out the background so it can be ignored later

img_edge = cv2.Canny(img, 30, 200)  # edge detection

contours = cv2.findContours(img_edge.copy(), cv2.RETR_TREE,
                             cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
screen_count = None

#  detecting plate in the image
for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approximation = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
    if len(approximation) == 4:  # approximated contour has four points
        screen_count = approximation
        cv2.drawContours(img, [screen_count], -1, (0, 255, 0), 3)
        break

#  putting a mask on area that is not a license plate
mask = np.zeros(img_gray.shape, np.uint8)
img_masked = cv2.drawContours(mask, [screen_count], 0, 255, -1,)
img_masked = cv2.bitwise_and(img, img, mask=mask)

#  cropping masked image
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
img_cropped = img_gray[topx:bottomx+1, topy:bottomy+1]

cv2.imshow('car', img_cropped)
cv2.waitKey(0)
