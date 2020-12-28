import numpy as np
import cv2
import imutils
from operator import itemgetter


def find_license_plate(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (620, 480))  # resizing so the resolution is always the same

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting to gray, just getting rid of nasty RGB values
    img_gray = cv2.bilateralFilter(img_gray, 13, 35, 35)  # blurring out the background so it can be ignored later

    img_edge = cv2.Canny(img, 30, 200)  # edge detection

    #  finding all the contours in the image, then sorting them and taking only first 10 of them
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
    img_masked = cv2.drawContours(mask, [screen_count], 0, 255, -1, )
    img_masked = cv2.bitwise_and(img, img, mask=mask)

    #  cropping masked image
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    img_cropped = img_gray[topx:bottomx + 1, topy:bottomy + 1]
    return __extract_chars(img_cropped)


def __extract_chars(plate):
    plate = cv2.resize(plate, (plate.shape[1] * 2, plate.shape[0] * 2))

    plate_contrast = (255 / 1) * (plate / (255 / 1)) ** 2
    plate_contrast = np.array(plate_contrast, dtype=np.uint8)

    plate_edge = cv2.Canny(plate_contrast, 30, 200)
    contours, _ = cv2.findContours(plate_edge.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    # cv2.drawContours(plate, contours, -1, (0, 255, 0), 2)

    #  create one bounding box for every contour found
    bb_list = []
    for c in contours:
        bb = cv2.boundingRect(c)
        bb_list.append(bb)

    # removing rectangles that overlap existing ones
    bb_list = __remove_overlaps(bb_list)
    # removing rectangles that are inside of other rectangles
    bb_list = __remove_insides(bb_list)

    bb_list.sort(key=itemgetter(0))  # we do this so rectangles are in order from right to left

    # debug: draw boxes
    img_boxes = plate.copy()
    label = 1
    for bb in bb_list:
        x, y, w, h = bb
        cv2.rectangle(img_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
        img_cropped = plate[y:y+h, x:x+w]
        cv2.imwrite('tmp/' + str(label) + '.jpg', img_cropped)
        label += 1

    return img_boxes


def __remove_overlaps(bb_list):
    threshold = 5  # próg kiedy rectangle zostaje odrzucony jako taki sam, 5 wydaje sie git
    index = 0
    while index < len(bb_list):
        rect = bb_list[index]
        for other_rect in bb_list[index + 1:]:
            other_substracted = tuple(np.subtract(other_rect, (threshold, threshold, threshold, threshold)))
            other_added = tuple(np.add(other_rect, (threshold, threshold, threshold, threshold)))
            if other_substracted < rect < other_added:
                bb_list.remove(other_rect)
                index -= 1
        index += 1
    return bb_list


def __remove_insides(bb_list):
    index = 0
    while index < len(bb_list):
        x1, y1, w1, h1 = bb_list[index]
        for other_rect in bb_list[index + 1:]:
            x2, y2, w2, h2 = other_rect
            if ((x1 < x2) and
                    (y1 < y2) and
                    (x1 + w1 > x2 + w2) and
                    (y1 + h1 > y2 + h2)):
                bb_list.remove(other_rect)
                index -= 1
        index += 1
    return bb_list


if __name__ == '__main__':
    cv2.imshow('car', find_license_plate('rsc/photos/tab_004.jpg'))
    cv2.waitKey(0)
