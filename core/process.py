import cv2
import sys

import numpy as np
from skimage import measure, color
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.measure import regionprops


def get_closing(image):
    '''
    形态学闭操作
    :param image:
    :return:
    '''
    kernel = np.ones((15, 15), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def get_hough_lines(write_image):
    '''
    Canny算子进行边缘检测，然后进行累计概率霍夫变换找线条，加粗，加封闭。
    :param write_image:
    :return:
    '''
    changed_image = write_image.copy()
    GrayImage = cv2.cvtColor(write_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(GrayImage, 0, 150, apertureSize=3)
    minLineLength = 5
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength, maxLineGap)
    min_x1 = 999
    max_x2 = 0
    min_y1 = 999
    max_y2 = 0

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(write_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if x1 < min_x1:
                min_x1 = x1
            if x2 > max_x2:
                max_x2 = x2
            if y1 > min_y1:
                min_y1 = y1
            if y2 < max_y2:
                max_y2 = y2

    print(f"minx:{min_x1},maxx:{max_x2},miny:{min_y1},maxy:{max_y2}")
    for line in lines:
        for x1, y1, x2, y2 in line:
            if -10 < y1 - y2 < 10:
                cv2.line(changed_image, (min_x1, y1), (max_x2, y2), (0, 0, 0), 2)
            if -10 < x2 - x1 < 10:
                cv2.line(changed_image, (x1, min_y1), (x2, max_y2), (0, 0, 0), 2)

    cv2.line(changed_image, (min_x1, 0), (min_x1, changed_image.shape[0]), (0, 0, 0), 2)
    cv2.line(changed_image, (max_x2, 0), (max_x2, changed_image.shape[0]), (0, 0, 0), 2)
    return changed_image, write_image


def get_thresh(image):
    '''
    阈值操作
    :param image:
    :return:
    '''
    GrayImage = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)
    # THRESH_BINARY 代表 value = value > threshold ? max_value: 0
    _, th = cv2.threshold(GrayImage, 150, 255, cv2.THRESH_BINARY)
    return th


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Proper usage : {sys.argv[0]} <image_file_path>")

    image = cv2.imread(sys.argv[1])

    max_area = image.shape[0] * image.shape[1] * 0.1

    changed_image, hough_lines = get_hough_lines(image.copy())

    closing = get_closing(changed_image)

    th = get_thresh(closing)

    # cv2.imshow('changed_image', changed_image)
    # cv2.waitKey()

    # cv2.imwrite(f'{sys.argv[1][:-4]}_houghlines.jpg', hough_lines)

    labels = measure.label(th, connectivity=1)
    dst = color.label2rgb(labels)

    f, ax1 = plt.subplots(1, 1, figsize=(16, 9))
    ax1.imshow(image)
    # ax2.imshow(dst)

    rects = []

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = .8
    fontColor = (0, 255, 0)
    lineType = 1

    cnt = 0

    if labels.max() + 1 < 5:
        print('这张图片不是表格图')
        sys.exit(1)

    for region in regionprops(labels):
        # take regions with large enough areas
        if region.area < max_area:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            # cv2.putText(image, str(region.area), (minc, minr + 30), font, fontScale, fontColor, lineType)
            rects.append({'x': minc, 'y': minr, 'w': maxc - minc, 'h': maxr - minr})
            ax1.add_patch(rect)
            cv2.rectangle(image, (minc, minr), (maxc, maxr), (0, 0, 255), 2)
        else:
            cnt += 1
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            rect_dict = {
                'x': minc,
                'y': minr,
                'w': maxc - minc,
                'h': maxr - minr
            }
            rects.append(rect_dict)
            cv2.putText(image, str(region.area), (minc, minr + 30), font, fontScale, (0, 0, 255), lineType)

    # ax3.imshow(image)
    cv2.imshow('image', image)
    cv2.waitKey()
    print(rects)

    plt.tight_layout()
    plt.show()
