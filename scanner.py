import cv2 as cv
import numpy as np
import typing

def scan_strip(img_path: str, num_swatches: int):
    img = cv.imread(img_path)

    img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

    img_blur = cv.GaussianBlur(img_gray,(3,3), sigmaX=0, sigmaY=0)


    cv.imshow('Blurred', img_blur)
    cv.waitKey(0)

    # Threshold
    ret, img_thresh = cv.threshold(img_blur,127,255,cv.THRESH_TOZERO)

    black_pixels = np.where(
        (img_thresh[:, :] == 0)
    )

    # set those pixels to white
    img_thresh[black_pixels] = [255]

    cv.imshow('Threshold', img_thresh)
    cv.waitKey(0)

    # Increase contrast
    alpha = 2
    beta = -255
    img_contrast = cv.addWeighted(img_thresh, alpha, img_thresh, 0, beta)
    img_contrast = cv.addWeighted(img_contrast, alpha, img_contrast, 0, beta)
    img_contrast = cv.addWeighted(img_contrast, alpha, img_contrast, 0, beta)

    cv.imshow('Contrast', img_contrast)
    cv.waitKey(0)

    # Threshold
    ret, img_contrast_thresh = cv.threshold(img_contrast,127,255,cv.THRESH_BINARY_INV)

    cv.imshow('Contrast and Threshold', img_contrast_thresh)
    cv.waitKey(0)

    kernel = np.ones((5,5),np.uint8)

    img_vert_mask = cv.erode(img_contrast_thresh, kernel, iterations = 1)

    cv.imshow('Vertical line mask', img_vert_mask)
    cv.waitKey(0)

    contours, hierarchy = cv.findContours(img_vert_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    img_cont = img.copy()

    for cnt in contours:
        cv.drawContours(img_cont, [cnt], 0, (0,0,255), 2)

    cv.imshow('Contours', img_cont)
    cv.waitKey(0)

    final = np.zeros(img.shape,np.uint8)
    mask = np.zeros(img_gray.shape,np.uint8)

    print(len(contours) == num_swatches)
    
    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 4
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)
    
    dominant = np.full(img.shape, palette[np.argmax(counts)])
    
    cv.imshow('Palette', dominant)
    cv.waitKey(0)

    colors = []
    colorskey = []

    for i in range(0,len(contours)):
        mask[...]=0
        cv.drawContours(mask,contours,i,255,-1)
        cv.drawContours(final,contours,i,cv.mean(img,mask),-1)
        M = cv.moments(contours[i])
        cY = int(M["m01"] / M["m00"])
        colors.append(cv.mean(img,mask))
        colorskey.append(cY)
        
        dominant = np.zeros(img.shape, np.uint8)
        dominant[:] = cv.mean(img, mask)[:-1]
        
        cv.imshow('Palette', dominant)
        cv.waitKey(0)

    colors = [x[0] for x in sorted(zip(colors, colorskey), key= lambda item: item[1])]

    cv.imshow('Colors', final)
    cv.waitKey(0)
    
if __name__ == '__main__':
    scan_strip('test.jpg', 4)