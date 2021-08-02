import cv2 as cv

def get_laplacian(img):
    return cv.Laplacian(img,cv.CV_64F)

def get_sobel(img, ksize = 3):
    return cv.Sobel(img,cv.CV_64F,1,1,ksize=ksize)
