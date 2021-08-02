import cv2 as cv
from borders import *

path = "data/test/02fdc282-e250-40c1-af0e-2668e499bea9.png"
img = cv.imread(path)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

r1 = get_laplacian(img)
r2 = get_sobel(img, ksize=3)

#r1 = cv.resize(r1, (400, 400), cv.INTER_AREA)
#r2 = cv.resize(r2, (400, 400), cv.INTER_CUBIC)

cv.imshow("laplacian", r1)
cv.imshow("sobel", r2)
cv.waitKey(0)
