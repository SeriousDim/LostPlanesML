import cv2 as cv
from imutils import paths

path = "data/sorted_train/1"

imagePaths = list(paths.list_images(path))

c = 1
for p in imagePaths:
    image = cv.imread(p)
    image = cv.flip(image, 1)
    cv.imwrite("{0}/fliiped_{1}.png".format(path, c), image)
    c += 1
