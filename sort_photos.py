import cv2 as cv
import csv

path = "/data/train"
output = "C:/My folder/Projects/lost_planes/data/sorted_train"

with open('data/train.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        img = cv.imread(path + "/" + row[1] + ".png")
        cv.imwrite(output + "/" + row[0] + "/" + row[1] + ".png", img)

print("Done!")
