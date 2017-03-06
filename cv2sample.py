#!/usr/bin/python

import cv2
import sys
import os.path


if __name__ == "__main__":
    picdir = sys.argv[1]
    print picdir
    if not os.path.exists(picdir):
        print "{} is not exist".format(picdir)
        exit()

    img=cv2.imread(picdir,cv2.CV_LOAD_IMAGE_GRAYSCALE)
    cv2.imshow("test",img)
    cv2
    k=cv2.waitKey(0)
    if k==27: #esc
        cv2.destroyAllWindows()
    else:
        cv2.destroyAllWindows()

"""
    def onmouse(event, x, y, flags, param):
        h, w = img.shape[:2]
        h1, w1 = small.shape[:2]
        x, y = 1.0*x*h/h1, 1.0*y*h/h1
        zoom = cv2.getRectSubPix(img, (800, 600), (x+0.5, y+0.5))
        cv2.imshow('zoom', zoom)
"""