#!/usr/bin/python

import os
import os.path
import cv2
import string
import sys
import numpy


def imggray(srcfile,dstdir):
    if not str(srcfile).endswith("jpg"):
        return

    f=file(srcfile)
    try:
        a=numpy.asarray(bytearray(f.read()),dtype=numpy.uint8)
        im=cv2.imdecode(a, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    finally:
        f.close()
    fname=os.path.join(dstdir,srcfile)
    rc=cv2.imwrite(fname,im)
    if not rc:
        raise Exception("Failed to write file {}".format(fname))


def getimgrect():
    cv2.getRectSubPix()
