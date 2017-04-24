#!/usr/bin/python
#指定文件名，根据标注文件中的信息截取目标，显示并存储
import cv2
import sys
import os.path
import xml.etree.ElementTree as et
import string

"""
   picname,a pure filename,exclude path '/'
"""
def get_np_rect(picname):
    pe=picname.split(".")
    fname="{}.xml".format(pe[0])
    datafile=os.path.join("./data",fname)
    tree=et.parse(datafile)
    folderelement=tree.find("object")
    bndelement=folderelement.find("bndbox")
    xmin=bndelement.find("xmin").text
    ymin=bndelement.find("ymin").text
    xmax=bndelement.find("xmax").text
    ymax=bndelement.find("ymax").text
    return xmin,ymin,xmax,ymax


"""
   usage cv2sample picfilename, exclude pathname
"""
if __name__ == "__main__":
    picname = sys.argv[1]
    picdir=os.path.join("./pic",picname)
    print picdir
    if not os.path.exists(picdir):
        print "{} is not exist".format(picdir)
        exit()
    xmin,ymin,xmax,ymax=get_np_rect(picname)
    w,h=int(xmax)-int(xmin),int(ymax)-int(ymin)
    img=cv2.imread(picdir,cv2.CV_LOAD_IMAGE_GRAYSCALE)
    zoom=cv2.getRectSubPix(img,(w,h),(int(xmin)+w/2,int(ymin)+h/2))
    zoom=cv2.resize(zoom,(256,256))
    cv2.imshow("test",zoom)
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
