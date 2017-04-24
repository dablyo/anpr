#!/usr/bin/python
#accroding to position/number info from annottion xml file, get roi and write img file
import cv2
import sys
import os.path
import xml.etree.ElementTree as et
import shutil


IAMGE_DIR="/home/wang/git/nppic/su_np_color"
DATA_DIR="/home/wang/git/nppic/su_np_annotation"
ZM_DIR="/home/wang/git/nppic/su_np_color-256x128"

"""
   picname,a pure filename,exclude path '/'
"""
def get_np_rect(picname):
    pe=picname.split(".")
    #fname="{}.xml".format(pe[0][:len(pe[0])-5])   #for 20170218_091624_gray.jpg
    fname="{}.xml".format(pe[0])   #for 20170218_091624.jpg
    
    datafile=os.path.join(DATA_DIR,fname)
    if not os.path.exists(datafile):
        return None,None,None,None,None
    tree=et.parse(datafile)
    folderelement=tree.find("object")
    num=folderelement.find("name").text
    bndelement=folderelement.find("bndbox")
    xmin=bndelement.find("xmin").text
    ymin=bndelement.find("ymin").text
    xmax=bndelement.find("xmax").text
    ymax=bndelement.find("ymax").text

    return num,xmin,ymin,xmax,ymax

def gen_a_sam(idx,picname):
    picdir=os.path.join(IAMGE_DIR,picname)
    if not os.path.exists(picdir):
        print "{} is not exist".format(picdir)
        return
    num,xmin,ymin,xmax,ymax=get_np_rect(picname)
    if num ==None:
        return
    w,h=int(xmax)-int(xmin),int(ymax)-int(ymin)
    #img=cv2.imread(picdir,cv2.CV_LOAD_IMAGE_GRAYSCALE)
    img=cv2.imread(picdir)
    zoom=cv2.getRectSubPix(img,(w,h),(int(xmin)+w/2,int(ymin)+h/2))
    #zoom=cv2.resize(zoom,(256,128))
    #cv2.imshow("tt",zoom)
    #kk=cv2.waitKey(0)
    #if kk==27:
    #    cv2.destroyAllWindows()

    fname="%08d_0%s_1.png" % (int(idx),num[1:])
    dstfname=os.path.join(ZM_DIR,fname)
    cv2.imwrite(dstfname,zoom)
    print "{} has been created".format(dstfname)
    return

if __name__ == "__main__":
    if os.path.exists(ZM_DIR):
        shutil.rmtree(ZM_DIR)
    os.mkdir(ZM_DIR)
    i=0
    for parent,dirnames,filenames in os.walk(IAMGE_DIR):
        for filename in filenames:
            img=gen_a_sam(i,filename)
            i=i+1



