#!/usr/bin/python
#彩色文件灰化后文件名有变化，这里将标注文件名和图像文件名，以及标注文件内的图像文件名一致化
import sys
import os
import os.path
import xml.etree.ElementTree as et

def walkdir(xmldir,picdir):
    if not os.path.exists(xmldir):
        print "{} doesn't exist.".format(xmldir)
        return

    pe=picdir.split("/")
    picpname=pe[len(pe)-1]  #directly parent name
    ret=[]
    for parent,dirnames,filenames in os.walk(xmldir):
        for filename in filenames:
            tree=et.parse(os.path.join(parent,filename))
            root=tree.getroot()
            folderelement=tree.find("folder")
            folderelement.text=picpname
	        #folderelement.text=picpname
            filenameelement=tree.find("filename")
            newpathname="{}.jpg".format(str(os.path.join(picdir,filenameelement.text)))
            pathelement=tree.find("path")
            pathelement.text=newpathname #update path
            tree.write(os.path.join(parent,filename))
            print folderelement.text
            print pathelement.text


"""
   usage  xmlmani.py  $datadir  $imgdir
"""

if __name__ == "__main__":
    walkdir(sys.argv[1],sys.argv[2])


