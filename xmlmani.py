#!/usr/bin/python


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
            filenameelement=tree.find("filename")
            pathelement=tree.find("path")
	        folderelement.text=picpname  #update folder
            newpathname="{}.jpg".format(str(os.path.join(picdir,filenameelement.text)))
            pathelement.text=newpathname #update path
            tree.write(os.path.join(parent,filename))
            print folderelement.text
            print pathelement.text


"""
   usage  xmlmani.py  $datadir  $imgdir
"""

if __name__ == "__main__":
    walkdir(sys.argv[1],sys.argv[2])


