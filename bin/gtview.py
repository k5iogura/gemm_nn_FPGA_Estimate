#!/usr/bin/env python
#! coding:utf-8
import os
import sys
import fnmatch
import argparse
import cv2
import re
import numpy as np
from pdb import *

color_table=[(255,0,0),(0,255,0),(0,0,255),(255,128,0),(128,256,0),(0,128,256)]
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='get image data by keyword search from flickr')
    parser.add_argument('dir_or_jpg_or_imagelist', nargs='+', type=str)
    parser.add_argument('--shuffle',           '-s'  , action='store_true')
    args = parser.parse_args()

    # retrieving files
    filenames=[]
    for arg in args.dir_or_jpg_or_imagelist:
        if os.path.isdir(arg):
            for root,dirs,names in os.walk(args.dir_or_jpg_or_imagelist[0]):
                finded_filenames=[os.path.join(root,f) for f in names
                    if fnmatch.fnmatch(os.path.join(root,f),'*.jpg')]
                filenames.extend(finded_filenames)
        elif os.path.isfile(arg):
            if len(re.findall('\.jpg',arg)) >0:
                filenames.append(arg)
            else:
                with open(arg) as f:
                    files = [str(i).strip() for i in f.readlines()]
                filenames.extend(files)
        else:
            print('unknown {}'.format(arg))
            sys.exit(1)

    l=-1
    # shuffle retrieved files
    if args.shuffle:
        filenames = np.asarray(filenames)
        filenames = filenames[np.random.permutation(len(filenames))]

    with_gt=with_no_gt=0
    for j in range(0,len(filenames)):
        l+=1
        i = filenames[l]
        txt = re.sub('\.jpg','.txt',i)
        if not os.path.exists(txt):
        #    print('txt not found %s'%txt)
            with_no_gt+=1
            continue
        with_gt+=1
        print(i)
        print(txt)
        img = cv2.imread(i)
        H,W = img.shape[:2]
        with open(txt) as f:
            e = [gt.strip().split() for gt in f.readlines()]
        for gt in e:
            c,rx,ry,rw,rh   = gt[:5]
            fx,fy,fw,fh = float(rx),float(ry),float(rw),float(rh)
            c,x,y,w,h   = int(c),W*fx,H*fy,W*fw,H*fh
            print('GT-ratio: {} bottom-left/top-right = {} {}/{} {} {}%'.format(c,fx,fy,fw,fh,100.*fw*fh))
            print('GT-pixel: {} bottom-left/top-right = {} {}/{} {} {}%'.format(c,int(x-w/2),int(y-h/2),int(x+w/2),int(y+h/2),100.*fw*fh))
            col = color_table[c%len(color_table)] 
            cv2.rectangle(img,(int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),col)
        cv2.imshow('GT-preview',img)
        while True:
            k=cv2.waitKey(30)
            if   k==27 or k==1048603:
                print('with_gt/with_no_gt=%d/%d'%(with_gt,with_no_gt))
                sys.exit(1)
            elif k==32 or k==1048608:break
            elif k==65361 or k==1113937:      # <- hidari yajirushi
                if l>0:l-=2
                break
            elif k==65363:break # -> migi yajirushi
            elif k!=-1:print('unknown keycode %d'%k)

    print('with_gt/with_no_gt=%d/%d'%(with_gt,with_no_gt))
