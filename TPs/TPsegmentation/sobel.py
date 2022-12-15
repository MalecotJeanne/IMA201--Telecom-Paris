#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 22 May 2019

@author: M Roux
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import skimage as sk

from scipy import ndimage
from scipy import signal

from skimage import io

from skimage import filters
import skimage.morphology as morpho  


##############################################

import mrlab as mr
import hysteresis as hys

##############################################"

############## le close('all') de Matlab
plt.close('all')
################################"



def tophat(im,rayon):
    se=morpho.square(rayon)
    ero=morpho.erosion(im,se)
    dil=morpho.dilation(ero,se)
    tophat=im-dil
    return tophat

ima=io.imread('images/cell.tif')


sigma=0
seuilnorme=0.6


gfima=filters.gaussian(ima,sigma)

plt.figure('Image originale')
plt.imshow(ima, cmap='gray')

plt.figure('Image filtrée (passe-bas)')
plt.imshow(gfima, cmap='gray')

gradx=mr.sobelGradX(gfima)
grady=mr.sobelGradY(gfima)  
      
plt.figure('Gradient horizontal')
plt.imshow(gradx, cmap='gray')

plt.figure('Gradient vertical')
plt.imshow(grady, cmap='gray')

norme=np.sqrt(gradx*gradx+grady*grady)

    
plt.figure('Norme du gradient')
plt.imshow(norme, cmap='gray')

direction=np.arctan2(grady,gradx)
    
plt.figure('Direction du Gradient')
plt.imshow(direction, cmap='gray')


contoursnorme =(norme>seuilnorme) 


plt.figure('Norme seuillée')
plt.imshow(255*contoursnorme)


contours=np.uint8(mr.maximaDirectionGradient(gradx,grady))

plt.figure('Maxima du gradient dans la direction du gradient')
plt.imshow(255*contours)


valcontours=(norme>seuilnorme)*contours
      
plt.figure()
plt.imshow(255*valcontours)
plt.show()

rayon=3
top=tophat(valcontours,rayon)

low = 1
high = 6

lowt = (top > low).astype(int)
hight = (top > high).astype(int)
hyst = filters.apply_hysteresis_threshold(top, low, high)

plt.imshow(hyst)