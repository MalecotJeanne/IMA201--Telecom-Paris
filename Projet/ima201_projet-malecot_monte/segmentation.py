"""import des librairies utiles"""
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage import color, morphology as morpho, filters, exposure

"""calcul du masque valant True sur la vignette de l'image et False autre part"""
def bords_noirs(img):
    (l,L,p) = img.shape

    if (np.mean(img[2,2,0])>20):
        return np.full((l,L), False)
    else :
        mask1 = morpho.flood(img[:,:,0], (1,1), connectivity=2, tolerance=45)
        mask2 = morpho.flood(img[:,:,0], (l-2,1), connectivity=2, tolerance=45)
        mask3 = morpho.flood(img[:,:,0], (l-2,L-2), connectivity=2, tolerance=45)
        mask4 = morpho.flood(img[:,:,0], (1,L-2), connectivity=2, tolerance=45)
        if (np.all(mask1)):
            mask1 = np.full(mask1.shape, False)
        if (np.all(mask2)):
            mask2 = np.full(mask2.shape, False)
        if (np.all(mask3)):
            mask3 = np.full(mask3.shape, False)
        if (np.all(mask4)):
            mask4 = np.full(mask4.shape, False)
        out = mask1 + mask2 + mask3 + mask4
        if (np.sum(out) > out.size/2) :
            out = np.full(out.shape, False)
        return out

"""Sélection d'un rectangle contenant des pixels de peau saine"""
def peau_saine(img):
    (l,L,p) = img.shape

    mask = bords_noirs(img)

    img_lab = color.rgb2lab(img)

    nu = 1/4
    s = math.floor(0.02*L)
    min_var_coeff = 0
    peauSaine = []

    n = math.floor((nu*l)/(0.02*L))

    for k in range(n):
        rec_gauche = []
        rec_droite = []
        rec_haut = []
        rec_bas = []

        for i in range((k+1)*s,l-(k+1)*s):
            for j in range(k*s,(k+1)*s):
                if (mask[i,j] == 0):
                    rec_gauche.append(img_lab[i,j])

            for j in range(L-(k+1)*s,L-k*s):
                if (mask[i,j] == 0):
                    rec_droite.append(img_lab[i,j])

        for j in range(k*s,L-(k*s)):
            for i in range(k*s,(k+1)*s):
                if (mask[i,j] == 0):
                    rec_haut.append(img_lab[i,j])

            for i in range(l-(k+1)*s,l-k*s):
                if (mask[i,j] == 0):
                    rec_bas.append(img_lab[i,j])

        rec_gauche = np.array(rec_gauche)
        rec_droite = np.array(rec_droite)
        rec_haut = np.array(rec_haut)
        rec_bas = np.array(rec_bas)

        rec = np.concatenate((rec_haut.reshape((rec_haut.size//3,3)),rec_bas.reshape((rec_bas.size//3,3)),rec_gauche.reshape((rec_gauche.size//3,3)),rec_droite.reshape((rec_droite.size//3,3))))
        moy = [np.mean(rec[:,k]) for k in range(3)]
        sigma = [np.std(rec[:,k]) for k in range(3)]
        var_coeff = abs(sum(sigma[i]/moy[i] for i in range(3)))
        if(k == 0 or var_coeff<min_var_coeff):
            min_var_coeff = var_coeff
            peauSaine = rec

    return peauSaine

"""remplacement de la vignette par la valeur médiane du rectangle de peau saine"""
def remplacement_vignette(img,mask,peauSaine):
    img_n = np.copy(img)
    img_n[mask] = 255*color.lab2rgb([[np.median(peauSaine,axis=0)]]).reshape(3)
    return img_n

"""calcul de la distance euclidienne entre l'image de base et la médiane du rectangle de peau saine"""
def image_intensite(img,img_n):
    peauSaine = peau_saine(img)   
    img_lab = color.rgb2lab(img_n)
    (l,L,p) = img.shape

    median = np.median(peauSaine, axis = 0)
    im_int = np.zeros((l,L))

    for x in range(l):
        for y in range(L):
            im_int[x,y] = np.sqrt((img_lab[x,y,0] - median[0])**2 + (img_lab[x,y,1] - median[1])**2 + (img_lab[x,y,2] - median[2])**2)

    im_int = im_int.astype(np.uint8)

    # application d'un filtre médian

    ws = math.floor(0.01*L)

    im_int = filters.median(im_int,np.ones((ws,ws)))

    return im_int

"""calcul du vecteur de variance interclasse"""
def variance_vecteur(img):

    counts, bin_centers = exposure.histogram(img)

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(counts)
    weight2 = np.cumsum(counts[::-1])[::-1]
    # class means for all possible thresholds
    mean1 = np.cumsum(counts * bin_centers) / weight1
    mean2 = (np.cumsum((counts * bin_centers)[::-1]) / weight2[::-1])[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

    return variance12

def th(img):
    (l,L,p) = img.shape
    s = math.floor(0.02*L)

    #calcul de sigma et phi pour la région 1 (l'image en entier)

    sigma1 = variance_vecteur(img)
    phi1 = 2*np.linalg.norm(sigma1)

    #calcul de sigma et phi pour la région 2 (les diagonales et la croix centrale)

    ligne_horiz = img[l//2-s//2:l//2+s//2,:]
    ligne_vert = img[:,L//2-s//2:L//2+s//2]
    ligne_diag1 = img.diagonal(0)
    ligne_diag2 = np.fliplr(img).diagonal(0)
    
    for k in range(1,s):
        ligne_diag1 = np.concatenate((ligne_diag1,img.diagonal(k)))
        ligne_diag2 = np.concatenate((ligne_diag2,np.fliplr(img).diagonal(k)))

    ligne_horiz = ligne_horiz.reshape((ligne_horiz.size//3,3))
    ligne_vert = ligne_vert.reshape((ligne_vert.size//3,3))
    ligne_diag1 = ligne_diag1.reshape((ligne_diag1.size//3,3))
    ligne_diag2 = ligne_diag2.reshape((ligne_diag2.size//3,3))

    zone2 = np.concatenate((ligne_diag1,ligne_diag2,ligne_horiz,ligne_vert))

    sigma2 = variance_vecteur(zone2)
    phi2 = 2*np.linalg.norm(sigma2)

    if (len(sigma1) != len(sigma2)):
        diff = len(sigma1) - len(sigma2)
        if (diff < 0) :
            sigma1 = np.concatenate((sigma1, np.zeros(diff)))
        else:
            sigma2 = np.concatenate((sigma2, np.zeros(diff)))

    return np.argmax(sigma1**2/phi1 + sigma2**2/phi2)

def ts(img,beta):
    
    peauSaine = peau_saine(img)
    median = np.median(peauSaine, axis = 0)

    (L,p) = peauSaine.shape

    for x in range(L):
        peauSaine[x] = np.sqrt((peauSaine[x,0] - median[0])**2 + (peauSaine[x,1] - median[1])**2 + (peauSaine[x,2] - median[2])**2)

    gamma5 = np.percentile(peauSaine,5)
    gamma50 = np.percentile(peauSaine,50)

    return gamma50 + beta*(gamma50-gamma5)

def t(img,alpha,beta):
    th_ = th(img)
    ts_ = ts(img,beta)

    if (ts_ > th_):
        alpha = 1

    return alpha*th_ + (1-alpha)*ts_

"""calcul de l'image segmentée par rapport à t"""
def im_thresh(img,im_int,alpha,beta):
    (l,L) = im_int.shape
    t_ = t(img,alpha,beta)

    im_thresh = np.where(im_int<t_,0,255)

    im_thresh = np.where(bords_noirs(img), 0, im_thresh)

    im_thresh = morpho.opening(morpho.closing(im_thresh,morpho.disk(0.01*L)),morpho.disk(0.01*L))

    return im_thresh

"""remplissage de la région dont fait partie le pixel (u,v) par la valeur label"""
def remplissage_region(I, u, v, label):
    (l,L) = I.shape

    S = []
    S.append((u,v))
    while (S != []):
        (x,y) = S.pop()
        if (I[x][y] == 255):
            I[x][y] = label
            if(x<l-1):
                S.append((x+1,y))
            if(y<L-1):
                S.append((x,y+1))
            if(y>0):
                S.append((x,y-1))
            if(x>0):
                S.append((x-1,y))
    return None

"""séparation des zones blanches de l'image en différente régions"""
def marquage_regions(I):

    (l,L) = I.shape
    m=2
    for u in range(l):
        for v in range(L):
            if (I[u][v] == 255):
                remplissage_region(I,u,v,m)
                m += 1
    return I

"""choix de la région la meilleure d'après le critère du papier"""
def best_region(im_labeled, im_int):

  (l,L) = im_labeled.shape
  (x0,y0) = (l//2,L//2)
  sigx = sigy = 0.1*L
  N = [0]*(im_labeled.max()+1)
  for x in range(l):
    for y in range(L):
        i = im_labeled[x,y]
        N[i] += np.exp(-((x-x0)**2/(2*sigx**2)+(y-y0)**2/(2*sigy**2)))*im_int[x,y]

  index_best_region = N.index(max(N))

  im_final = np.where(im_labeled==index_best_region, 255,0)
  im_final = im_final.astype(np.uint8)
  return im_final

"""application du processus de post-processing en entier"""
def post_processing(im_thresh,im_int):
    (l,L) = im_thresh.shape

    im_thresh = morpho.opening(morpho.closing(im_thresh,morpho.disk(0.01*L)),morpho.disk(0.01*L))

    im_region = marquage_regions(np.copy(im_thresh))
    im_region = im_region.astype(np.uint8)

    im_final = best_region(im_region, im_int) 

    im_final = filters.gaussian(im_final)

    return im_final

"""application du processus de segmentation en entier"""
def segmentation_finale(img,alpha,beta):
    peauSaine = peau_saine(img)
    mask = bords_noirs(img)
    img_n = remplacement_vignette(img,mask,peauSaine)
    img_int = image_intensite(img,img_n)
    img_th = im_thresh(img,img_int,alpha,beta)
    img_fin = post_processing(img_th,img_int)
    return img_fin


