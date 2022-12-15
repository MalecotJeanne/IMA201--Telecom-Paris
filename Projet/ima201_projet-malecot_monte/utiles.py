"""import de librairies utiles"""

import matplotlib.pyplot as plt
from skimage import morphology
import numpy as np


def view_can(ima):
    """Affiche  les 3 canaux d'une image"""

    fig = plt.figure(figsize=(10, 8))

    fig.add_subplot(221).set_title('Image RVB')
    plt.imshow(ima)

    fig.add_subplot(222).set_title('Canal Rouge')
    plt.imshow(ima[:, :, 0], cmap='gray')

    fig.add_subplot(223).set_title('Canal Vert')
    plt.imshow(ima[:, :, 1], cmap='gray')

    fig.add_subplot(224).set_title('Canal Bleu')
    plt.imshow(ima[:, :, 2], cmap='gray')

    plt.show()


def dice_score(masque_th, masque_exp):
    """renvoie le dice score de deux masques"""
    inter = (masque_th & masque_exp) ^ ((1-masque_th) & (1-masque_exp))
    return np.sum(inter)/(masque_th.size)


def contours(ima, masque):
    """renvoie l'image avec le contour rouge associé au masque donné"""
    contour = masque ^ morphology.dilation(masque, morphology.disk(4))
    contour = morphology.closing(contour, morphology.disk(2))
    ima_contour = ima.copy()

    (m, n) = ima.shape[0:2]
    for i in range(m):
        for j in range(n):
            if contour[i, j] == 1:
                ima_contour[i][j] = [255, 0, 00]

    return ima_contour
