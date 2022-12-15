""" import des librairies utiles """
import numpy as np
import skimage
from skimage import morphology


def masque_b_3dir(ima, thresh):  # thresh = Définition du seuil de thresholding
    """Création du masque binaire"""

    # Création des matrices S0 S45 et S90
    s_0 = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])

    s_45 = np.eye(9)
    s_45[0, 0] = 0
    s_45[8, 8] = 0

    s_90 = s_0.transpose()

    o_r = ima[:, :, 0]
    o_g = ima[:, :, 1]
    o_b = ima[:, :, 2]

    g_r = np.absolute(o_r - np.maximum(np.maximum(skimage.morphology.closing(o_r, s_0),
                      skimage.morphology.closing(o_r, s_45)), skimage.morphology.closing(o_r, s_90)))
    g_g = np.absolute(o_g - np.maximum(np.maximum(skimage.morphology.closing(o_g, s_0),
                      skimage.morphology.closing(o_g, s_45)), skimage.morphology.closing(o_g, s_90)))
    g_b = np.absolute(o_b - np.maximum(np.maximum(skimage.morphology.closing(o_b, s_0),
                      skimage.morphology.closing(o_b, s_45)), skimage.morphology.closing(o_b, s_90)))

    m_r = g_r > thresh
    m_g = g_g > thresh
    m_b = g_b > thresh

    m_matrix = 1-np.multiply(np.multiply((1 - m_r), (1 - m_g)), (1 - m_b))

    return m_matrix


def masque_b_5dir(ima, thresh):  # thresh = Définition du seuil de thresholding
    """Création du masque binaire"""

    # Création des matrices S0 S45 et S90
    s_0 = np.array([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])

    s_30 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [
                    0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    s_45 = np.eye(9)
    s_45[0, 0] = 0
    s_45[8, 8] = 0

    s_60 = s_30.transpose()

    s_90 = s_0.transpose()

    o_r = ima[:, :, 0]
    o_g = ima[:, :, 1]
    o_b = ima[:, :, 2]

    def max_5(array_1, array_2, array_3, array_4, array_5):
        """renvoie le maximum de 5 tableaux numpy"""
        return np.maximum(array_1, np.maximum(array_2, np.maximum(array_3, np.maximum(array_4, array_5))))

    g_r = np.absolute(o_r - max_5(skimage.morphology.closing(o_r, s_0), skimage.morphology.closing(o_r, s_30),
                      skimage.morphology.closing(o_r, s_45), skimage.morphology.closing(o_r, s_60), skimage.morphology.closing(o_r, s_90)))
    g_g = np.absolute(o_g - max_5(skimage.morphology.closing(o_g, s_0), skimage.morphology.closing(o_g, s_30),
                      skimage.morphology.closing(o_g, s_45), skimage.morphology.closing(o_g, s_60), skimage.morphology.closing(o_g, s_90)))
    g_b = np.absolute(o_b - max_5(skimage.morphology.closing(o_b, s_0), skimage.morphology.closing(o_b, s_30),
                      skimage.morphology.closing(o_b, s_45), skimage.morphology.closing(o_b, s_60), skimage.morphology.closing(o_b, s_90)))

    m_r = g_r > thresh
    m_g = g_g > thresh
    m_b = g_b > thresh

    m_matrix = 1-np.multiply(np.multiply((1 - m_r), (1 - m_g)), (1 - m_b))

    return m_matrix


def close_open(masque, h_radius):
    """closing puis opening avec des masques de rayon h_radius"""
    masque = skimage.morphology.closing(
        masque, skimage.morphology.disk(h_radius))
    masque = skimage.morphology.opening(
        masque, skimage.morphology.disk(h_radius))

    return masque


def open_close(masque, h_radius):
    """opening puis closing avec des masques de rayon h_radius"""
    masque = skimage.morphology.opening(
        masque, skimage.morphology.disk(h_radius))
    masque = skimage.morphology.closing(
        masque, skimage.morphology.disk(h_radius))

    return masque


def bords_zone(i, j, m, n, rad):
    """détermination des bords d'une zone de rayon rad dans le cas idéal, en adaptant les cas des bords"""
    gauche, droite, haut, bas = 0, 0, 0, 0
    if i < rad:
        haut = 0
        bas = 2*rad
        if j < rad:
            # coin en haut à gauche
            gauche = 0
            droite = 2*rad
        elif j > n-rad - 1:
            # coin en haut à droite
            gauche = -2*rad
            droite = n-1
        else:
            # zone en haut hors coins
            gauche = j-rad
            droite = j+rad
    if i > m-rad - 1:
        haut = -2*rad
        bas = m-1
        if j < rad:
            # coin en bas à gauche
            gauche = 0
            droite = 2*rad
        elif j > n-rad - 1:
            # coin en bas à droite
            gauche = -2*rad
            droite = n-1
        else:
            # zone en bas hors coins
            gauche = j-rad
            droite = j+rad
    else:
        haut = i-rad
        bas = i+rad
        if j < rad:
            # zone à gauche hors coins
            gauche = 0
            droite = 2*rad
        elif j > n-rad - 1:
            # zone à droite hors coins
            gauche = -2*rad
            droite = n-1
        else:
            # centre de l'image
            gauche = j-rad
            droite = j+rad

    return (gauche, droite, haut, bas)


def hair_removal(ima, masque, rad=20):
    """retire les poils d'une image à l'aide du masque binaire ; le rayon (rad) détermine la taille de la zone observée pour chaque pixel correspondant à un poil"""
    (m, n) = ima.shape[0:2]
    masque_copy = masque.copy()
    for i in range(0, m):
        for j in range(0, n):

            if masque_copy[i, j] == 0:

                (gauche, droite, haut, bas) = bords_zone(i, j, m, n, rad)

                petit_masque = masque_copy[haut:bas, gauche:droite]
                pm_size = petit_masque.sum()

                zone_r = np.multiply(
                    ima[haut:bas, gauche:droite, 0], petit_masque)
                zone_v = np.multiply(
                    ima[haut:bas, gauche:droite, 1], petit_masque)
                zone_b = np.multiply(
                    ima[haut:bas, gauche:droite, 2], petit_masque)

                ima[i, j, 0] = zone_r.sum()/pm_size  # r mean
                ima[i, j, 1] = zone_v.sum()/pm_size  # v mean
                ima[i, j, 2] = zone_b.sum()/pm_size  # b mean

                masque_copy[i, j] = 1


def masque_cleaning(masque):
    """nettoyage d'un masque en supprimant ce qui correspond à du bruit, d'après la méthode du papier Dull Razor"""

    (m, n) = masque.shape[0:2]
    masque_copy = morphology.closing(np.copy(masque), morphology.star(2))
    masque_copy = morphology.erosion(masque_copy, morphology.disk(2))

    # definition des fonctions pour tracer les lignes dans chaque direction, à partir d'un pixel
    def ligne_n(ima, i, j):
        """longueur de la ligne vers le haut"""
        lgth = 0
        x = i
        while x >= 0 and ima[x][j] == 0:
            lgth += 1
            x -= 1
        return lgth

    def ligne_ne(ima, i, j):
        """longueur de la ligne vers le haut-droite"""
        lgth = 0
        x = i
        y = j
        while x >= 0 and y < n and ima[x][y] == 0:
            lgth += 1
            y += 1
            x -= 1
        return lgth

    def ligne_e(ima, i, j):
        """longueur de la ligne vers la droite"""
        lgth = 0
        y = j
        while y < n and ima[i][y] == 0:
            lgth += 1
            y += 1
        return lgth

    def ligne_se(ima, i, j):
        """longueur de la ligne vers le bas-droite"""
        lgth = 0
        x = i
        y = j
        while x < m and y < n and ima[x][y] == 0:
            lgth += 1
            y += 1
            x += 1
        return lgth

    def ligne_s(ima, i, j):
        """longueur de la ligne vers le bas"""
        lgth = 0
        x = i
        while x < m and ima[x][j] == 0:
            lgth += 1
            x += 1
        return lgth

    def ligne_so(ima, i, j):
        """longueur de la ligne vers le bas-gauche"""
        lgth = 0
        x = i
        y = j
        while x < m and y >= 0 and ima[x][y] == 0:
            lgth += 1
            y -= 1
            x += 1
        return lgth

    def ligne_o(ima, i, j):
        """longueur de la ligne vers la gauche"""
        lgth = 0
        y = j
        while y >= 0 and ima[i][y] == 0:
            lgth += 1
            y -= 1
        return lgth

    def ligne_no(ima, i, j):
        """longueur de la ligne vers le haut-gauche"""
        lgth = 0
        x = i
        y = j
        while x >= 0 and y >= 0 and ima[x][y] == 0:
            lgth += 1
            y -= 1
            x -= 1
        return lgth

    for i in range(m):
        for j in range(n):
            if masque[i][j] == 0:  # le pixel correspond à un poil
                directions = np.array([ligne_n(masque, i, j), ligne_ne(masque, i, j), ligne_e(masque, i, j), ligne_se(
                    masque, i, j), ligne_s(masque, i, j), ligne_so(masque, i, j), ligne_o(masque, i, j), ligne_no(masque, i, j)])
                directions_sans_max = np.copy(directions)
                if directions.max() >= 5:  # paramètre choisit empiriquement
                    dir_i = np.argmax(directions)
                    directions_sans_max[dir_i] = 0
                    for dir in directions_sans_max:
                        if dir == directions.max():
                            masque_copy[i][j] = 1
                else:
                    masque_copy[i][j] = 1

    return masque_copy


def dull_razor(ima):
    """suppression des poils d'une image, rassemblant toutes les étapes de la création du masque au traitement de l'image"""
    # copie de l'image

    ima_hr = ima.copy()

    # création du masque

    masque_b_5 = masque_b_5dir(ima_hr, 250)

    # nettoyage du masque

    masque_clean = morphology.binary_opening(masque_b_5, morphology.square(2))
    top_hat = 1-morphology.white_tophat((1-masque_clean), morphology.square(2))
    masque_clean = masque_clean-top_hat
    masque_clean = 1 - \
        morphology.binary_opening(masque_clean, morphology.square(2))

    masque_clean = masque_cleaning(
        morphology.opening(masque_b_5, morphology.disk(1)))

    # prise en compte des ombres

    masque_ombres = morphology.opening(masque_clean, morphology.star(5))
    masque_ombres = morphology.erosion(masque_ombres, morphology.disk(3))

    # hair removal

    hair_removal(ima_hr, masque_ombres, 25)

    return ima_hr


def smart_dull_razor(ima):
    """suppression des poils d'une image, rassemblant toutes les étapes de la création du masque au traitement de l'image,
    et ignorant les images sans poils"""
    # copie de l'image

    ima_hr = ima.copy()

    # création du masque

    masque_b_5 = masque_b_5dir(ima_hr, 250)

    # Détection de la présence ou non de poils.

    masque_simple = masque_cleaning(masque_b_5)

    tophat = masque_simple + \
        morphology.black_tophat(masque_simple, morphology.disk(5))
    if (1-tophat).sum() < 10000:
        return ima_hr

    # nettoyage du masque

    masque_clean = morphology.binary_opening(masque_b_5, morphology.square(2))
    top_hat = 1-morphology.white_tophat((1-masque_clean), morphology.square(2))
    masque_clean = masque_clean-top_hat
    masque_clean = 1 - \
        morphology.binary_opening(masque_clean, morphology.square(2))

    masque_clean = masque_cleaning(
        morphology.opening(masque_b_5, morphology.disk(1)))

    # prise en compte des ombres

    masque_ombres = morphology.opening(masque_clean, morphology.star(5))
    masque_ombres = morphology.erosion(masque_ombres, morphology.disk(3))

    # hair removal

    hair_removal(ima_hr, masque_ombres, 25)

    return ima_hr
