"""import des librairies utiles"""
import numpy as np 
from skimage import color
from scipy.optimize import minimize

"""création du set de pixels représentant la peau saine"""
def S(hsv_im):
    (m, n, x) = hsv_im.shape

    S11 = hsv_im[0:40, 0:40]
    S21 = hsv_im[m-40:m, 0:40]
    S31 = hsv_im[m-40:m, n-40:n]
    S41 = hsv_im[0:40, n-40:n]

    S1 = [(0, 0, 0) for k in range(1600)]
    S2 = [(0, 0, 0) for k in range(1600)]
    S3 = [(0, 0, 0) for k in range(1600)]
    S4 = [(0, 0, 0) for k in range(1600)]

    k = 0

    for i in range(40):
        for j in range(40):
            S1[k] = (S11[i, j][2], i, j)
            S2[k] = (S21[i, j][2], m-40+i, j)
            S3[k] = (S31[i, j][2], m-40+i, n-40+j)
            S4[k] = (S41[i, j][2], i, n-40+j)
            k += 1

    s = np.concatenate((S1, S2, S3, S4))
    return s

"""création de la nouvelle image après le pre-processing"""
def newIm(hsv_im,s):

    """calcul de l'erreur entre le canal V de l'image et le plan courbé z(x,y)"""
    def error(P):

        """définition du plan courbé en fonction des paramètres donnés en entrée"""
        def z(x,y):
            return P[0]*x**2 + P[1]*y**2 + P[2]*x*y + P[3]*x + P[4]*y + P[5]

        e = 0

        for j in range(6400):
            (V,x,y) = tuple(s[j])
            e += (V - z(x,y))**2

        return e

    p = [0, 0, 0, 0, 0, 0]

    P = minimize(error, p, method='Powell').x

    def z(x,y):
        return P[0]*x**2 + P[1]*y**2 + P[2]*x*y + P[3]*x + P[4]*y + P[5]

    (m,n,b) = hsv_im.shape

    R = hsv_im

    for x in range(m):
        for y in range(n):
            R[x,y,2] = R[x,y,2]/z(x,y)

    return color.hsv2rgb(R)
