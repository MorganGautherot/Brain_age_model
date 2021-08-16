from random import gauss
import random
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from transformations import rotation_matrix

def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) **0.5
    return [x/mag for x in vec]

def coordinateTransformWrapperReg(X_T1, maxDeg=20, maxShift=5):
    if(random.random() >= 0.5):
        return X_T1
    else:
        angle = maxDeg*2*(random.random()-0.5)
        randomAngle = np.radians(angle)
        unitVec = tuple(make_rand_vector(3))
        shiftVec = [maxShift*2*(random.random()-0.5),
                maxShift*2*(random.random()-0.5),
                maxShift*2*(random.random()-0.5)]
        X_T1 = coordinateTransform(X_T1, randomAngle, unitVec, shiftVec)
    
        return X_T1

def coordinateTransform(vol, randomAngle, unitVec, shiftVec, order=1, mode='constant'):
    # from transformations import rotation_matri
    ax = (list(vol.shape))
    ax = [ ax[i] for i in [1,0,2]]
    coords = np.meshgrid(np.arange(ax[0]), np.arange(ax[1]), np.arange(ax[2]))
    
    # stack the meshgrid to position vectors, center them around 0 but subtracting dim/2
    xyz = np.vstack([coords[0].reshape(-1)-float(ax[0])/2,         # x coordinate, centered
                     coords[1].reshape(-1)-float(ax[1])/2,         # y coordinate, centered 
                     coords[2].reshape(-1)-float(ax[2]/2),         # z coordinate, centered
                     np.ones((ax[0], ax[1], ax[2])).reshape(-1)])  # 1 for homogeneous coordinates

    
    # create transformation matrix
    mat=rotation_matrix(randomAngle, unitVec)
    
    # apply transformation
    transformed_xyz=np.dot(mat, xyz)
    
    # extract coordinates, don't use transformed_xyz[3,:] that's the homogeneous coordinate, always 1
    x=transformed_xyz[0,:]+float(ax[0])/2+shiftVec[0]
    y=transformed_xyz[1,:]+float(ax[1])/2+shiftVec[1]
    z=transformed_xyz[2,:]+float(ax[2])/2+shiftVec[2]
    
    x=x.reshape((ax[1], ax[0], ax[2]))
    y=y.reshape((ax[1], ax[0], ax[2]))
    z=z.reshape((ax[1], ax[0], ax[2]))
    
    new_xyz=[y, x, z]
    new_vol=map_coordinates(vol, new_xyz, order=order, mode=mode)
    return new_vol

def matrice_translation(path, pixel_max=10, trans_rate=0.5):

    translation = random.randint(1, 100)

    if translation >= trans_rate*100 :

        which_translation = random.randint(1,14)
        pixel_translation = random.randint(1, pixel_max)
        
        wm_old_matrice = nib.load(path['wm']).get_data()
        wm_matrice = np.zeros(SIZE[0:3])
        
        gm_old_matrice = nib.load(path['gm']).get_data()
        gm_matrice = np.zeros(SIZE[0:3])

        csf_old_matrice = nib.load(path['csf']).get_data()
        csf_matrice = np.zeros(SIZE[0:3])

        raw_old_matrice = nib.load(path['raw_proc']).get_data()
        raw_matrice = np.zeros(SIZE[0:3])

        if which_translation == 1:
            wm_matrice [pixel_translation:, :, :] = wm_old_matrice[pixel_translation:, :, :] 
            gm_matrice [pixel_translation:, :, :] = gm_old_matrice[pixel_translation:, :, :] 
            csf_matrice [pixel_translation:, :, :] = csf_old_matrice[pixel_translation:, :, :] 
            raw_matrice [pixel_translation:, :, :] = raw_old_matrice[pixel_translation:, :, :] 
        if which_translation == 2:
            wm_matrice [pixel_translation:, pixel_translation:, :] = wm_old_matrice[pixel_translation:, pixel_translation:, :]
            gm_matrice [pixel_translation:, pixel_translation:, :] = gm_old_matrice[pixel_translation:, pixel_translation:, :]
            csf_matrice [pixel_translation:, pixel_translation:, :] = csf_old_matrice[pixel_translation:, pixel_translation:, :]
            raw_matrice [pixel_translation:, pixel_translation:, :] = raw_old_matrice[pixel_translation:, pixel_translation:, :]
        if which_translation == 3:
            wm_matrice [pixel_translation:, :, pixel_translation:] = wm_old_matrice[pixel_translation:, :, pixel_translation:]
            gm_matrice [pixel_translation:, :, pixel_translation:] = gm_old_matrice[pixel_translation:, :, pixel_translation:]
            csf_matrice [pixel_translation:, :, pixel_translation:] = csf_old_matrice[pixel_translation:, :, pixel_translation:]
            raw_matrice [pixel_translation:, :, pixel_translation:] = raw_old_matrice[pixel_translation:, :, pixel_translation:]
        if which_translation == 4:
            wm_matrice [pixel_translation:, pixel_translation:, pixel_translation:] = wm_old_matrice[pixel_translation:, pixel_translation:, pixel_translation:]
            gm_matrice [pixel_translation:, pixel_translation:, pixel_translation:] = gm_old_matrice[pixel_translation:, pixel_translation:, pixel_translation:]
            csf_matrice [pixel_translation:, pixel_translation:, pixel_translation:] = csf_old_matrice[pixel_translation:, pixel_translation:, pixel_translation:]
            raw_matrice [pixel_translation:, pixel_translation:, pixel_translation:] = raw_old_matrice[pixel_translation:, pixel_translation:, pixel_translation:]
        if which_translation == 5:
            wm_matrice [:, pixel_translation:, :] = wm_old_matrice[:, pixel_translation:, :]
            gm_matrice [:, pixel_translation:, :] = gm_old_matrice[:, pixel_translation:, :]
            csf_matrice [:, pixel_translation:, :] = csf_old_matrice[:, pixel_translation:, :]
            raw_matrice [:, pixel_translation:, :] = raw_old_matrice[:, pixel_translation:, :]
        if which_translation == 6:
            wm_matrice [:, pixel_translation:, pixel_translation:] = wm_old_matrice[:, pixel_translation:, pixel_translation:]
            gm_matrice [:, pixel_translation:, pixel_translation:] = gm_old_matrice[:, pixel_translation:, pixel_translation:]
            csf_matrice [:, pixel_translation:, pixel_translation:] = csf_old_matrice[:, pixel_translation:, pixel_translation:]
            raw_matrice [:, pixel_translation:, pixel_translation:] = raw_old_matrice[:, pixel_translation:, pixel_translation:]
        if which_translation == 7:
            wm_matrice [:, :, pixel_translation:] = wm_old_matrice[:, :, pixel_translation:] 
            gm_matrice [:, :, pixel_translation:] = gm_old_matrice[:, :, pixel_translation:] 
            csf_matrice [:, :, pixel_translation:] = csf_old_matrice[:, :, pixel_translation:] 
            raw_matrice [:, :, pixel_translation:] = raw_old_matrice[:, :, pixel_translation:] 
        if which_translation == 8:
            wm_matrice [:-pixel_translation, :, :] = wm_old_matrice[:-pixel_translation, :, :]
            gm_matrice [:-pixel_translation, :, :] = gm_old_matrice[:-pixel_translation, :, :]
            csf_matrice [:-pixel_translation, :, :] = csf_old_matrice[:-pixel_translation, :, :]
            raw_matrice [:-pixel_translation, :, :] = raw_old_matrice[:-pixel_translation, :, :]
        if which_translation == 9:
            wm_matrice [:-pixel_translation, :-pixel_translation, :] = wm_old_matrice[:-pixel_translation, :-pixel_translation, :]
            gm_matrice [:-pixel_translation, :-pixel_translation, :] = gm_old_matrice[:-pixel_translation, :-pixel_translation, :]
            csf_matrice [:-pixel_translation, :-pixel_translation, :] = csf_old_matrice[:-pixel_translation, :-pixel_translation, :]
            raw_matrice [:-pixel_translation, :-pixel_translation, :] = raw_old_matrice[:-pixel_translation, :-pixel_translation, :]
        if which_translation == 10:
            wm_matrice [:-pixel_translation, :, :-pixel_translation] = wm_old_matrice[:-pixel_translation, :, :-pixel_translation]
            gm_matrice [:-pixel_translation, :, :-pixel_translation] = gm_old_matrice[:-pixel_translation, :, :-pixel_translation]
            csf_matrice [:-pixel_translation, :, :-pixel_translation] = csf_matrice[:-pixel_translation, :, :-pixel_translation]
            raw_matrice [:-pixel_translation, :, :-pixel_translation] = raw_old_matrice[:-pixel_translation, :, :-pixel_translation]
        if which_translation == 11:
            wm_matrice [:-pixel_translation, :-pixel_translation, :-pixel_translation] = wm_old_matrice[:-pixel_translation, :-pixel_translation, :-pixel_translation]
            gm_matrice [:-pixel_translation, :-pixel_translation, :-pixel_translation] = gm_old_matrice[:-pixel_translation, :-pixel_translation, :-pixel_translation]
            csf_matrice [:-pixel_translation, :-pixel_translation, :-pixel_translation] = csf_old_matrice[:-pixel_translation, :-pixel_translation, :-pixel_translation]
            raw_matrice [:-pixel_translation, :-pixel_translation, :-pixel_translation] = raw_old_matrice[:-pixel_translation, :-pixel_translation, :-pixel_translation]
        if which_translation == 12:
            wm_matrice [:, :-pixel_translation, :] = wm_old_matrice[:, :-pixel_translation, :]
            gm_matrice [:, :-pixel_translation, :] = gm_old_matrice[:, :-pixel_translation, :]
            csf_matrice [:, :-pixel_translation, :] = csf_old_matrice[:, :-pixel_translation, :]
            raw_matrice [:, :-pixel_translation, :] = raw_old_matrice[:, :-pixel_translation, :]
        if which_translation == 13:
            wm_matrice [:, :-pixel_translation, :-pixel_translation] = wm_old_matrice[:, :-pixel_translation, :-pixel_translation]
            gm_matrice [:, :-pixel_translation, :-pixel_translation] = gm_old_matrice[:, :-pixel_translation, :-pixel_translation]
            csf_matrice [:, :-pixel_translation, :-pixel_translation] = csf_old_matrice[:, :-pixel_translation, :-pixel_translation]
            raw_matrice [:, :-pixel_translation, :-pixel_translation] = raw_old_matrice[:, :-pixel_translation, :-pixel_translation]
        if which_translation == 14:
            wm_matrice [:, :, :-pixel_translation] = wm_old_matrice[:, :, :-pixel_translation]
            gm_matrice [:, :, :-pixel_translation] = gm_old_matrice[:, :, :-pixel_translation]
            csf_matrice [:, :, :-pixel_translation] = csf_old_matrice[:, :, :-pixel_translation]
            raw_matrice [:, :, :-pixel_translation] = raw_old_matrice[:, :, :-pixel_translation]  
    else :
        wm_matrice = nib.load(path['wm']).get_data()
        gm_matrice = nib.load(path['gm']).get_data()
        csf_matrice = nib.load(path['csf']).get_data()
        raw_matrice = nib.load(path['raw_proc']).get_data()

    return (wm_matrice, gm_matrice, csf_matrice, raw_matrice)
