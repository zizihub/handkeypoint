# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:43:29 2017
@author: zhaoy
"""
import numpy as np
import cv2

# reference facial points, a list of coordinates (x,y)
REFERENCE_FACIAL_POINTS = [
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 92.3655014],
    [62.72990036, 92.20410156]
]

DEFAULT_CROP_SIZE = (96, 112)


class FaceWarpException(Exception):
    def __str__(self):
        return 'In File {}:{}'.format(
            __file__, super.__str__(self))


def get_affine_transform_matrix(src_pts, dst_pts):
    """
    Function:
    ----------
        get affine transform matrix 'tfm' from src_pts to dst_pts
    Parameters:
    ----------
        @src_pts: Kx2 np.array
            source points matrix, each row is a pair of coordinates (x, y)
        @dst_pts: Kx2 np.array
            destination points matrix, each row is a pair of coordinates (x, y)
    Returns:
    ----------
        @tfm: 2x3 np.array
            transform matrix from src_pts to dst_pts
    """

    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

#    #print(('src_pts_:\n' + str(src_pts_))
#    #print(('dst_pts_:\n' + str(dst_pts_))

    A, res, rank, s = np.linalg.lstsq(src_pts_, dst_pts_)

#    #print(('np.linalg.lstsq return A: \n' + str(A))
#    #print(('np.linalg.lstsq return res: \n' + str(res))
#    #print(('np.linalg.lstsq return rank: \n' + str(rank))
#    #print(('np.linalg.lstsq return s: \n' + str(s))

    if rank == 3:
        tfm = np.float32([
            [A[0, 0], A[1, 0], A[2, 0]],
            [A[0, 1], A[1, 1], A[2, 1]]
        ])
    elif rank == 2:
        tfm = np.float32([
            [A[0, 0], A[1, 0], 0],
            [A[0, 1], A[1, 1], 0]
        ])

    return tfm


def warp_and_crop_face(src_img,
                       facial_pts,
                       reference_pts=None,
                       crop_size=(96, 112)):
    """
    Function:
    ----------
        apply affine transform 'trans' to uv
    Parameters:
    ----------
        @src_img: 3x3 np.array
            input image
        @facial_pts: could be
            1)a list of K coordinates (x,y)
        or
            2) Kx2 or 2xK np.array
            each row or col is a pair of coordinates (x, y)
        @reference_pts: could be
            1) a list of K coordinates (x,y)
        or
            2) Kx2 or 2xK np.array
            each row or col is a pair of coordinates (x, y)
        or
            3) None
            if None, use default reference facial points
        @crop_size: (w, h)
            output face image size
        @align_type['affine']: use all points to do affine transform
    Returns:
    ----------
        @face_img: output face image with size (w, h) = @crop_size
    """

    if reference_pts is None:
        reference_pts = REFERENCE_FACIAL_POINTS

    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape
    if max(ref_pts_shp) < 3 or min(ref_pts_shp) != 2:
        raise FaceWarpException(
            'reference_pts.shape must be (K,2) or (2,K) and K>2')

    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape
    if max(src_pts_shp) < 3 or min(src_pts_shp) != 2:
        raise FaceWarpException(
            'facial_pts.shape must be (K,2) or (2,K) and K>2')

    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

#    #print('--->src_pts:\n', src_pts
#    #print('--->ref_pts\n', ref_pts

    if src_pts.shape != ref_pts.shape:
        raise FaceWarpException(
            'facial_pts and reference_pts must have the same shape')

    tfm = get_affine_transform_matrix(src_pts, ref_pts)

#    #print('--->Transform matrix: '
#    #print(('type(tfm):' + str(type(tfm)))
#    #print(('tfm.dtype:' + str(tfm.dtype))
#    #print( tfm

    face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))

    return face_img
