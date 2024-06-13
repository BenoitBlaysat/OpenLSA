#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 13:49:59 2022

This is a simple python script written by the Clermont's EM team to
retrieve displacement maps for a pair of images, pattern beeing periodic.

These python codes can be used for non-profit academic research only. They are
distributed under the terms of the GNU general public license v3.

Anyone finding the python codes useful is kindly asked to cite:

# [1] M. Grédiac, B. Blaysat, and F. Sur. Extracting displacement and strain fields from
checkerboard images with the localized spectrum analysis. Experimental Mechanics, 59(2):207–218,
2019.
# [2] B. Blaysat, F. Sur, T. Jailin, A. Vinel and M. Grédiac. Open LSA: an Open-source toolbox for
computing full-field displacements from images of periodic patterns. Submitted to SoftwareX, 2024

@author: UCA/IP - M3G - EM team
"""
import numpy as np
import cv2
from scipy import ndimage
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import re
import os
NoneType = type(None)


###############################################################################
# %% Usefull functions
def scal_prod(input_1, input_2):
    """ Scalar product"""
    assert isinstance(input_1, (np.ndarray, int, float, np.generic, np.complexfloating))
    if isinstance(input_1, np.ndarray):
        assert input_1.dtype in (int, float, np.generic, np.complexfloating)
    assert isinstance(input_2, (np.ndarray, int, float, np.generic, np.complexfloating))
    if isinstance(input_2, np.ndarray):
        assert input_2.dtype in (int, float, np.generic, np.complexfloating)

    return input_1.real*input_2.real + input_1.imag*input_2.imag


def a01_2_axy(vec01, a01):
    """ Compute given vector a01 (expressed in basis vec01) in basis (ex, ey)"""
    assert isinstance(vec01, (list, np.ndarray))
    assert len(vec01) == 2
    assert_array(a01)
    assert len(a01.shape) > 1 and a01.shape[1] == 2

    return a01[:, 0]*vec01[0] + a01[:, 1]*vec01[1]


def axy_2_a01(vec01, axy):
    """ Compute given vector axy (ex, ey) in basis vec01"""
    assert isinstance(vec01, (list, np.ndarray))
    assert len(vec01) == 2
    assert_array(axy)

    op_00, op_01, op_10, op_11 = vec01[0].real, vec01[1].real, vec01[0].imag, vec01[1].imag
    det_op = op_00*op_11-op_01*op_10
    iop_00, iop_01, iop_10, iop_11 = op_11/det_op, -op_01/det_op, -op_10/det_op, op_00/det_op
    return np.array([axy.real*iop_00 + axy.imag*iop_01,
                     axy.real*iop_10 + axy.imag*iop_11])


def compute_rbm(disp, coord_x, coord_y):
    """ Computing the RBM part of a displacement"""
    assert_array([disp, coord_x, coord_y])
    assert coord_x.shape == disp.shape
    assert coord_y.shape == disp.shape

    coord_x = (coord_x - coord_x.mean())/(2*(coord_x.max()-coord_x.min()))
    coord_y = (coord_y - coord_y.mean())/(2*(coord_y.max()-coord_y.min()))
    operator = np.array([[len(coord_x), 0, 0],
                         [0, len(coord_y), 0],
                         [0, 0, np.sum(coord_x**2+coord_y**2)]])
    right_hand_member = np.array([np.sum(disp.real),
                                  np.sum(disp.imag),
                                  np.sum(coord_x*disp.imag + coord_y*disp.real)])
    dof = np.linalg.lstsq(operator, right_hand_member, rcond=None)[0]
    return dof[0] + 1j*dof[1] + dof[2]*(coord_y + 1j*coord_x)


def reject_outliers(data, bandwitch=3):
    """ Removing outliers"""
    assert_array(data)
    assert isinstance(bandwitch, (int, float, np.generic))
    assert bandwitch > 0

    return abs(data - np.nanmean(data)) < bandwitch * np.nanstd(data)

###############################################################################


# %% OpenCV functions for raw estimating of the displacement field
def estimate_u(img1, img2, filter_size=None, dis_init=None):
    """ This function estimates the displacement that warps image img1 into image img2 using the
    Dense Inverse Search optical flow algorithm from the OpenCV Python library """
    assert_array([img1, img2])
    assert img2.shape == img1.shape
    # optical flow will be needed, so it is initialized
    dis = cv2.DISOpticalFlow_create()
    img1_uint8 = make_it_uint8(img1)
    img2_uint8 = make_it_uint8(img2)
    if filter_size is not None:
        img1_uint8 = ndimage.gaussian_filter(img1_uint8, filter_size)
        img2_uint8 = ndimage.gaussian_filter(img2_uint8, filter_size)
    if dis_init is not None:
        dis_init_mat = np.zeros([img1_uint8.shape[0], img1_uint8.shape[1], 2], dtype='float32')
        dis_init_mat[:, :, 0], dis_init_mat[:, :, 1] = dis_init.real, dis_init.imag
        flow = dis.calc(img1_uint8, img2_uint8,  warp_flow(dis_init_mat, dis_init_mat))
    else:
        flow = dis.calc(img1_uint8, img2_uint8, None)
    return flow[:, :, 0] + 1j*flow[:, :, 1]


def warp_flow(img, flow):
    """ This function correctly warps a displacement to correctly feed the Dense Inverse Search
    optical flow algorithm"""
    flow = -flow
    flow[:, :, 0] += np.arange(flow.shape[1])
    flow[:, :, 1] += np.arange(flow.shape[0])[:, np.newaxis]
    return cv2.remap(img, flow, None, cv2.INTER_LINEAR)


def make_it_uint8(img):
    """ This function format input image img into 8 bits depth"""
    assert_array(img)

    return np.uint8(img*2**(8-round(np.log2(img.max()))))


# %% function to open a s3 located list of images.
def provide_s3_path(s3_dictionary, im_extensions, im_pattern, verbose):
    """ This function reads the s3_dictionary to provide a list of paths to a set of images
    s3_dictionary is formated as
        . if credentials are given in a ~/.aws/config file for instance
            s3_dictionary = {'s3_endpoint_url': s3_endpoint_url
                             's3_bucket_name': s3_bucket_name,
                             's3_path_2_im': s3_path_2_im,
                             's3_path_2_folder': s3_path_2_folder}
        . if the connection is anonymous
            s3_dictionary = {'s3_access_key_id': 'anonymous',
                             's3_endpoint_url': s3_endpoint_url
                             's3_bucket_name': s3_bucket_name,
                             's3_path_2_im': s3_path_2_im,
                             's3_path_2_folder': s3_path_2_folder}
        . if the credential are given by the dictionary
            s3_dictionary = {'s3_access_key_id':ACCESS_KEY,
                             's3_secret_access_key':SECRET_KEY,
                             's3_session_token':SESSION_TOKEN,
                             's3_endpoint_url': s3_endpoint_url
                             's3_bucket_name': s3_bucket_name,
                             's3_path_2_im': s3_path_2_im,
                             's3_path_2_folder': s3_path_2_folder} """
    assert isinstance(s3_dictionary, dict)
    assert 's3_endpoint_url' in s3_dictionary
    assert 's3_bucket_name' in s3_dictionary
    assert 's3_path_2_im' in s3_dictionary or 's3_path_2_folder' in s3_dictionary

    credentials_flag = 0
    if 's3_access_key_id' in s3_dictionary:
        if s3_dictionary['s3_access_key_id'] == 'anonymous':
            credentials_flag = 1
        else:
            credentials_flag = 2

    folder_flag = 0
    if 's3_path_2_im' in s3_dictionary:
        im_stack = s3_dictionary['s3_path_2_im']
    else:
        folder_flag = 1

    if folder_flag == 1:
        if credentials_flag == 0:
            s3_client = boto3.client('s3',
                                     endpoint_url=s3_dictionary['s3_endpoint_url'])
        elif credentials_flag == 1:
            s3_client = boto3.client('s3',
                                     endpoint_url=s3_dictionary['s3_endpoint_url'],
                                     config=Config(signature_version=UNSIGNED))
        else:
            s3_client = boto3.client('s3',
                                     aws_access_key_id=s3_dictionary['s3_access_key_id'],
                                     aws_secret_access_key=s3_dictionary['s3_secret_access_key'],
                                     aws_session_token=s3_dictionary['s3_session_token'],
                                     endpoint_url=s3_dictionary['s3_endpoint_url'])

        response = s3_client.list_objects_v2(Bucket=s3_dictionary['s3_bucket_name'],
                                             Prefix=s3_dictionary['s3_path_2_folder'])
        if 'Contents' in response:
            im_stack = [item['Key'] for item in response['Contents']
                        if item['Key'].lower().endswith(tuple(im_extensions))]
            pattern = re.compile(im_pattern)
            im_stack = [item for item in im_stack
                        if pattern.match(os.path.basename(os.path.splitext(item)[0]))]
        if verbose:
            print(f"      A path to a s3 folder is given: {len(im_stack):d} images are found.")

    if credentials_flag == 0:
        s3_resource = boto3.resource('s3',
                                     endpoint_url=s3_dictionary['s3_endpoint_url'])
    elif credentials_flag == 1:
        s3_resource = boto3.resource('s3',
                                     endpoint_url=s3_dictionary['s3_endpoint_url'],
                                     config=Config(signature_version=UNSIGNED))
    else:
        s3_resource = boto3.resource('s3',
                                     aws_access_key_id=s3_dictionary['s3_access_key_id'],
                                     aws_secret_access_key=s3_dictionary['s3_secret_access_key'],
                                     aws_session_token=s3_dictionary['s3_session_token'],
                                     endpoint_url=s3_dictionary['s3_endpoint_url'])

    return im_stack, s3_resource.Bucket(s3_dictionary['s3_bucket_name'])

# %% assertion checks
def assert_point(point):
    """ check assertion for point """
    if isinstance(point, list):
        for elem_of_list in point:
            assert_point(elem_of_list)
    else:
        assert isinstance(point, (NoneType, np.ndarray))
        if isinstance(point, np.ndarray):
            assert point.shape == (2,)


def assert_array(array):
    """ check assertion for array """
    if isinstance(array, list):
        for elem_of_list in array:
            assert_array(elem_of_list)
    else:
        assert isinstance(array, np.ndarray)
        isinstance(array.item(0), (int, float, complex, np.generic, np.complexfloating))
