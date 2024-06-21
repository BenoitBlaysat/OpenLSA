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
# [2] B. Blaysat, F. Sur, T. Jailin, A. Vinel and M. Grédiac. Open LSA: Open-source toolbox for
computing full-field displacements from images of periodic patterns. Submitted to SoftwareX, 2024

@author: UCA/IP - M3G - EM team
"""

# %% Loading Libraries

# regular librairies
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import ScalarFormatter
from scipy import interpolate

# Librairies devlopped in Clermont Ferrand
from openlsa import OpenLSA
from openlsa.utils import provide_s3_path

# Loading image paths (UCA' s3 server)
bucket = 'zzz-1np0up-gnoq0c-nafe9v-pta6fh-hwf14i-openlsa-data'
s3_dict_ref = {'s3_endpoint_url': 'https://s3.mesocentre.uca.fr',
               's3_bucket_name': bucket,
               's3_access_key_id': 'anonymous',
               's3_path_2_folder': '2024-01_Bois/Peinture/Paysage/Ref'}
im_stack_ref, bucket_ref = provide_s3_path(s3_dict_ref, '.tif', r'CAM1_\d{6}', True)
s3_dict_def = {'s3_endpoint_url': 'https://s3.mesocentre.uca.fr',
               's3_bucket_name': bucket,
               's3_access_key_id': 'anonymous',
               's3_path_2_folder': '2024-01_Bois/Peinture/Paysage/Def'}
im_stack_def, bucket_def = provide_s3_path(s3_dict_def, '.tif', r'CAM1_\d{6}', True)

width = 1024
im_crop = [2200, 2200+width, 2100, 2100+width]
# %% LSA - calculation of the reference state using an image stack
my_lsa, phi_r, img_0, kernel, stats = OpenLSA.compute_refstate_from_im_stack(im_extensions='.tif',
                                                                             s3_dictionary=s3_dict_ref,
                                                                             im_pattern=r'CAM1_\d{6}',
                                                                             im_crop=im_crop,
                                                                             verbose=True,
                                                                             display=False)

# Loading an image of the reference state
img_t = np.array(Image.open(io.BytesIO(bucket_def.Object(im_stack_def[0]).get()['Body'].read())),
                 dtype=float)[im_crop[0]:im_crop[1], im_crop[2]:im_crop[3]]

# Displacement calculation
my_lsa.options['display'] = False
my_lsa.options['verbose'] = False
my_lsa.roi = (np.abs(np.arange(-width/2, width/2)+1) < 500).reshape([-1, 1]) \
    @ (np.abs(np.arange(-width/2, width/2)+1) < 500).reshape([1, -1])
my_lsa.roi[:, :100] = False
my_lsa.roi[:100, :] = False
phi_t, __ = my_lsa.compute_phases_mod(img_t, kernel)
# my_lsa.options['display'] = True
phi_t, uinit = my_lsa.temporal_unwrap(img_0, img_t, phi_r, phi_t)
displ_ims = my_lsa.compute_displacement(phi_r, phi_t, uinit=uinit)

# %% LSA - calculation of the reference state using a single image
phi_0, __ = my_lsa.compute_phases_mod(img_0, kernel)
phi_t, uinit = my_lsa.temporal_unwrap(img_0, img_t, phi_0, phi_t)
displ_reg = my_lsa.compute_displacement(phi_0, phi_t, uinit=uinit)

# %% LSA - calculation of the current state using also an image stack
my_lsa2, phi_t, img_t, kernel, stats = OpenLSA.compute_refstate_from_im_stack(im_extensions='.tif',
                                                                              s3_dictionary=s3_dict_def,
                                                                              im_pattern=r'CAM1_\d{6}',
                                                                              im_crop=im_crop,
                                                                              kernel_std=my_lsa.pitch().max(),
                                                                              verbose=True,
                                                                              display=False)
my_lsa2.roi = (np.abs(np.arange(-width/2, width/2)+1) < 500).reshape([-1, 1]) \
    @ (np.abs(np.arange(-width/2, width/2)+1) < 500).reshape([1, -1])
my_lsa2.roi[:, :100] = False
my_lsa2.roi[:100, :] = False
phi_t, uinit = my_lsa.temporal_unwrap(img_0, img_t, phi_r, phi_t)
displ_200 = my_lsa2.compute_displacement(phi_r, phi_t, uinit=uinit)

# %% Computing strain fields and extracting line data
eps_22_reg, __ = np.gradient(displ_reg.imag)
eps_22_ims, __ = np.gradient(displ_ims.imag)
eps_22_200, __ = np.gradient(displ_200.imag)
map_std = [np.nanstd(eps_22_reg-eps_22_200),
           np.nanstd(eps_22_ims-eps_22_200)]

Points_x = [531, 615]
Points_y = [795, 752]
Line_dist = np.sqrt((Points_x[1]-Points_x[0])**2 + (Points_y[1]-Points_y[0])**2)
points_s = np.arange(int(Line_dist))
points_x = Points_x[0] + points_s/Line_dist*(Points_x[1]-Points_x[0])
points_y = Points_y[0] + points_s/Line_dist*(Points_y[1]-Points_y[0])

tmp_reg = eps_22_reg.copy()
tmp_reg[np.isnan(tmp_reg)] = 0
tmp_reg_fct = interpolate.RectBivariateSpline(np.arange(tmp_reg.shape[0]),
                                              np.arange(tmp_reg.shape[1]),
                                              tmp_reg)
eps_22_reg_line = np.array([tmp_reg_fct(points_x[i], points_y[i])[0]
                            for i in range(len(points_x))]).ravel()

tmp_ims = eps_22_ims.copy()
tmp_ims[np.isnan(tmp_ims)] = 0
tmp_ims_fct = interpolate.RectBivariateSpline(np.arange(tmp_ims.shape[0]),
                                              np.arange(tmp_ims.shape[1]),
                                              tmp_ims)
eps_22_ims_line = np.array([tmp_ims_fct(points_x[i], points_y[i])[0]
                            for i in range(len(points_x))]).ravel()

tmp_200 = eps_22_200.copy()
tmp_200[np.isnan(tmp_200)] = 0
tmp_200_fct = interpolate.RectBivariateSpline(np.arange(tmp_200.shape[0]),
                                              np.arange(tmp_200.shape[1]),
                                              tmp_200)
eps_22_200_line = np.array([tmp_200_fct(points_x[i], points_y[i])[0]
                            for i in range(len(points_x))]).ravel()

line_data = [eps_22_reg_line, eps_22_ims_line, eps_22_200_line]

# %% Makeing figure
texts = ['std($\u03b5^{1~im}_{22}$-$\u03b5^{ex}_{22}$)=%1.2fx$10^{-4}$ [px]',
         'std($\u03b5^{im~stack}_{22}$-$\u03b5^{ex}_{22}$)=%1.2fx$10^{-4}$ [px]']

line_styles = ['-', '-', '--']
line_colors = ['k', 'r', 'b']

formatter = ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-3, 3))
fig, axs = plt.subplots(1, 3)
for j in range(2):
    axs[j].get_xaxis().set_visible(False)
    axs[j].get_yaxis().set_visible(False)
    axs[j].axis('off')
    axs[j].plot(Points_x, Points_y, color=line_colors[j])

im = [axs[0].imshow(eps_22_reg, vmin=-4e-3, vmax=1e-3),
      axs[1].imshow(eps_22_ims, vmin=-4e-3, vmax=1e-3)]

axs[2].yaxis.set_major_formatter(formatter)
axs[2].yaxis.tick_right()
axs[2].yaxis.set_label_position("right")

j = 0 # => reg LSA
axs[2].plot(points_s, line_data[j], line_styles[j], color=line_colors[j])
divider = make_axes_locatable(axs[j])
t = fig.text(axs[j].get_position().x0 + 0.05,
             axs[j].get_position().y0 + 0.48,
             '$\u03b5^{1~im}_{22}$',
             ha='center', va='center', fontsize=10, color='gray')
t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))

fig.text((axs[j].get_position().x0+axs[j].get_position().x1)/2,
         axs[j].get_position().y0+0.537,
         'Using a single \n image' % (map_std[j]*1e4),
         ha='center', va='center', fontsize=10, color='gray')

fig.text((axs[j].get_position().x0+axs[j].get_position().x1)/2,
         axs[j].get_position().y0+0.2,
         texts[j] % (map_std[j]*1e4),
         ha='center', va='center', fontsize=7, color='gray')

fig.text((axs[j].get_position().x0+axs[j].get_position().x1)/2,
         axs[j].get_position().y0+0.1,
         '(a)',
         ha='center', va='center', fontsize=10, color='k')

fig.text((axs[-1].get_position().x0+axs[-1].get_position().x1)/2,
         axs[-1].get_position().y0+0.18 - 0.04*j,
         'std($\u03b5^{1~im}_{22}$-$\u03b5^{nr}_{22}$)=%1.2fx$10^{-4}$ [px]' % np.nanstd((eps_22_reg_line-eps_22_200_line)*1e4),
         ha='center', va='center', fontsize=8, color=line_colors[j])

fig.text((axs[-1].get_position().x0+axs[-1].get_position().x1)/2,
         axs[-1].get_position().y0+0.1 - 0.04*j,
         '(c)',
         ha='center', va='center', fontsize=10, color='k')
j = 1 # => LSA when ref state computed with an image stack
axs[2].plot(points_s, line_data[j], line_styles[j], color=line_colors[j])
divider = make_axes_locatable(axs[j])
t = fig.text(axs[j].get_position().x0 + 0.065,
             axs[j].get_position().y0 + 0.48,
             '$\u03b5^{im~stack}_{22}$',
             ha='center', va='center', fontsize=10, color='gray')
t.set_bbox(dict(facecolor='white', alpha=0.7, linewidth=0))

fig.text((axs[j].get_position().x0+axs[j].get_position().x1)/2,
         axs[j].get_position().y0+0.537,
         'Using the image \n stack' % (map_std[j]*1e4),
         ha='center', va='center', fontsize=10, color='gray')

fig.text((axs[j].get_position().x0+axs[j].get_position().x1)/2,
         axs[j].get_position().y0+0.2,
         texts[j] % (map_std[j]*1e4),
         ha='center', va='center', fontsize=7, color='gray')

fig.text((axs[j].get_position().x0+axs[j].get_position().x1)/2,
         axs[j].get_position().y0+0.1,
         '(b)',
         ha='center', va='center', fontsize=10, color='k')

fig.text((axs[-1].get_position().x0+axs[-1].get_position().x1)/2,
         axs[-1].get_position().y0+0.18 - 0.04*j,
         'std($\u03b5^{im~stack}_{22}$-$\u03b5^{nr}_{22}$)=%1.2fx$10^{-4}$ [px]' % np.nanstd((eps_22_ims_line-eps_22_200_line)*1e4),
         ha='center', va='center', fontsize=8, color=line_colors[j])
j = 2 # plotting data along a line
axs[2].plot(points_s, line_data[j], line_styles[j], color=line_colors[j])

cax = fig.add_axes([axs[0].get_position().x0+0.01,
                    axs[0].get_position().y1-0.18,
                    axs[1].get_position().x1 - axs[0].get_position().x0, 0.02])
cbar = fig.colorbar(im[1], cax=cax, orientation='horizontal')
cax.xaxis.set_ticks_position('top')
cax.xaxis.set_major_formatter(formatter)

axs[2].set_position([2*axs[1].get_position().x0 - axs[0].get_position().x0,
                     axs[0].get_position().y0 + 0.25,
                     axs[0].get_position().x1-axs[0].get_position().x0,
                     axs[0].get_position().y0 + 0.18])

fig.text((axs[j].get_position().x0+axs[j].get_position().x1)/2,
         axs[j].get_position().y0+0.36,
         '$\u03b5_{22}$ along \n extracted line',
         ha='center', va='center', fontsize=10, color='gray')

fig.text(axs[j].get_position().x0 + 0.029,
         axs[j].get_position().y0 + 0.02,
         '$\u03b5^{nr}_{22}$',
         ha='center', va='center', fontsize=8, color='b')
fig.text(axs[j].get_position().x0 + 0.051,
         axs[j].get_position().y0 + 0.02 + 0.03,
         '$\u03b5^{im~stack}_{22}$',
         ha='center', va='center', fontsize=8, color='k')
fig.text(axs[j].get_position().x0 + 0.034,
         axs[j].get_position().y0 + 0.02 + 2*0.03,
         '$\u03b5^{1~im}_{22}$',
         ha='center', va='center', fontsize=8, color='r')

plt.show(block=False)
plt.show()

plt.savefig("Illustration_ImStack4RefState.pdf", format="pdf", bbox_inches="tight")

