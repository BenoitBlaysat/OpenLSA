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
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Librairies devlopped in Clermont Ferrand
from openlsa import OpenLSA

# Loading images
img_0 = np.array(Image.open("Wood_0.tif"))
img_t = np.array(Image.open("Wood_1.tif"))

# Initializing LSA & kernel
my_lsa = OpenLSA(img_0)
kernel = my_lsa.compute_kernel(std=my_lsa.pitch().max())

# Computing phases
phi_0, __ = my_lsa.compute_phases_mod(img_0, kernel)
phi_t, __ = my_lsa.compute_phases_mod(img_t, kernel)

# Solving the temporal unwraping
phi_t, uinit = my_lsa.temporal_unwrap(img_0, img_t, phi_0, phi_t)

# Computing of the displacement
displacement = my_lsa.compute_displacement(phi_0, phi_t, uinit=uinit)

# Computing strain fields
dude2, dude1 = np.gradient(displacement)
eps_11 = dude1.real
eps_22 = dude2.imag
eps_12 = 0.5*(dude1.imag+dude2.real)

# %% Figures

FlagFig = {'Fig2': True,
           'Fig3': True,
           'Fig4': True,
           'Fig5': True,
           'Fig6': True,
           'Fig7': True,
           'Fig8': True}
SaveAsPdf = True

plt.close('all')

plt.rcParams.update({'font.size': 22})
if (('Fig2' in FlagFig.keys()) and FlagFig['Fig2']) \
or (('Fig7' in FlagFig.keys()) and FlagFig['Fig7']):
    img_wood = np.array(Image.open("Wood.png"))
    fig = plt.figure()
    im = plt.imshow(img_wood, extent=(0, img_0.shape[1], 0, img_0.shape[0]))
    fig.axes[0].set_xlim([-60, img_0.shape[1]+200])
    fig.axes[0].set_ylim([-60, img_0.shape[0]+200])
    fig.axes[0].get_xaxis().set_visible(False)
    fig.axes[0].get_yaxis().set_visible(False)
    fig.axes[0].axis('off')
    fig.axes[0].plot([0, img_0.shape[1], img_0.shape[1], 0, 0],
                     [0, 0, img_0.shape[0], img_0.shape[0], 0],
                     'k')
    plt.arrow(100, 100, 0, 300,
              color='k',
              length_includes_head=True,
              head_width=50)
    plt.text(140, 360, "$e\u0332_2$")
    plt.arrow(100, 100, 300, 0,
              color='k',
              length_includes_head=True,
              head_width=50)
    plt.text(370, 160, "$e\u0332_1$")
    nb_lines = 30
    len_lines = 100
    ang_lines = np.pi/12
    y_end = np.ones(nb_lines)-15
    x_end = np.linspace(0, img_0.shape[1], nb_lines)
    y_start = y_end-len_lines*np.cos(ang_lines)
    x_start = x_end-len_lines*np.sin(ang_lines)
    fig.axes[0].plot([x_end[0], x_end[-1]], [y_end[0]+2, y_end[-1]+2], 'k')
    for i in range(nb_lines):
        fig.axes[0].plot([x_start[i], x_end[i]],
                         [y_start[i], y_end[i]],
                         'k')

    nb_arrows = 10
    x_start = np.linspace(30, img_0.shape[1]-30, nb_arrows)
    fig.axes[0].plot([x_end[0], x_end[-1]], [y_end[0]+2, y_end[-1]+2], 'k')
    for i in range(nb_arrows):
        plt.arrow(x_start[i], img_0.shape[0]+200, 0, -150,
                  color='k',
                  length_includes_head=True,
                  linewidth=5,
                  head_width=20)
    plt.show(block=False)
    if SaveAsPdf:
        plt.savefig("Wood.pdf", format="pdf", bbox_inches="tight")

    fig, ax = plt.subplots()
    im = plt.imshow(img_0, cmap='gray', vmin=0, vmax=2**16-1)
    ax.set_xlabel("position along $e\u0332_1$ [px]")
    ax.set_ylabel("position along $e\u0332_2$ [px]")
    ax.set_xlim([0, img_0.shape[1]])
    ax.set_ylim([0, img_0.shape[0]])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    CloseUp = [500, 500]
    ax.plot([CloseUp[0], CloseUp[0]+50, CloseUp[0]+50, CloseUp[0], CloseUp[0]],
            [CloseUp[1], CloseUp[1], CloseUp[1]+50, CloseUp[1]+50, CloseUp[1]],
            color='red',
            linewidth=2)
    ax.plot([my_lsa.temp_unwrap['pt_2_follow'][1]-15,
             my_lsa.temp_unwrap['pt_2_follow'][1]+15,
             my_lsa.temp_unwrap['pt_2_follow'][1]+15,
             my_lsa.temp_unwrap['pt_2_follow'][1]-15,
             my_lsa.temp_unwrap['pt_2_follow'][1]-15],
            [my_lsa.temp_unwrap['pt_2_follow'][0]-15,
             my_lsa.temp_unwrap['pt_2_follow'][0]-15,
             my_lsa.temp_unwrap['pt_2_follow'][0]+15,
             my_lsa.temp_unwrap['pt_2_follow'][0]+15,
             my_lsa.temp_unwrap['pt_2_follow'][0]-15],
            color='forestgreen',
            linewidth=2)
    ax.plot(my_lsa.temp_unwrap['pt_2_follow'][1], my_lsa.temp_unwrap['pt_2_follow'][0],
            '+r', markersize=30, markeredgewidth=3)
    ax.plot([my_lsa.temp_unwrap['pt_2_follow'][1]-15,
             my_lsa.temp_unwrap['pt_2_follow'][1]+15,
             my_lsa.temp_unwrap['pt_2_follow'][1]+15,
             my_lsa.temp_unwrap['pt_2_follow'][1]-15,
             my_lsa.temp_unwrap['pt_2_follow'][1]-15],
            [my_lsa.temp_unwrap['pt_2_follow'][0]-15,
             my_lsa.temp_unwrap['pt_2_follow'][0]-15,
             my_lsa.temp_unwrap['pt_2_follow'][0]+15,
             my_lsa.temp_unwrap['pt_2_follow'][0]+15,
             my_lsa.temp_unwrap['pt_2_follow'][0]-15],
            color='forestgreen',
            linewidth=3)
    plt.show(block=False)
    if SaveAsPdf:
        plt.savefig("ImgRef.pdf", format="pdf", bbox_inches="tight")

    ax.set_xlim([CloseUp[0]-3, CloseUp[0]+50+3])
    ax.set_ylim([CloseUp[1]-3, CloseUp[1]+50+3])
    plt.show(block=False)
    if SaveAsPdf:
        plt.savefig("ImgRef_cu.pdf", format="pdf", bbox_inches="tight")

    ax.set_xlim([my_lsa.temp_unwrap['pt_2_follow'][1]-15-3,
                 my_lsa.temp_unwrap['pt_2_follow'][1]+15+3])
    ax.set_ylim([my_lsa.temp_unwrap['pt_2_follow'][0]-15-3,
                 my_lsa.temp_unwrap['pt_2_follow'][0]+15+3])
    plt.show(block=False)
    if SaveAsPdf:
        plt.savefig("ImgRef_cu_feature.pdf", format="pdf", bbox_inches="tight")

if ('Fig3' in FlagFig.keys()) and FlagFig['Fig3']:
    img_odd = np.hstack((img_0,
                         np.zeros([img_0.shape[0],
                                   1-np.mod(img_0.shape[1], 2)])))
    img_odd = np.vstack((img_odd,
                         np.zeros([1-np.mod(img_odd.shape[0], 2),
                                   img_odd.shape[1]])))

    fft_img_abs = np.abs(np.fft.fftshift(np.fft.fft2(img_odd)))
    freq_x, freq_y = np.meshgrid(np.linspace(-0.5, 0.5, img_odd.shape[1]),
                                 np.linspace(-0.5, 0.5, img_odd.shape[0]),
                                 indexing='xy')

    fig, ax = plt.subplots(1)
    plt.imshow(np.log10(fft_img_abs), alpha=0.8,
               extent=(-0.5, 0.5, -0.5, 0.5),
               origin='lower')

    # removing central peak
    fft_img_abs[np.sqrt(freq_x**2+freq_y**2) < 1/30] = 1
    fft_img_abs[np.sqrt(freq_x**2+freq_y**2) > 1/(2*np.sqrt(2))] = 1

    # Look for the highest peak
    loc_of_peak = np.unravel_index(np.argmax(fft_img_abs),
                                   fft_img_abs.shape)
    vec_k = freq_x[loc_of_peak] + 1j*freq_y[loc_of_peak]

    # Keeping the one on the side of the spectral representation defined by
    # init_angle
    if (np.abs(np.angle(vec_k))-0) > np.pi/2:
        vec_k *= np.exp(1j*np.pi)
    vec_k = [vec_k, vec_k*np.exp(1j*np.pi/2)]

    tmp = fft_img_abs.copy()
    tmp[fft_img_abs == 1] = np.nan
    plt.imshow(np.log10(tmp),
               extent=(-0.5, 0.5, -0.5, 0.5),
               origin='lower')
    ax.set_xlabel("frequency along $e\u0332_1$ [px$^{-1}$]")
    ax.set_ylabel("frequency along $e\u0332_2$ [px$^{-1}$]")
    ax.set_xticks([-0.5, 0, 0.5])
    ax.set_yticks([-0.5, 0, 0.5])
    ax.plot([-0.26, 0.26, 0.26, -0.26, -0.26],
            [-0.26, -0.26, 0.26, 0.26, -0.26],
            color='black',
            linestyle='dashed',
            linewidth=1)
    plt.show(block=False)
    if SaveAsPdf:
        plt.savefig("ImgRef_fft.pdf", format="pdf", bbox_inches="tight")

    colors = ['#580F41', '#FBDD7E']
    texts = ["$k\u0332_{⍺}$", "$k\u0332_{β}$"]
    for ivec in range(2):
        plt.arrow(0, 0, vec_k[ivec].real, vec_k[ivec].imag,
                  color=colors[ivec],
                  length_includes_head=True,
                  head_width=0.01)
        plt.text(1.1*vec_k[ivec].real,
                 1.1*vec_k[ivec].imag,
                 texts[ivec], color='k')
    ax.set_xlim([-0.26, 0.26])
    ax.set_ylim([-0.26, 0.26])
    ax.set_xticks([-0.2, 0, 0.2])
    ax.set_yticks([-0.2, 0, 0.2])
    plt.show(block=False)
    if SaveAsPdf:
        plt.savefig("ImgRef_fft_veck.pdf", format="pdf", bbox_inches="tight")

if ('Fig4' in FlagFig.keys()) and FlagFig['Fig4']:
    uphi_0, __ = my_lsa.compute_phases_mod(img_0, kernel, unwrap=False)
    tmps = [uphi_0[0], uphi_0[1], phi_0[0].copy(), phi_0[1].copy()]
    names = ['phi0e1_raw', 'phi0e2_raw', 'phi0e1', 'phi0e2']

    for i, phi in enumerate(tmps):
        fig, ax = plt.subplots()
        phi.data[~my_lsa.roi] = np.nan
        im = phi.imshow(cax=ax)
        ax.set_xlabel("position along $e\u0332_1$ [px]")
        ax.set_ylabel("position along $e\u0332_2$ [px]")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('[rad]')
        plt.show(block=False)
        if SaveAsPdf:
            plt.savefig(f"{names[i]}.pdf", format="pdf", bbox_inches="tight")

if ('Fig5' in FlagFig.keys()) and FlagFig['Fig5']:
    tmps = [phi_t[0], phi_t[1]]
    names = ['phi1e1', 'phi1e2']
    bounds = [[-16, -7], [-13, -8]]

    for i, phi in enumerate(tmps):
        fig, ax = plt.subplots()
        im = phi.imshow(cax=ax, vmin=bounds[i][0], vmax=bounds[i][1])
        ax.set_xlabel("position along $e\u0332_1$ [px]")
        ax.set_ylabel("position along $e\u0332_2$ [px]")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('[rad]')
        plt.show(block=False)
        if SaveAsPdf:
            plt.savefig(f"{names[i]}.pdf", format="pdf", bbox_inches="tight")

if ('Fig6' in FlagFig.keys()) and FlagFig['Fig6']:
    tmps = [displacement.real, displacement.imag]
    names = ['u1', 'u2']

    for i, tmp in enumerate(tmps):
        fig, ax = plt.subplots()
        im = plt.imshow(tmp)
        ax.set_xlabel("position along $e\u0332_1$ [px]")
        ax.set_ylabel("position along $e\u0332_2$ [px]")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('[px]')
        plt.show(block=False)
        if SaveAsPdf:
            plt.savefig(f"{names[i]}.pdf", format="pdf", bbox_inches="tight")

if ('Fig8' in FlagFig.keys()) and FlagFig['Fig8']:
    tmps = [eps_11, eps_22, eps_12]
    names = ['eps_11', 'eps_22', 'eps_12']
    bounds = [[-2, 2], [-8, 1], [-2, 2]]

    for i, tmp in enumerate(tmps):
        fig, ax = plt.subplots()
        im = plt.imshow(tmp*1e3, vmin=bounds[i][0], vmax=bounds[i][1])
        ax.set_xlabel("position along $e\u0332_1$ [px]")
        ax.set_ylabel("position along $e\u0332_2$ [px]")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('x$ 10^{-3}$ [-]')
        plt.show(block=False)
        if SaveAsPdf:
            plt.savefig(f"{names[i]}.pdf", format="pdf", bbox_inches="tight")

plt.show()
