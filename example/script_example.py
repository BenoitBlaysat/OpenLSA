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

# Librairies devlopped in Clermont Ferrand
from openlsa import OpenLSA

# Loading images
img_0 = np.array(Image.open("Wood_0.tif"))
img_t = np.array(Image.open("Wood_1.tif"))

# Initializing LSA & kernel
my_lsa = OpenLSA(img_0, verbose=True, display=True)
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
