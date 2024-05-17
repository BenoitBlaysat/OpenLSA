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

# %% Required Libraries
import numpy as np
from numpy import ma
from scipy.ndimage import map_coordinates
from skimage.restoration import unwrap_phase
import matplotlib.pyplot as plt
from openlsa import openlsa as OpenLSA


# %% Class phase and phases
class Phase():
    """ Phase is a class that helps manipulate phase maps. This class has three attributes:\n
        vec_k: wave vector along which phase has been extracted\n
        data: extracted phase (numpy array)\n
        shape: shape of the numpy array collecting the phase modulation"""

    vec_k = None
    data = None
    shape = None

    def __init__(self, phase, vec_k):
        """ Class constructor """
        assert isinstance(phase, np.ndarray)
        assert isinstance(vec_k, (complex, np.complexfloating))
        self.vec_k = vec_k
        self.data = phase
        self.shape = phase.shape

    def __add__(self, other):
        """Note that + returns a numpy array, not a Phase class"""
        assert isinstance(other, Phase)
        return self.data + other.data

    def __sub__(self, other):
        """Note that + returns a numpy array, not a Phase class"""
        assert isinstance(other, Phase)
        return self.data - other.data

    def copy(self):
        """ Method that copies a given Phase class"""
        return Phase(self.data.copy(), self.vec_k.copy())

    def unwrap(self, roi=None):
        """ Method that unwraps the phase map. A region of interest can be provided to reduce
        computing cost"""
        if roi is None:
            roi = np.ones(self.data.shape, dtype='bool')
        else:
            assert isinstance(roi, np.ndarray)
            assert roi.dtype == bool
        phi = ma.masked_array(self.data, roi)
        self.data = np.array(unwrap_phase(phi).filled(0))

    def interp(self, point_yx_input, order=1):
        """ Method that interpolates the phase map."""
        assert isinstance(point_yx_input, np.ndarray)
        assert point_yx_input.dtype in (int, float, np.generic, np.complexfloating)
        if point_yx_input.dtype == np.complexfloating:
            point_yx_input = [point_yx_input.imag, point_yx_input.real]
        return Phase(map_coordinates(self.data,
                                     np.array(point_yx_input).reshape([2, -1]),
                                     order=order).ravel(), self.vec_k)

    def add_corr(self, corr):
        """ Method that add a correction to the phase map."""
        assert isinstance(corr, np.ndarray)
        assert corr.shape == self.shape or np.prod(corr.shape) == 1
        self.data[self.data != 0] += corr.ravel()

    def vec_dir(self):
        """ Method that returns the unit vector of the vector wave assigned to the phase maps."""
        return self.vec_k/np.abs(self.vec_k)

    def format_as_xy(self):
        """ Method that returns the phase maps as a field of vectors.
        Complex writing style is used."""
        return self.data*self.vec_dir()

    def imshow(self, cax=None, **kwargs):
        """ Method that returns the handle of the axis of a figure displaying the phase map"""
        if cax is None:
            __, cax = plt.subplots()
        else:
            assert isinstance(cax, plt.Axes)
        return cax.imshow(self.data, **kwargs)

    def save(self, filename):
        """ Method that writes into a .npz file a given Phase class"""
        assert isinstance(filename, str)
        np.savez_compressed(filename.split(".")[0] + 'npz',
                            vec_k=self.vec_k,
                            data=self.data)

    @staticmethod
    def load(filename):
        """ Method that reads a .npz file and creates a Phase class"""
        assert isinstance(filename, str)
        with np.load(filename.split(".")[0] + 'npz') as data:
            tmp = OpenLSA(data['data'], data['vec_k'])
        return tmp


class Phases():
    """ Phases is a class that helps manipulate collection of phase maps. It is mainly a list of
    Phase and it enables method propagation from the list to each of its item.\n
    This class has two attributes:\n
        phases: list of Phase classes\n
        shape: shape of the any class Phase of phases"""

    def __init__(self, list_of_phases):
        assert isinstance(list_of_phases, list)
        for phase in list_of_phases:
            assert isinstance(phase, Phase)
        self.phases = list_of_phases
        self.shape = list_of_phases[0].shape

    def __len__(self):
        return len(self.phases)

    def __add__(self, other):
        """Note that + returns a list of numpy array, not a Phases class"""
        assert isinstance(other, Phases)
        return [self.phases[i] + other.phases[i] for i in range(len(self))]

    def __sub__(self, other):
        """Note that + returns a list of numpy array, not a Phases class"""
        assert isinstance(other, Phases)
        return [self.phases[i] - other.phases[i] for i in range(len(self))]

    def __getitem__(self, index):
        assert isinstance(index, (list, int))
        if isinstance(index, int):
            return self.phases[index]
        return Phases(self.phases[index])

    def copy(self):
        """ Method that copies a given Phases class"""
        return Phases([self.phases[i].copy() for i in range(len(self))])

    def unwrap(self, roi=None):
        """ Method that unwraps the list of phase maps.
        A region of interest can be provided to reduce computing cost"""
        if roi is None:
            roi = np.ones(self.phases[0].shape, dtype='bool')
        else:
            assert isinstance(roi, np.ndarray)
            assert roi.dtype == bool
        for i in range(len(self)):
            self.phases[i].unwrap(roi=roi)

    def interp(self, point_yx, order=1):
        """ Method that interpolates the list of phase maps."""
        return Phases([self[i].interp(point_yx, order=order) for i in range(len(self))])

    def add_corr(self, corrs):
        """ Method that add a correction to the list of phase maps."""
        for i in range(len(self)):
            self.phases[i].add_corr(corrs[i])

    def vec_k(self, comp=None):
        """ Method that returns the list of vector waves assigned to the list of phase maps."""
        if comp is None:
            return [self.vec_k(comp=i) for i in range(len(self))]
        return self.phases[comp].vec_k

    def vec_karray(self, comp=None):
        """ Method that returns an array made with the list of vector waves assigned to the list
        of phase maps."""
        return np.array(self.vec_k(comp)).reshape(-1, 1)

    def vec_dir(self, comp=None):
        """ Method that returns the list of unit vectors associated with the vector wave assigned
        to the phase maps."""
        if comp is None:
            return [self.vec_dir(comp=i) for i in range(len(self))]
        return self.phases[comp].vec_dir()

    def format_as_xy(self):
        """ Method that returns the list of phase maps as field of vectors.
        Complex writing style is used."""
        return [self.phases[i].format_as_xy() for i in range(len(self))]

    def save(self, filename):
        """ Method that writes into a .npz file a given Phase class"""
        tmp_vec_k = np.zeros([1, len(self)], complex)
        tmp_data = np.zeros([self.shape[0], self.shape[1], len(self)])
        for iloop in range(len(self)):
            tmp_vec_k[:, iloop] = self.phases[iloop].vec_k
            tmp_data[:, :, iloop] = self.phases[iloop].data
        np.savez_compressed(filename.split(".")[0] + '.npz',
                            vec_k=tmp_vec_k,
                            data=tmp_data)

    @staticmethod
    def load(filename):
        """ Method that reads a .npz file and creates a Phases class"""
        with np.load(filename.split(".")[0] + '.npz') as data:
            tmp_vec_k = data['vec_k']
            tmp_data = data['data']
        tmp = Phases([Phase(tmp_data[:, :, i], tmp_vec_k[:, i].tolist()[0])
                      for i in range(tmp_vec_k.shape[1])])
        return tmp
