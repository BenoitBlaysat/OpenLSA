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
import os
import glob
import io
import pickle
from types import NoneType
import numpy as np
from scipy.ndimage import map_coordinates
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from openlsa.utils import make_it_uint8
from openlsa.utils import scal_prod, axy_2_a01, a01_2_axy, compute_rbm, estimate_u, reject_outliers
from openlsa.utils import provide_s3_path
from openlsa.phase import Phase, Phases


# %% Class LSA
class OpenLSA():
    """ This Class merges the recent progresses the Clermont's EM team made about the Localized
    Spectrum Analysis technique. This class has four attibutes and many methods. The four
    attributes are
        vec_k: this list collects the wave vectors that corresponds to the reference state.
        roi: this array of booleans defines to the studied region of interest, 1 or True
             corresponding to the pixel of interest.
        options: this dictionary defines the verbose attributes of the methods. It can be
                 restricted to terminal outputs (verbose key) or provide figures (display key).
        temp_unwrap: this is a dictionary helpful for the temporal unwrapping.
    An extra (hidden) attribute consists to the pixel coordinates __px_z. Complex writting style
    is used: its real (resp. imaginary) part corresponds to the column (resp. line) coordinates"""

    vec_k = None
    roi = None
    options = {'display': None,
               'verbose': None}
    temp_unwrap = {'pt_2_follow': None,
                   'template': None,
                   'roi_75percent': None}
    __px_z = None

    # %% Class constructor
    def __init__(self, img=None,
                 vec_k=None, max_pitch=30, min_pitch=2*np.sqrt(2),
                 init_angle=0,
                 roi=None,
                 pt_2_follow=None,
                 template=None,
                 roi_75percent=None,
                 display=False,
                 verbose=False):

        # << ------ check if input variables are correct
        assert isinstance(img, (NoneType, np.ndarray))
        assert isinstance(vec_k, (NoneType, list))
        if isinstance(vec_k, list):
            for vec in vec_k:
                assert isinstance(vec, (complex, np.complexfloating))
        assert_point(pt_2_follow)
        assert isinstance(max_pitch, (int, np.generic))
        assert max_pitch > 0
        assert isinstance(min_pitch, (int, np.generic))
        assert min_pitch >= 2*np.sqrt(2)
        assert min_pitch < max_pitch
        assert isinstance(init_angle, (int, np.generic))
        assert isinstance(roi, (NoneType, np.ndarray))
        if isinstance(roi, np.ndarray):
            assert roi.dtype == bool
            if isinstance(img, np.ndarray):
                assert img.shape == roi.shape
        assert isinstance(template, (NoneType, np.ndarray))
        assert isinstance(roi_75percent, (NoneType, np.ndarray))
        if isinstance(roi_75percent, np.ndarray):
            assert roi_75percent.dtype == bool
        assert isinstance(display, bool)
        assert isinstance(verbose, bool)
        # ------ >>

        self.vec_k = vec_k
        self.roi = roi
        self.temp_unwrap['pt_2_follow'] = pt_2_follow
        self.temp_unwrap['template'] = template
        self.temp_unwrap['roi_75percent'] = roi_75percent
        self.options['display'] = display
        self.options['verbose'] = verbose

        if img is None:
            if vec_k is None:
                print(["Please feed me, I have nothing to eat here... I need an image or the",
                       " pitch and the angle of the periodic pattern"])
                return

        if img is not None:
            if vec_k is None:
                self.__compute_vec_k(img,
                                     max_pitch=max_pitch, min_pitch=min_pitch,
                                     init_angle=init_angle)

        if vec_k is not None:
            if isinstance(vec_k, list) and len(vec_k) == 2:
                self.vec_k = vec_k
            elif vec_k.dtype == 'complex':
                self.vec_k = [vec_k, vec_k*np.exp(1j*np.pi/2)]
            else:
                print('error')

        if img is not None:
            self.__def_px_loc(img.shape)
        elif roi is not None:
            self.__def_px_loc(roi.shape)

    # %% extra constructors
    # defining pixel coordinate system
    def __def_px_loc(self, dim):
        px_x, px_y = np.meshgrid(np.arange(dim[1], dtype=int), np.arange(dim[0], dtype=int))
        self.__px_z = px_x + 1j*px_y

    # computing pattern properties, i.e. wave vectors
    def __compute_vec_k(self,
                        img,
                        max_pitch=30,
                        min_pitch=2*np.sqrt(2),
                        init_angle=0):
        """Compute the Pitch and the Angle from the location of the peak in the spectral
        representation of a given image "img"
        "max_pitch"" refers to the highest possible pitch"""

        if self.options['verbose']:
            print('\n Looking for pattern properties\n',
                  '------------------------------')

        img_odd = np.hstack((img,
                             np.zeros([img.shape[0],
                                       1-np.mod(img.shape[1], 2)])))
        img_odd = np.vstack((img_odd,
                             np.zeros([1-np.mod(img_odd.shape[0], 2),
                                       img_odd.shape[1]])))

        fft_img_abs = np.abs(np.fft.fftshift(np.fft.fft2(img_odd)))
        freq_x, freq_y = np.meshgrid(np.linspace(-0.5, 0.5, img_odd.shape[1]),
                                     np.linspace(-0.5, 0.5, img_odd.shape[0]),
                                     indexing='xy')

        if self.options['display']:
            plt.subplots(1)
            plt.imshow(np.log10(fft_img_abs), alpha=0.8,
                       extent=(-0.5, 0.5, -0.5, 0.5),
                       origin='lower')

        # removing central peak
        fft_img_abs[np.sqrt(freq_x**2+freq_y**2) < 1/max_pitch] = 1

        # removing area conducting to too small pitch
        fft_img_abs[np.sqrt(freq_x**2+freq_y**2) > 1/min_pitch] = 1

        # Look for the highest peak
        loc_of_peak = np.unravel_index(np.argmax(fft_img_abs),
                                       fft_img_abs.shape)
        vec_k = freq_x[loc_of_peak] + 1j*freq_y[loc_of_peak]

        # Keeping the one on the side of the spectral representation defined by
        # init_angle
        angle = np.mod(np.angle(vec_k)-init_angle, np.pi/2) + init_angle
        if angle > np.pi/2:
            angle -= np.pi/2
        vec_k = np.abs(vec_k)*np.exp(1j*(angle))
        self.vec_k = [vec_k, vec_k*np.exp(1j*np.pi/2)]

        if self.options['display']:
            tmp = fft_img_abs.copy()
            tmp[fft_img_abs == 1] = np.nan
            plt.imshow(np.log10(tmp),
                       extent=(-0.5, 0.5, -0.5, 0.5),
                       origin='lower')
            plt.title('Spectogram (log scale)')
            colors = ['red', 'blue']
            texts = ["$k_1$", "$k_2$"]
            for ivec in range(2):
                plt.arrow(0, 0, self.vec_k[ivec].real, self.vec_k[ivec].imag,
                          color=colors[ivec],
                          length_includes_head=True,
                          head_width=0.01)
                plt.text(1.1*self.vec_k[ivec].real,
                         1.1*self.vec_k[ivec].imag,
                         texts[ivec], color='k')
            plt.xlabel("frequency along ${e}_1$ [px$^{-1}$]")
            plt.ylabel("frequency along ${e}_2$ [px$^{-1}$]")
            plt.show()

        if self.options['verbose']:
            print(f"      Pattern pitch = {self.pitch(0):02.02f} [px]\n"
                  f"      Pattern angle = {self.angle(0, deg=True):03.02f} [deg]")

    # %% Some usefull functions
    def copy(self):
        """ Method that copies an OpenLSA class."""
        return OpenLSA(vec_k=self.vec_k, roi=self.roi,
                       pt_2_follow=self.temp_unwrap['pt_2_follow'],
                       template=self.temp_unwrap['template'],
                       roi_75percent=self.temp_unwrap['roi_75percent'],
                       display=self.options['display'],
                       verbose=self.options['verbose'])

    # %% Some usefull functions
    def pitch(self, comp=None):
        """  Method that returns the list of the pitchs [px] through which the periodic pattern is
        encoded. If a component (comp) is specified, output is reduced to it."""
        if comp is None:
            return 1/np.abs(self.vec_k)
        else:
            assert isinstance(comp, int)

        return 1/np.abs(self.vec_k[comp])

    def angle(self, comp=None, deg=False):
        """  Method that returns the list of the angles [rad] through which the periodic pattern is
        encoded. If a component (comp) is specified, output is reduced to it.  if deg is true,
        output is given in degrees"""
        assert isinstance(deg, bool)
        if comp is None:
            return np.angle(self.vec_k, deg=deg)
        else:
            assert isinstance(comp, int)
        return np.angle(self.vec_k[comp], deg=deg)

    def vec_dir(self, comp=None):
        """  Method that returns the list of the uni vector [-] through which the periodic pattern
        is encoded. If a component (comp) is specified, output is reduced to it."""
        if comp is None:
            return np.exp(1j*np.angle(self.vec_k))
        else:
            assert isinstance(comp, int)
        return np.exp(1j*np.angle(self.vec_k[comp]))

    def px_z(self, roi=False):
        """  Method that returns pixel coordinates, formated as complex."""
        assert isinstance(roi, bool)

        if roi:
            return self.__px_z[self.roi]
        return self.__px_z

    # %% Kernel building
    def compute_kernel(self, std=None):
        """ Method that computes LSA gaussian normalized kernel, of standard diviation "std"."""
        if std is None:
            std = self.pitch().max()
        else:
            assert isinstance(std, (int, float, np.generic))

        t_noy = np.ceil(4*std)
        px_x, px_y = np.meshgrid(np.arange(-t_noy, t_noy+1), np.arange(-t_noy, t_noy+1))
        kernel = np.exp(-(px_x**2+px_y**2)/(2*std**2))
        return kernel/np.sum(kernel)

    # %% LSA core functions
    def compute_mod_arg(self, img, vec_k, kernel):
        """Method that computes the convolution between the kernel and the WFT taken at the
        frequency of |vec_k| and in the direction of its angle.
        vec_k is the wave vector that characterize the pattern periodicity
        kernel is the kernel used for LSA"""
        assert isinstance(img, np.ndarray) and isinstance(img.item(0), (int, float, np.generic))
        assert isinstance(vec_k, (complex, np.complexfloating))
        assert isinstance(kernel, np.ndarray) \
            and isinstance(kernel.item(0), (int, float, np.generic))

        w_f_r = cv2.filter2D(img*np.cos(-2*np.pi*scal_prod(vec_k, self.__px_z)), -1, kernel)
        w_f_i = cv2.filter2D(img*np.sin(-2*np.pi*scal_prod(vec_k, self.__px_z)), -1, kernel)
        w_f = w_f_r + 1j*w_f_i
        return np.abs(w_f), Phase(np.angle(w_f), vec_k)

    def compute_phases_mod(self, img, kernel=None, roi_coef=0.2, unwrap=True):
        """LSA coreL return phases and magnitudes of an image for a list of wave vectors
        kernel is the kernel used for LSA
        roi_coef defines the thresshold used for defining the region of interest
        unwrap is an option for returning wrapped phase modulations."""
        assert isinstance(img, np.ndarray) and isinstance(img.item(0), (int, float, np.generic))
        assert isinstance(roi_coef, (int, float, np.generic))
        assert 0 < roi_coef
        assert isinstance(unwrap, bool)

        if self.options['verbose']:
            print('\n Computing the phase modulations\n',
                  '-------------------------------')
        if kernel is None:
            kernel = self.compute_kernel()
        else:
            assert isinstance(kernel, np.ndarray) \
                and isinstance(kernel.item(0), (int, float, np.generic))

        if self.__px_z is None:
            self.__def_px_loc(img.shape)

        mods, phis = [None]*len(self.vec_k), [None]*len(self.vec_k)
        for i, vec_k in enumerate(self.vec_k):
            mods[i], phis[i] = self.compute_mod_arg(img, vec_k, kernel)
        phi = Phases(phis)

        # let's compute a equivalent pixel wise modulus -> used for defining a masked area to
        # reduce the unwraping computing cost
        mod = sum(mods[i]**2 for i in range(len(mods)))
        loc_roi = mod < mod.max()*roi_coef**2

        # unwrap phases
        if unwrap:
            phi.unwrap(loc_roi)

        # define the RoI if undefined
        if self.roi is None:
            self.roi = ~loc_roi

        if self.temp_unwrap['roi_75percent'] is None:
            self.temp_unwrap['roi_75percent'] = mod > (mod.max()*0.75**2)

        return phi, mods

    def temporal_unwrap(self, img1, img2, phi_1, phi_2,
                        point1=None, point2=None, uinit=None):
        """ Method that does the temporal unwrap between two phases (from phi_1 to phi_2).
        Corresponding images are used (img1 and img2), and an initial displacement uiint can be
        provided to help the pairing process."""
        assert isinstance(img1, np.ndarray) and isinstance(img1.item(0), (int, float, np.generic))
        assert isinstance(img2, np.ndarray) and isinstance(img2.item(0), (int, float, np.generic))
        assert isinstance(phi_1, Phases)
        assert phi_1.shape == img1.shape
        assert isinstance(phi_2, Phases)
        assert phi_2.shape == img1.shape
        assert_point(point1)
        assert_point(point2)
        assert isinstance(uinit, (NoneType, np.ndarray))
        if isinstance(uinit, np.ndarray):
            assert uinit.shape == img1.shape

        self.check_temp_unwrap(img1, point1=point1)
        if point2 is None:
            point2, uinit = self.rough_point2point(img1, img2, dis_init=uinit)
        return self.jump_correction(phi_1, phi_2, point2), uinit

    def check_temp_unwrap(self, img, point1=None):
        """ Method that checks if the features neede for the temporal unwrap have been
        initialized. If not, it runs the methods to make it."""
        assert isinstance(img, np.ndarray) and isinstance(img.item(0), (int, float, np.generic))
        assert_point(point1)

        if point1 is None:
            if self.temp_unwrap['pt_2_follow'] is None:
                self.init_pt_2_follow(img)
        else:
            if self.temp_unwrap['pt_2_follow'] is not None:
                if np.linalg.norm(self.temp_unwrap['pt_2_follow']-point1) > 1e-14:
                    print('Warning: overwriting the location of point to follow accross images.')
            self.temp_unwrap['pt_2_follow'] = point1
        self.init_template(img)

    def init_pt_2_follow(self, img):
        """ Method that defines the location of the feature to be followed accross images."""
        assert isinstance(img, np.ndarray) and isinstance(img.item(0), (int, float, np.generic))

        blur_size = int(self.pitch().max()**3)
        roi_75percent = cv2.blur(make_it_uint8(255*self.temp_unwrap['roi_75percent']),
                                 (blur_size, blur_size))/255.
        img_roi = cv2.blur((img*roi_75percent).astype('uint8'),
                           (int(np.ceil(self.pitch().max())), int(np.ceil(self.pitch().max()))))
        img_roi = img_roi*self.roi
        self.temp_unwrap['pt_2_follow'] = np.array(np.unravel_index(np.argmax(img_roi),
                                                                    img_roi.shape))

    def init_template(self, img):
        """ Method that defines the feature, i.e. template, to be followed accross images."""
        assert isinstance(img, np.ndarray) and isinstance(img.item(0), (int, float, np.generic))

        ceil_pitch = int(np.ceil(self.pitch().max()))
        width = 2*ceil_pitch
        point1 = self.temp_unwrap['pt_2_follow'].ravel()
        self.temp_unwrap['template'] = cv2.blur(img,
                                                (ceil_pitch,
                                                 ceil_pitch))[point1[0]-width:point1[0]+width+1,
                                                              point1[1]-width:point1[1]+width+1]

    def rough_point2point(self, img1, img2, point1=None, dis_init=None):
        """ Method that calculates the location of a feature in img1 in img2. point1 is the pixel
        coordinates of the feature in img1, dis_init corresponds to an initial guess of the
        displacement from img1 to img2. It is formated into a complex number, the real part
        being the displacement, in the line direction, and the imaginary part the displacement
        in the direction of the columns."""
        assert isinstance(img1, np.ndarray) and isinstance(img1.item(0), (int, float, np.generic))
        assert isinstance(img2, np.ndarray) and isinstance(img2.item(0), (int, float, np.generic))
        assert img2.shape == img1.shape
        assert_point(point1)
        assert isinstance(dis_init, (NoneType, np.ndarray))
        if isinstance(dis_init, np.ndarray):
            assert dis_init.shape == img1.shape

        if self.options['verbose']:
            print('\n Estimate rough displacement for temporal unwrapping\n',
                  '---------------------------------------------------')

        img1_8bit = make_it_uint8(img1)
        img2_8bit = make_it_uint8(img2)
        self.check_temp_unwrap(img1_8bit, point1=point1)
        point1 = self.temp_unwrap['pt_2_follow'].ravel()

        # calculation of the rough displacement thanks to Open
        flow = estimate_u(img1_8bit, img2_8bit,
                          filter_size=(self.pitch().max()/np.sqrt(2))/2, dis_init=dis_init)
        if map_coordinates(self.roi, point1.reshape([2, -1])) != 1:
            print('Given point to follow is not within the roi')

        # Looking for the displacement thanks to DisFlow
        if np.linalg.norm(point1-point1.astype(int)) == 0:
            point2 = [point1 + (flow.imag[tuple(point1)], flow.real[tuple(point1)])]
        else:
            print('The given point is not an integer, an interpolation is introduced')
            uxy = map_coordinates(flow, point1.reshape([2, -1]))
            point2 = [point1 + (uxy.imag, uxy.real)]

        # Looking for the displacement thanks to template matching
        res = cv2.matchTemplate(cv2.blur(img2_8bit, (int(np.ceil(self.pitch().max())),
                                                     int(np.ceil(self.pitch().max())))),
                                self.temp_unwrap['template'], cv2.TM_SQDIFF_NORMED)
        __, __, min_loc, __ = cv2.minMaxLoc(res)
        point2.append((np.array((min_loc[1], min_loc[0]))
                       + (self.temp_unwrap['template'].shape[0] - 1)/2).astype(float))

        if np.linalg.norm(point2[1] - point2[0]) > 1:
            # Calculation of the two residuals (using the SSD criteria)
            ssd = [np.linalg.norm(img1[int(point1[0])-15:int(point1[0])+15,
                                       int(point1[1])-15:int(point1[1])+15]
                                  - img2[int(point2[i][0])-15:int(point2[i][0])+15,
                                         int(point2[i][1])-15:int(point2[i][1])+15])
                   for i in range(2)]

            print("WARNING - temporal unwrapping \n"
                  "     Disflow and pattern matching provided different results : \n"
                  f"         DisFlow           = [{point1[0]:d}, {point1[1]:d}] "
                  f"-> [{point2[0][0]:0.2f}, {point2[0][1]:0.2f}] "
                  f"-> SSD = {ssd[0]:0.2f}\n"
                  f"         Pattern Matching  = [{point1[0]:d}, {point1[1]:d}] "
                  f"-> [{point2[1][0]:0.2f}, {point2[1][1]:0.2f}] "
                  f"-> SSD = {ssd[1]:0.2f}")

            # The solution corresponds to the one of smallest residual
            if ssd[0] > ssd[1]:
                point2 = point2.reverse()

        if self.options['display']:

            __, fig_ax = plt.subplots(3)
            fig_ax[0].set_title('Pattern in the ref image')
            fig_ax[0].imshow(img1[int(point1[0])-15:int(point1[0])+15,
                                  int(point1[1])-15:int(point1[1])+15],
                             cmap='gray', vmin=0, vmax=2**(round(np.log2(img1.max())))-1)
            fig_ax[1].set_title('Pattern in the cur image (DisFlow)')
            fig_ax[1].imshow(img2[int(point2[0][0])-15:int(point2[0][0])+15,
                                  int(point2[0][1])-15:int(point2[0][1])+15],
                             cmap='gray', vmin=0, vmax=2**(round(np.log2(img1.max())))-1)
            fig_ax[2].set_title('Pattern in the cur image (Pattern matching)')
            fig_ax[2].imshow(img2[int(point2[1][0])-15:int(point2[1][0])+15,
                                  int(point2[1][1])-15:int(point2[1][1])+15],
                             cmap='gray', vmin=0, vmax=2**(round(np.log2(img1.max())))-1)
            plt.tight_layout()
            plt.show()

        return point2[0], flow

    def jump_correction(self, phi_1, phi_2, point2, point1=None):
        """ Method that pairs the phase phi_2 to phi_1 accordlingly to the fact that point1 of
        phi_1 moved to point2 of phi_2"""
        assert isinstance(phi_1, Phases)
        assert isinstance(phi_2, Phases)
        assert phi_2.shape == phi_1.shape
        assert isinstance(point2, np.ndarray) and point2.shape == (2,)
        assert_point(point1)

        if point1 is None:
            point1 = self.temp_unwrap['pt_2_follow'].astype(float)
        point1_z = point1 @ [[1j], [1]]
        u_xy = (point2 - point1) @ np.array([[1j], [1]])
        disp = axy_2_a01(phi_2.vec_dir(), u_xy)
        rphi_1, rphi_2 = phi_1.interp(point1), phi_2.interp(point2)
        delta_phi = np.array(rphi_1 - rphi_2)
        corr = np.round(scal_prod(rphi_1.vec_karray()-rphi_2.vec_karray(), point1_z)
                        + delta_phi/(2*np.pi) - np.abs(rphi_2.vec_karray())*disp)
        phi_2.add_corr(2*np.pi*corr)
        return phi_2

    def compute_displacement(self, phi_1, phi_2,
                             list_of_points=None,
                             min_iter=3, max_iter=15, uinit=None):
        """ Method that computes the displacement field from the reference phase (phi_1) to current
        phase (phi_2).
        phi_1: phase of the reference state.
        phi_2: phase of the current state.
        list_of_points: point coordinates where the displacement is seek.
        min_iter: minimum number of iterations of the fixed point algorithm.
        max_iter: maximum number of iterations of the fixed point algorithm.
        uinit: guess for fixed point initialisation
        """
        assert isinstance(phi_1, Phases)
        assert isinstance(phi_2, Phases)
        assert phi_2.shape == phi_1.shape
        assert isinstance(list_of_points, (NoneType, np.ndarray))
        if isinstance(list_of_points, np.ndarray):
            if len(list_of_points) == 1:
                assert list_of_points.shape[0] == 2
            else:
                assert list_of_points.shape[1] == 2
        assert isinstance(min_iter, int)
        assert isinstance(max_iter, (int, float, np.generic))
        assert min_iter <= max_iter
        assert isinstance(uinit, (NoneType, np.ndarray))
        if isinstance(uinit, np.ndarray):
            assert uinit.shape == phi_1.shape

        if self.options['verbose']:
            print('\n Computing the displacement field\n', '--------------------------------')

        # Formating the coordinates of the point where the displacement is seek.
        if list_of_points is None:
            z_roi = self.__px_z[self.roi]
        else:
            z_roi = (list_of_points[:, 0] + 1j*list_of_points[:, 1])

        # Calculation of constant terms
        rphi_1 = phi_1.interp(z_roi)
        abs_k2 = np.abs(phi_2.vec_karray())
        cst_term = scal_prod(phi_1.vec_karray() - phi_2.vec_karray(), z_roi)/abs_k2

        # Displacement initialisation
        if uinit is None:
            u01 = cst_term + np.array(rphi_1 - phi_2.interp(z_roi))/(2*np.pi*abs_k2)
            disp = a01_2_axy(phi_2.vec_dir(), u01.T)
        else:
            disp = map_coordinates(uinit, (z_roi.imag.ravel(),
                                           z_roi.real.ravel()), order=1)

        # Fixed point algorithm
        stop_crit = 5e-4*len(z_roi)
        for loop_n in range(max_iter):
            u01 = cst_term + np.array(rphi_1 - phi_2.interp(z_roi + disp))/(2*np.pi*abs_k2)
            new_disp = a01_2_axy(phi_2.vec_dir(), u01.T)
            delta = new_disp - disp
            disp = new_disp
            if loop_n > min_iter and (np.linalg.norm(delta[np.isfinite(delta)], 2)
                                      < np.sqrt(2)*stop_crit):
                break

        if loop_n == max_iter-1:
            print('WARNING - Displacement calculation - Fixed-point not converged')
        elif self.options['verbose']:
            print(f"     Fixed-point converged in {loop_n:d} iterations")

        if list_of_points is None:
            output_disp = np.zeros(phi_1.shape, complex) + np.nan*(1+1j)
            output_disp[self.roi] = disp.ravel()
            return output_disp
        return disp

    def save(self, filename):
        """ Method that writes a back-up class data file using the pickles format.
        filename is the name/path used to define the write down the data."""
        assert isinstance(filename, str)

        if filename.split(".")[-1] == 'pkl':
            with open(filename, 'wb') as file:
                pickle.dump({'vec_k': self.vec_k, 'roi': self.roi,
                             'pt_2_follow': self.temp_unwrap['pt_2_follow'],
                             'roi_75percent': self.temp_unwrap['roi_75percent'],
                             'template': self.temp_unwrap['template'],
                             'display': self.options['display'],
                             'verbose': self.options['verbose']}, file)
        else:
            print('Error - unknown extension')

    # %% extra function
    @staticmethod
    def compute_refstate_from_im_stack(im_folder=None, im_extensions='.tif', im_pattern='',
                                       im_stack=None, s3_dictionary=None,
                                       roi_coef=0.2, kernel_std=None, **kwargs):
        """ Often, multiple images are taken at reference state. This function extracts phase
        fields for all images, and averages them by taking into account the rigid body motion
        that might occur in between. The reference coordinate system corresponds to the one
        given by the first image. Arguments im_folder, im_extensions, im_pattern, and im_stack
        help define the stack of images, roi_coef corresponds to the coef used for defining the
        region of interest (see the compute_phases_mod method), kernel_width is the std of the LSA
        Gaussian kernel and finally kwargs concatenates all arguments of the OpenLSA constructor.
        s3_dictionary is formated as
            s3_dictionary = {'s3_access_key_id':ACCESS_KEY,
                             's3_secret_access_key':SECRET_KEY,
                             's3_session_token':SESSION_TOKEN,
                             's3_endpoint_url': s3_endpoint_url
                             's3_bucket_name': s3_bucket_name,
                             's3_path_2_im': s3_path_2_im,
                             's3_path_2_folder': s3_path_2_folder}"""
        assert isinstance(im_folder, (NoneType, str))
        assert isinstance(im_extensions, (str, list))
        if isinstance(im_extensions, list):
            for im_extension in im_extensions:
                assert isinstance(im_extension, str)
        assert isinstance(im_pattern, str)
        assert isinstance(im_stack, (NoneType, list))
        if isinstance(im_stack, list):
            assert isinstance(im_stack[0], np.ndarray)
            for img in im_stack[1:]:
                assert isinstance(img, np.ndarray)
                assert img.shape == im_stack[0].shape
        assert isinstance(s3_dictionary, (NoneType, dict))
        assert isinstance(roi_coef, (int, float))
        assert 0 < roi_coef
        assert isinstance(kernel_std, (NoneType, int, float))

        if s3_dictionary is None:
            s3_dictionary = {'s3_access_key_id': None, 's3_secret_access_key': None,
                             's3_session_token': None, 's3_endpoint_url': None,
                             's3_bucket_name': None, 's3_path_2_im': None,
                             's3_im_pattern': None, 's3_path_2_folder': None}

        s3_flag = False
        im_extensions = [im_extensions] if not isinstance(im_extensions, list) else im_extensions
        opt_display_and_verbose = [False, False]
        if 'display' in kwargs:
            opt_display_and_verbose[0] = kwargs['display']
        if 'verbose' in kwargs:
            opt_display_and_verbose[1] = kwargs['verbose']

        if opt_display_and_verbose[1]:
            print('\n Computing a noise-reduced reference state\n',
                  '-----------------------------------------')

        if im_folder is not None:
            im_stack = []
            for ext in im_extensions:
                im_stack.extend(glob.glob(os.path.join(im_folder, f"*{im_pattern}*{ext}")))
            if opt_display_and_verbose[1]:
                print("      A path to a folder is given, ", f"{len(im_stack)} images are found.")

        if im_stack is None:
            if s3_dictionary is None:
                print('\n --------------------',
                      '\n Error - I need paths to images or at least to a folder',
                      '\n --------------------')
            else:
                s3_flag = True
                im_stack, bucket = provide_s3_path(s3_dictionary, im_extensions, im_pattern,
                                                   opt_display_and_verbose[1])

        if opt_display_and_verbose[1]:
            print(f"      Reference state defined with the image named: \n      -> {im_stack[0]}")

        if s3_flag:
            data_object = bucket.Object(im_stack[0])
            img_ref = np.array(Image.open(io.BytesIO(data_object.get()['Body'].read())),
                               dtype=float)
        else:
            img_ref = np.array(Image.open(im_stack[0]), dtype=float)

        mylsa = OpenLSA(img_ref, **kwargs)
        if kernel_std is None:
            kernel_std = mylsa.pitch().max()
        kernel = mylsa.compute_kernel(std=kernel_std)
        mylsa.options['display'], mylsa.options['verbose'] = False, False

        if opt_display_and_verbose[1]:
            print(f"      Step 1/{len(im_stack)}")
            print('        Computing phase modulations')

        phi_ref, mods = mylsa.compute_phases_mod(img_ref, kernel, roi_coef=roi_coef)
        phi_ref_av = phi_ref.copy()
        for comp in [0, 1]:
            phi_ref_av[comp].data[:] = phi_ref_av[comp].data[:]/len(im_stack)

        # let's compute a equivalent pixel wise modulus -> used for defining a masked area
        mod = sum(mods[i]**2 for i in range(len(mods)))*mylsa.roi
        loc_roi = ~(cv2.blur((255*(mod <= (0.5**2)*mod.max())).astype('uint8'), (25, 25)) > 0)
        nb_pixels = np.sum(loc_roi)
        var_coef = 1 + 1/(len(im_stack)-1)
        coord_z = mylsa.px_z(roi=True)

        stats = {'mean_u1': np.zeros(nb_pixels),
                 'mean_u2': np.zeros(nb_pixels),
                 'mean_eps_11': np.zeros(nb_pixels),
                 'mean_eps_12': np.zeros(nb_pixels),
                 'mean_eps_22': np.zeros(nb_pixels),
                 'var_u1': np.zeros(nb_pixels),
                 'var_u2': np.zeros(nb_pixels),
                 'var_eps_11': np.zeros(nb_pixels),
                 'var_eps_12': np.zeros(nb_pixels),
                 'var_eps_22': np.zeros(nb_pixels),
                 'std_eq_u1': 0., 'std_eq_u2': 0.,
                 'std_eq_eps_11': 0., 'std_eq_eps_12': 0., 'std_eq_eps_22': 0.}

        for i, img_name in enumerate(im_stack[1:]):
            if opt_display_and_verbose[1]:
                print(f"      Step {i+2}/{len(im_stack)}")
                print('        Computing phase modulations')
            if s3_flag:
                data_object = bucket.Object(img_name)
                img_loc = np.array(Image.open(io.BytesIO(data_object.get()['Body'].read())),
                                   dtype=float)
            else:
                img_loc = np.array(Image.open(img_name), dtype=float)
            phi_loc = mylsa.compute_phases_mod(img_loc, kernel)[0]
            phi_loc, __ = mylsa.temporal_unwrap(img_ref, img_loc, phi_ref, phi_loc)
            uxy_loc = mylsa.compute_displacement(phi_ref, phi_loc)
            uxy_loc_rbm = compute_rbm(uxy_loc[mylsa.roi], coord_z.real, coord_z.imag)
            phi_loc_rbm = phi_loc.interp(coord_z + uxy_loc_rbm)
            u01_loc_rbm = axy_2_a01(mylsa.vec_dir(), uxy_loc_rbm)

            for comp in [0, 1]:
                phi_u01_loc_rbm = (2*np.pi/mylsa.pitch()[comp])*u01_loc_rbm[comp]
                phi_ref_av[comp].data[mylsa.roi] += (phi_loc_rbm[comp].data
                                                     + phi_u01_loc_rbm)/len(im_stack)

            # Computation of the noise properties of the displacements
            if opt_display_and_verbose[1]:
                print('        Updating the statistical analysis of the displacement',
                      'and strain fields')

            # here the assumption is that the roi is larger than the one considered for the noise
            # estimation ! i.e. the threshold used for defining the roi is smaller than 0.5
            uxy_res = uxy_loc
            uxy_res[mylsa.roi] = uxy_res[mylsa.roi]-uxy_loc_rbm
            dude2, dude1 = np.gradient(uxy_res)

            loc = {'u1': uxy_res[loc_roi].real,
                   'u2': uxy_res[loc_roi].imag,
                   'eps_11': dude1[loc_roi].real,
                   'eps_12': 0.5*(dude1[loc_roi].imag+dude2[loc_roi].real),
                   'eps_22': dude2[loc_roi].imag}

            # variance computation based on the previous variance and average (cf formula)
            for comp in ['u1', 'u2', 'eps_11', 'eps_12', 'eps_22']:
                stats[f"var_{comp}"] = (i/(i+1))*(stats[f"var_{comp}"]
                                                  + ((stats[f"mean_{comp}"] - loc[comp])**2)/(i+1))
                stats[f"mean_{comp}"] = (loc[comp] + i*stats[f"mean_{comp}"])/(i+1)

        item_2_keep = reject_outliers(stats["var_eps_11"]
                                      + 2*stats["var_eps_12"]
                                      + stats["var_eps_22"], bandwitch=3)
        for comp in ['u1', 'u2', 'eps_11', 'eps_12', 'eps_22']:
            stats[f"var_{comp}"][~item_2_keep] = np.nan
            stats[f"mean_{comp}"][~item_2_keep] = np.nan
            stats[f"var_{comp}"] *= var_coef
            stats[f"std_eq_{comp}"] = np.sqrt(np.nanmean(stats[f"var_{comp}"]))

        mylsa.options['display'] = opt_display_and_verbose[0]
        mylsa.options['verbose'] = opt_display_and_verbose[1]

        if mylsa.options['display']:
            for comp in ['u1', 'u2', 'eps_11', 'eps_12', 'eps_22']:
                __, fig_ax = plt.subplots(2)
                # histo
                fig_ax[0].hist(stats[f"var_{comp}"], bins=100, density=True)
                fig_ax[0].set_yscale('log')
                if comp[0] == 'u':
                    fig_ax[0].set_xlabel(f"Variance of {comp} [px]")
                else:
                    fig_ax[0].set_xlabel(f"Variance of {comp} [-]")
                fig_ax[0].set_ylabel("Frequency [%]")
                fig_ax[0].grid(visible=True)
                # map
                fig_ax[1].imshow(img_ref, alpha=0.8)
                tmp = np.zeros(img_ref.shape) + np.nan
                tmp[loc_roi] = stats[f"var_{comp}"]
                fig_im = fig_ax[1].imshow(tmp)
                fig_ax[1].set_xlabel("position along $e_1$ [px]")
                fig_ax[1].set_ylabel("position along $e_2$ [px]")
                divider = make_axes_locatable(fig_ax[1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = plt.colorbar(fig_im, cax=cax)
                if comp[0] == 'u':
                    cbar.set_label(f"Variance of {comp} [px]")
                else:
                    cbar.set_label(f"Variance of {comp} [-]")
                plt.show(block=False)
                plt.tight_layout()

        if mylsa.options['verbose']:
            print("       ┌───────────────────────────────────────────────┐\n",
                  "      │  Considering these 'averaged' reference phase │\n",
                  "      │ modulations, expected equivalent stds will be │\n",
                  f"      │        .std(u1) = {stats['std_eq_u1']:.2E} [px]               │\n",
                  f"      │        .std(u2) = {stats['std_eq_u2']:.2E} [px]               │\n",
                  f"      │        .std(eps_11) = {stats['std_eq_eps_11']:.2E} [-]            │\n",
                  f"      │        .std(eps_12) = {stats['std_eq_eps_12']:.2E} [-]            │\n",
                  f"      │        .std(eps_22) = {stats['std_eq_eps_22']:.2E} [-]            │\n",
                  "      └───────────────────────────────────────────────┘")

        return mylsa, phi_ref_av, img_ref, kernel, stats

    @staticmethod
    def load(name):
        """ Method that loads a back-up class data file using the pickles format.
        filename is the name/path used to define the write down the data."""
        assert isinstance(name, str)
        if name[-4:] == '.pkl':
            with open(name, 'rb') as file:
                data = pickle.load(file)
            tmp = OpenLSA(vec_k=data['vec_k'], roi=data['roi'],
                          pt_2_follow=data['pt_2_follow'],
                          template=data['template'],
                          roi_75percent=data['roi_75percent'],
                          display=data['display'], verbose=data['verbose'])
            return tmp
        print('Error - unknown extension')
        return None


def assert_point(point):
    """ check assertion for point """
    assert isinstance(point, (NoneType, np.ndarray))
    if isinstance(point, np.ndarray):
        assert point.shape == (2,)
