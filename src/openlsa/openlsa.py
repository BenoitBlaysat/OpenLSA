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

# %% Required Libraries
import os
import glob
import io
import pickle
import boto3
import numpy as np
from numpy import ma
from scipy.ndimage import map_coordinates
from scipy import ndimage
from PIL import Image
from skimage.restoration import unwrap_phase
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
        return 1/np.abs(self.vec_k[comp])

    def angle(self, comp=None, deg=False):
        """  Method that returns the list of the angles [rad] through which the periodic pattern is
        encoded. If a component (comp) is specified, output is reduced to it.  if deg is true,
        output is given in degrees"""
        if comp is None:
            return np.angle(self.vec_k, deg=deg)
        return np.angle(self.vec_k[comp], deg=deg)

    def vec_dir(self, comp=None):
        """  Method that returns the list of the uni vector [-] through which the periodic pattern
        is encoded. If a component (comp) is specified, output is reduced to it."""
        if comp is None:
            return np.exp(1j*np.angle(self.vec_k))
        return np.exp(1j*np.angle(self.vec_k[comp]))

    def px_z(self, roi=False):
        """  Method that returns pixel coordinates, formated as complex."""
        if roi:
            return self.__px_z[self.roi]
        return self.__px_z

    # %% Kernel building
    def compute_kernel(self, std=None):
        """ Method that computes LSA gaussian normalized kernel, of standard diviation "std"."""
        if std is None:
            std = self.pitch().max()
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
        w_f_r = cv2.filter2D(img*np.cos(-2*np.pi*scal_prod(vec_k, self.__px_z)), -1, kernel)
        w_f_i = cv2.filter2D(img*np.sin(-2*np.pi*scal_prod(vec_k, self.__px_z)), -1, kernel)
        w_f = w_f_r + 1j*w_f_i
        return np.abs(w_f), Phase(np.angle(w_f), vec_k)

    def compute_phases_mod(self, img, kernel=None, roi_coef=0.2, unwrap=True):
        """LSA coreL return phases and magnitudes of an image for a list of wave vectors
        kernel is the kernel used for LSA
        roi_coef defines the thresshold used for defining the region of interest
        unwrap is an option for returning wrapped phase modulations."""

        if self.options['verbose']:
            print('\n Computing the phase modulations\n',
                  '-------------------------------')
        if kernel is None:
            self.compute_kernel()
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
        self.check_temp_unwrap(img1, point1=point1)
        if point2 is None:
            point2, uinit = self.rough_point2point(img1, img2, dis_init=uinit)
        return self.jump_correction(phi_1, phi_2, point2), uinit

    def check_temp_unwrap(self, img, point1=None):
        """ Method that checks if the features neede for the temporal unwrap have been
        initialized. If not, it runs the methods to make it."""
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
                  f"-> [{point2[0][1]:0.2f}, {point2[0][1]:0.2f}] "
                  f"-> SSD = {ssd[0]:0.2f}\n"
                  f"         Pattern Matching  = [{point1[0]:d}, {point1[1]:d}] "
                  f"-> [{point2[1][1]:0.2f}, {point2[1][1]:0.2f}] "
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
        if filename.split(".")[-1] == '.pkl':
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
                                       roi_coef=0.2, kernel_width=0, **kwargs):
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
        if kernel_width == 0:
            kernel_width = mylsa.pitch().max()
        kernel = mylsa.compute_kernel(std=kernel_width)
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

        return mylsa, phi_ref_av, kernel, stats

    @staticmethod
    def load(name):
        """ Method that loads a back-up class data file using the pickles format.
        filename is the name/path used to define the write down the data."""
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

###############################################################################


# %% Class phase and phases
class Phase():
    """ Phase is a class that helps manipulate phase maps. This class has three attributes:\n
        vec_k: wave vector along which phase has been extracted\n
        data: extracted phase (numpy array)\n
        shape: shape of the numpy array collecting the phase modulation"""

    def __init__(self, phase, vec_k):
        self.vec_k = vec_k
        self.data = phase
        self.shape = phase.shape

    def __add__(self, other):
        """Note that + returns a numpy array, not a Phase class"""
        return self.data + other.data

    def __sub__(self, other):
        """Note that + returns a numpy array, not a Phase class"""
        return self.data - other.data

    def copy(self):
        """ Method that copies a given Phase class"""
        return Phase(self.data.copy(), self.vec_k.copy())

    def unwrap(self, roi=None):
        """ Method that unwraps the phase map. A region of interest can be provided to reduce
        computing cost"""
        if roi is None:
            roi = np.ones(self.data.shape, dtype='bool')
        phi = ma.masked_array(self.data, roi)
        self.data = np.array(unwrap_phase(phi).filled(0))

    def interp(self, point_yx_input, order=1):
        """ Method that interpolates the phase map."""
        if isinstance(np.array(point_yx_input).ravel().tolist()[0], complex):
            point_yx = [point_yx_input.imag, point_yx_input.real]
        else:
            point_yx = point_yx_input
        return Phase(map_coordinates(self.data,
                                     np.array(point_yx).reshape([2, -1]),
                                     order=order).ravel(), self.vec_k)

    def add_corr(self, corr):
        """ Method that add a correction to the phase map."""
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
        return cax.imshow(self.data, **kwargs)

    def save(self, filename):
        """ Method that writes into a .npz file a given Phase class"""
        np.savez_compressed(filename.split(".")[0] + 'npz',
                            vec_k=self.vec_k,
                            data=self.data)

    @staticmethod
    def load(filename):
        """ Method that reads a .npz file and creates a Phase class"""
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
        self.phases = list_of_phases
        self.shape = list_of_phases[0].shape

    def __len__(self,):
        return len(self.phases)

    def __add__(self, other):
        """Note that + returns a list of numpy array, not a Phases class"""
        return [self.phases[i] + other.phases[i] for i in range(len(self))]

    def __sub__(self, other):
        """Note that + returns a list of numpy array, not a Phases class"""
        return [self.phases[i] - other.phases[i] for i in range(len(self))]

    def __getitem__(self, index):
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
        np.savez_compressed(filename.split(".")[0] + 'npz',
                            vec_k=tmp_vec_k,
                            data=tmp_data)

    @staticmethod
    def load(filename):
        """ Method that reads a .npz file and creates a Phases class"""
        with np.load(filename.split(".")[0] + 'npz') as data:
            tmp_vec_k = data['vec_k']
            tmp_data = data['data']
        tmp = Phases([Phase(tmp_data[:, :, i], tmp_vec_k[:, i].tolist()[0])
                      for i in range(tmp_vec_k.shape[1])])
        return tmp

###############################################################################


# %% Usefull functions
def scal_prod(input_1, input_2):
    """ Scalar product"""
    return input_1.real*input_2.real + input_1.imag*input_2.imag


def a01_2_axy(vec01, a01):
    """ Compute given vector a01 (expressed in basis vec01) in basis (ex, ey)"""
    return a01[:, 0]*vec01[0] + a01[:, 1]*vec01[1]


def axy_2_a01(vec01, axy):
    """ Compute given vector axy (ex, ey) in basis vec01"""
    op_00, op_01, op_10, op_11 = vec01[0].real, vec01[1].real, vec01[0].imag, vec01[1].imag
    det_op = op_00*op_11-op_01*op_10
    iop_00, iop_01, iop_10, iop_11 = op_11/det_op, -op_01/det_op, -op_10/det_op, op_00/det_op
    return np.array([axy.real*iop_00 + axy.imag*iop_01,
                     axy.real*iop_10 + axy.imag*iop_11])


def compute_rbm(disp, coord_x, coord_y):
    """ Computing the RBM part of a displacement"""
    coord_x = (coord_x - coord_x.mean())/(2*(coord_x.max()-coord_x.min()))
    coord_y = (coord_y - coord_y.mean())/(2*(coord_y.max()-coord_y.min()))
    operator = np.array([[len(coord_x), 0, np.sum(coord_x)],
                         [0, len(coord_y), np.sum(coord_y)],
                         [np.sum(coord_x), np.sum(coord_y), np.sum(coord_x**2+coord_y**2)]])
    right_hand_member = np.array([np.sum(disp.real),
                                  np.sum(disp.imag),
                                  np.sum(coord_y*disp.real + coord_x*disp.imag)])
    dof = np.linalg.lstsq(operator, right_hand_member, rcond=None)[0]
    return dof[0] + 1j*dof[1] + dof[2]*(coord_y + 1j*coord_x)


def reject_outliers(data, bandwitch=3):
    """ Removing outliers"""
    return abs(data - np.nanmean(data)) < bandwitch * np.nanstd(data)

###############################################################################


# %% OpenCV functions for raw estimating of the displacement field
def estimate_u(img1, img2, filter_size=None, dis_init=None):
    """ This function estimates the displacement that warps image img1 into image img2 using the
    Dense Inverse Search optical flow algorithm from the OpenCV Python library """
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
    return np.uint8(img*2**(8-round(np.log2(img.max()))))


# %% function to open a s3 located list of images.
def provide_s3_path(s3_dictionary, im_extensions, im_pattern, verbose):
    """ This function reads the s3_dictionary to provide a list of paths to a set of images"""
    credentials_flag = False
    if 's3_access_key_id' in s3_dictionary.keys():
        if s3_dictionary['s3_access_key_id'] is not None:
            credentials_flag = True
    if 's3_access_key_id' in s3_dictionary.keys() and s3_dictionary['s3_path_2_im'] is not None:
        im_stack = s3_dictionary['s3_path_2_im']
    elif s3_dictionary['s3_path_2_folder'] is not None:
        if credentials_flag:
            s3_client = boto3.client('s3',
                                     aws_access_key_id=s3_dictionary['s3_access_key_id'],
                                     aws_secret_access_key=s3_dictionary['s3_secret_access_key'],
                                     aws_session_token=s3_dictionary['s3_session_token'],
                                     endpoint_url=s3_dictionary['s3_endpoint_url'])
        else:
            s3_client = boto3.client('s3', endpoint_url=s3_dictionary['s3_endpoint_url'])
        response = s3_client.list_objects_v2(Bucket=s3_dictionary['s3_bucket_name'],
                                             Prefix=s3_dictionary['s3_path_2_folder'])
        if 'Contents' in response:
            im_stack = [item['Key'] for item in response['Contents']
                        if item['Key'].lower().endswith(tuple(im_extensions))]
            im_stack = [item for item in im_stack if im_pattern in item]
        if verbose:
            print(f"      A path to a s3 folder is given: {len(im_stack):d} images are found.")
    else:
        print('Error: I do need a s3 path to images or at least to a folder')
    if credentials_flag:
        s3_resource = boto3.resource('s3',
                                     aws_access_key_id=s3_dictionary['s3_access_key_id'],
                                     aws_secret_access_key=s3_dictionary['s3_secret_access_key'],
                                     aws_session_token=s3_dictionary['s3_session_token'],
                                     endpoint_url=s3_dictionary['s3_endpoint_url'])
    else:
        s3_resource = boto3.resource('s3',
                                     endpoint_url=s3_dictionary['s3_endpoint_url'])
    return im_stack, s3_resource.Bucket(s3_dictionary['s3_bucket_name'])
