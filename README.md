# OpenLSA

OpenLSA is a Python Library developed for processing images of periodic patterns to retrieve the displacement that warps them.
Full-field displacement maps are thus deduced from a pair of images.
Designed for the Experimental Mechanics community, LSA is able to process optimized patterns in terms of metrological performance, such as the checkerboard one [1].

An illustration of a wooden specimen, subjected to a compression load, is detailed within the file "Script_example.py". An interested reader can refer to the LSA seminal paper [2].

[1] Increasing accuracy and precision of Digital Image Correlation through pattern optimization. Bomarito, G. F. and Hochhalter, J. D. and Ruggles, T. J. \& Cannon, A. H.; Optics and Lasers in Engineering; 2017 (doi: 10.1016/j.optlaseng.2016.11.005)

[2] Extracting displacement and strain fields from checkerboard images with the Localized Spectrum Analysis M. Grédiac, B. Blaysat \& F. Sur; Experimental Mechanics, 2019. (doi: 10.1007/s11340-018-00439-2)


## Getting started
OpenLSA sources are available at the [repository](https://pypi.org/project/openlsa/) or the library can be installed with pip:

```
pip install OpenLSA
```

## Recommendations
OpenLSA requires numerous Python libraries: numpy, scipy, scikit-image, opencv-python, pillow, matplotlib, and boto3.
All dependencies are correctly managed when using the pip installation.

## Usage 
A set of images and the minimal script example is provided in the [repository](https://pypi.org/project/openlsa/).
Nevertheless, the main steps for retrieving a displacement field from a reference ref_im.tif to a current cur_im.tif are given in what follows.


### Required libraries
```
import numpy as np
from PIL import Image
from openlsa import OpenLSA
```

### Loading images
```
img_0 = np.array(Image.open("ref_im.tif"))
img_t = np.array(Image.open("cur_im.tif"))
```
Images are loaded and formatted as numpy arrays.

### Initializing LSA & kernel
```
lsa = OpenLSA(img_0, verbose=True, display=True)
kernel = my_lsa.compute_kernel(std=my_lsa.pitch().max())
```
The ***lsa*** class constructor directly analyses the reference image to build up LSA parameters such as the two orthogonal wave vectors characterizing its pattern.
The ***verbose*** (respectively, ***display***) option is set to **True**, so the program, during execution, displays information in the terminal (respectively, separate figures).
The LSA kernel is also defined in the proposed script.
The smallest width is chosen here, with the standard deviation of the Gaussian kernel being set to the pattern period.

### Computing phases
```
phi_0, __ = my_lsa.compute_phases_mod(img_0, kernel)
phi_t, __ = my_lsa.compute_phases_mod(img_t, kernel)
```
The ***compute_phases_mod*** method is called to compute the phase modulations associated with each image.

### Solving the temporal unwraping
```
phi_t, __ = my_lsa.temporal_unwrap(img_0, img_t, phi_0, phi_t)
```
The pairing of the current phase modulations ***phi_t*** to ***phi_0*** is performed by following a specific point from the reference image to the current one.
Point selection is automatically defined.

### Computing of the displacement
```
u_xy = my_lsa.compute_displacement(phi_0, phi_t, min_iter=6)
```
Finally, a fixed point algorithm is called for computing the displacement field from both phase modulations ***phi_0*** and ***phi_t***.
Displacement ***u_xy*** is formatted as a complex number, its real (respectively, imaginary) part corresponding to the component in the row (respectively, column) direction.


## Roadmap
Short term developments will consist of:
- Extending the displacement calculation to take into account camera model.
- Adding the deconvolution algorithm.

## License
These python codes can be used for non-profit academic research only.
They are distributed under the terms of the GNU General Public License v3.

