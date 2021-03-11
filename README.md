# Kepler Apertures

Tools to create aperture mask for Kepler sources using PSF models build from Kepler's
Full Frame Images.

# Results

First we create PSF models using Kepler's FFI which contains ~10k-20k Gaia DR2 sources per Kepler's channel.

The following figure shows the PSF models in the focal plane. Channels at the border shows PSFs with very distorted shapes, while in the center these are round and smooth.

![PSF Models](https://github.com/jorgemarpa/kepler-apertures/blob/main/docs/focal_plane_psf_model.png)

Later this PSF models are used to compute optimal apertures to perform photometry. 

# Dependencies
* astropy
* scipy
* matplotlib
* photutils
