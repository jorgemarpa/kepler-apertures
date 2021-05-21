# Kepler Apertures

Tools to create aperture mask for Kepler sources using PRF models build from Kepler's
Full Frame Images.

# PRF models

First we create PRF models using Kepler's FFI which contains ~10k-20k Gaia DR2 sources per Kepler's channel.

The following figure shows the PRF models in the focal plane. Channels at the border shows PRFs with very distorted shapes, while in the center these are round and smooth.

![PRF Models](https://github.com/jorgemarpa/kepler-apertures/blob/paper-release/figures/focal_plane_prf_model.pdf)

Later this PRF models are used to compute optimal apertures to perform photometry.

# Kepler's EXBA masks

The EXBA masks are custom apertures observed by Kepler's first mission, they cover relatively dark regions of the Kepler field and were observed continuously between quarters 4 and 17. The scientific motivation to collect these data was to obtain an unbiased characterization of the eclipsing binary occurrence fraction in the Kepler field.

Here an example of the full EXBA mask observed in quarter 5 with channel 48

![exba_ch48](https://github.com/jorgemarpa/kepler-apertures/blob/paper-release/figures/EXBA_img_q5_ch48.pdf)

# Dependencies
* astropy
* scipy
* matplotlib
* photutils
