# Kepler Apertures

[![DOI](https://zenodo.org/badge/345841081.svg)](https://zenodo.org/badge/latestdoi/345841081)


Take me to the [documentation](https://jorgemarpa.github.io/kepler-apertures/).
Paper coming soon!

Tools to create aperture mask for Kepler sources using PRF models build from Kepler's
Full Frame Images.

Inspired by [`psfmachine`](https://github.com/SSDataLab/psfmachine).

# Instalation

```
pip install kepler-apertures
```

# PRF models

First we create PRF models using Kepler's FFI which contains ~10k Gaia EDR3 sources per Kepler's channel.

The following figure shows the PRF models in the focal plane. Channels at the border shows PRFs with very distorted shapes, while in the center these are round and smooth.

![PRF Models](https://github.com/jorgemarpa/kepler-apertures/blob/main/docs/focal_plane_prf_model.png)

Later this PRF models are used to compute apertures photometry.

# Kepler's EXBA masks

The EXBA masks are custom apertures observed by Kepler's first mission, they cover relatively dark regions of the Kepler field and were observed continuously between quarters 4 and 17. The scientific motivation to collect these data was to obtain an unbiased characterization of the eclipsing binary occurrence fraction in the Kepler field.

Here an example of the full EXBA mask observed in quarter 5 with channel 48

![exba_ch48](https://github.com/jorgemarpa/kepler-apertures/blob/main/docs/EXBA_img_q5_ch48.png)

# Light Curve examples

Light curve examples created with `kepler-apertures` for sources detected in the EXBA masks. These are 4 Eclipsing Binaries observed by Kepler.

![EBs](https://github.com/jorgemarpa/kepler-apertures/blob/main/docs/ebs.png)

# Usage

## PRF models
To create a new PRF model using a single channel (44) from a Kepler's FFI:

```python
import kepler_apertures as ka
psf = ka.KeplerFFI(quarter=5, channel=44, plot=True, save=False)

# plot the channel image if wanted
_ = psf.plot_image(sources=False)

# build a PRF shape models
psd.build_prf_model()

# save model to disk
psf.save_model(path="file_path")
```

See the full [tutorial](https://jorgemarpa.github.io/kepler-apertures/tutorials/create_PRF_tutorial/) for more details.

## Create Light Curves
To use the PRF model to create aperture masks on EXBA sources:

```python
import kepler_apertures as ka

# initialize an EXBA object and query Gaia EDR3 to get sources
exba = ka.EXBAMachine(quarter=5, channel=44, magnitude_limit=20, gaia_dr=3)

# plot the image if wanted
ax = exba.plot_image()

# load PRF shape from disk (this uses precomputed models published with the repo)
kprf = ka.KeplerPRF.load_from_file(quarter=5, channel=44)
# evaluate PRF shape on the  EXBA sources
psf_exba = kprf.evaluate_PSF(exba.dx, exba.dy)

# compute optimized aperture and flux metrics
ap_mask, crwd, cmplt, cut = kprf.optimize_aperture(psf_exba, idx=1, target_complet=0.5, target_crowd=1.)

# create light curve using aperture mask
exba.create_lcs(ap_mask)
```

See the full [tutorial](https://jorgemarpa.github.io/kepler-apertures/tutorials/using_PRF_on_exba/) for more details.

# Dependencies
* numpy
* scipy
* astropy
* matplotlib
* photutils
* pandas
* tqdm
* patsy
* pyia
* lightkurve
