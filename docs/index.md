# Kepler-Apertures

*Aperture Photometry with Kepler's PRFs*

Tools to create aperture mask for Kepler sources using PRF models build from Kepler's
Full Frame Images.

# Installation

```
pip install kepler-apertures
```

# What's happening?

`kepler-apertures` does the following:

* Creates PRF models from Kepler's FFIs.
* Uses PRF models to compute photometric apertures and flux metrics.
* Creates SAP light curves for more than 9,300 Gaia sources observed in Kepler's EXBA masks.

Also, `kepler-apertures`, allows you to compute your own PRF models and use them to
find new apertures mask.


# What does it look like?

## PRF models

We created PRF models using Kepler's FFI which contains ~10k Gaia EDR3 sources per Kepler's channel.

The following figure shows the PRF models in the focal plane. Channels at the border show PRFs with very distorted shapes, while in the center these are round and smooth.

![PRF Models](https://github.com/jorgemarpa/kepler-apertures/blob/paper-release/docs/focal_plane_prf_model.png)

Later this PRF models are used to compute apertures photometry.

## Kepler's EXBA masks

The EXBA masks are custom apertures observed by Kepler's first mission, they cover relatively dark regions of the Kepler field and were observed continuously between quarters 4 and 17. The scientific motivation to collect these data was to obtain an unbiased characterization of the eclipsing binary occurrence fraction in the Kepler field.

Here an example of the full EXBA mask observed in quarter 5 with channel 48

![exba_ch48](https://github.com/jorgemarpa/kepler-apertures/blob/paper-release/docs/EXBA_img_q5_ch48.png)

## EXBA Light Curves

Here two examples of light curves produced with `kepler-apertures` of sources found in the
EXBA mask

![ebs](https://github.com/jorgemarpa/kepler-apertures/blob/paper-release/docs/ebs.png)
![cand](https://github.com/jorgemarpa/kepler-apertures/blob/paper-release/docs/g304.png)

# What can I use it on?



# Example use

For how to use `kepler-apertures` see the notebook tutorials.



Funding for this project is provided by NASA ROSES grant number 80NSSC20K0874.
