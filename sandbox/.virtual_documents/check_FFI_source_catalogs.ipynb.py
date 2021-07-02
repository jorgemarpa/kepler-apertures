import numpy as np
import pandas as pd
from astropy.io import fits
import glob
from tqdm.auto import tqdm


depth_odd_mean  = 0.00503922
depth_odd_err   = 5.58157271e-04
# depth_even_mean = 0.00509802
depth_even_mean = 0.00504802
depth_even_err  = 5.71683118e-04

depth_mean = (depth_odd_mean + depth_even_mean)/2

weighted_dif = (depth_odd_mean - depth_even_mean)/(depth_odd_err**2 + depth_even_err**2)**0.5

s = (depth_odd_mean - depth_mean)**2 / depth_odd_err ** 2 + (depth_even_mean - depth_mean)**2 / depth_even_err ** 2


weighted_dif, s


(depth_odd_mean - depth_even_mean ) * 1e6


mjd_done = [
    55307.51459974,
    55399.03668604,
    55492.78603754,
    55585.55556804,
    55677.42404264,
    55769.45696544,
    55864.77969804,
    55958.40644554,
    56047.49693394,
    56137.50692204,
    56236.81420744,
    56330.54311544,
    56390.47480454,
]


fnames = glob.glob("../data/fits/ffi/*.fits")


mjds = []
for f in fnames:
    hdr = fits.open(f)[10].header
    mjds.append(float(hdr["MJDSTART"]))
    # if mjds[-1] in mjd_done:
    print(f.split("/")[-1].split("_")[0], mjds[-1])


len(mjds)


ch = 84
missing = []
for ch in range(1,85):
    if ch in [5, 6, 7, 8]:
        continue
    fnames_csv = glob.glob("../data/catalogs/ffi/source_catalog/channel_get_ipython().run_line_magic("i_*"", " % (ch))")

    cat_mjds = []

    for f in fnames_csv:
        aux = f.split("/")[-1].split("_")[-1][:-4]
        cat_mjds.append(float(aux))
    cat_mjds = sorted(cat_mjds)

    for mjd, f in zip(mjds, fnames):
        if mjd in cat_mjds:
            continue
        missing.append([f.split("/")[-1].split("_")[0], ch])
        print(ch, mjd, f.split("/")[-1].split("_")[0])


mjds = np.sort(mjds)
mjds.shape


df_flux = []
df_flux_err = []
df_coord = []
for mjd in tqdm(mjds):
    names = np.sort(glob.glob(
        "../data/catalogs/ffi/source_catalog/channel_*_source_catalog_mjd_get_ipython().run_line_magic("s.csv"", " % str(mjd)))")
    full_cat = []
    for f in names:
        cat = pd.read_csv(f)
        if "DR2" in cat.Gaia_source_id[0]:
            print(f)
        cat["Gaia_source_id"] = cat["Gaia_source_id"].map(lambda x: x.replace("DR2", "EDR3"))
        cat = cat.set_index('Gaia_source_id').drop('Unnamed: 0', axis=1)
        full_cat.append(cat)
    full_cat = pd.concat(full_cat)
    df_coord.append(full_cat.loc[:, ["RA", "DEC"]])
    df_flux.append(full_cat.loc[:, "Flux"])
    df_flux_err.append(full_cat.loc[:, "Flux_err"])
    # break


df_flux_all = pd.concat(df_flux, axis=1, keys=mjds)
df_flux_err_all = pd.concat(df_flux_err, axis=1, keys=mjds)
df_coord_all = pd.concat(df_coord, axis=0)
df_coord_all = df_coord_all[~df_coord_all.index.duplicated(keep="first")]


df_coord_all = df_coord_all.sort_index()
df_flux_all = df_flux_all.loc[df_coord_all.index]
df_flux_err_all = df_flux_err_all.loc[df_coord_all.index]


df_coord_all.shape, df_flux_all.shape, df_flux_err_all.shape


df_flux_all_nonan = df_flux_all.dropna(axis=0, how="any")
df_flux_err_all_nonan = df_flux_err_all.loc[df_flux_all_nonan.index]
df_coord_all_nonan = df_coord_all.loc[df_flux_all_nonan.index]


df_flux_all_nonan.shape, df_flux_err_all_nonan.shape, df_coord_all_nonan.shape


df_flux_all.dropna(axis=0, how="any", thresh=52).shape


df_flux_all.iloc[0]


df_flux_all_nonan.to_csv("../data/catalogs/ffi/ffi_sources_flux_01.csv")
df_flux_err_all_nonan.to_csv("../data/catalogs/ffi/ffi_sources_flux_err_01.csv")
df_coord_all_nonan.to_csv("../data/catalogs/ffi/ffi_sources_coord_01.csv")


import matplotlib.pyplot as plt


plt.scatter(df_coord_all_nonan.RA, df_coord_all_nonan.DEC, marker=".");


for k in range(0, 1000, 10):
    plt.errorbar(df_flux_all_nonan.columns, df_flux_all_nonan.iloc[k], yerr=df_flux_err_all_nonan.iloc[k])
# plt.yscale("log")
plt.show()


df_flux_all_nonan.iloc[0]


np.save("../data/catalogs/ffi/ffi_sources_coord.npy", df_coord_all.reset_index().to_numpy())


np.save("../data/catalogs/ffi/ffi_sources_flux.npy", df_flux_all.reset_index().to_numpy())


np.save("../data/catalogs/ffi/ffi_sources_flux_err.npy", df_flux_err_all.reset_index().to_numpy())
