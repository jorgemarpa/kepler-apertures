import os, glob
import numpy as np
import pandas as pd
import lightkurve as lk
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from tqdm.auto import tqdm


project_name = "kbonus-apexba"
channel = 27
quarter = 7

ap_names = ["O", 1, 2, 3, 4, 5, 6, 7]
ap_percut = ["op", 0 , 15, 30, 45, 60, 75, 90]


quarters = np.arange(5, 18)
channels = np.delete(np.arange(1, 85, 1), [4,5,6,7])
quarters, channels


cat = Table.read("../../EXBA_LCFs/data/hlsp_kbonus-apexba_kepler_v1.0_cat.fits")#.to_pandas().set_index("Gaia_designation")


for ch in tqdm(channels):
    files = np.sort(glob.glob("../data/export/get_ipython().run_line_magic("i/%02i/lc_q%i_ch%02i_*.fits.gz"", " % (quarter, ch, quarter, ch)))")
    for i, f in tqdm(enumerate(files), total=len(files), leave=False):
        hdu = fits.open(f)
        # update header cards
        phdr = hdu[0].header
        designation = phdr["OBJECT"]

        phdr.set("DOI", "", "Digital Object Identifier for the HLSP data collection", before="ORIGIN")
        phdr.set("HLSPID", "", "The identifier (acronym) for this HLSP collection", before="ORIGIN")
        phdr.set("HLSPLEAD", "Jorge Martinez-Palomera", "Full name of HLSP project lead", before="ORIGIN")
        phdr.set("HLSPVER", "1.0", "HLSP version", before="ORIGIN")
        phdr.set("LICENSE", "CC BY 4.0", "", before="ORIGIN")
        phdr.set("LICENURL", "https://creativecommons.org/licenses/by/4.0/", "", before="ORIGIN")

        phdr.set("MJD-BEG", hdu[1].data["time"][0], "Observation start time [MJD]", before="RADESYS")
        phdr.set("MJD-END", hdu[1].data["time"][-1], "Observation end time [MJD]", before="RADESYS")
        phdr.set("MJD-MED", hdu[1].data["time"][hdu[1].data["time"].shape[0]//2], 
                 "Observation mid time [MJD]", before="RADESYS")
        phdr.set("XPOSURE", 1625, "Duration of exposure [s]", before="RADESYS")

        phdr.set("OBSERVAT", "Kepler", "Observatory", before="TELESCOP")
        phdr.set("FILTER", "OPTICAL", "", after="INSTRUME")
        phdr.set("origin", "kepler-apertures", "Program used for photometry")
        phdr.set("creator", phdr["CREATOR"], "Program used to create file")
        phdr.set("PROCVER", "2.0.10dev", "Version of creator program")
        phdr.set("version", "0.1.0dev", "Version of origin program")
        phdr.set("aperture", "FFI-PRF aperture", "Method for aperture creation")
        # kepler cards:
        phdr.set("module", phdr["MODULE"], "", after="SEASON")
        phdr.set("output", phdr["output"], "", after="MODULE")
        phdr.set("channel", phdr["CHANNEL"], "", after="OUTPUT")
        phdr.set("quarter", phdr["QUARTER"], "", after="SEASON")
        phdr.set("INVESTID", phdr["CAMPAIGN"], "Investigation ID", after="MISSION")
        phdr.set("KEPLERID", cat[cat["Gaia_designation"] == designation]["KIC"][0] if type(cat[cat["Gaia_designation"] == designation]["KIC"][0]) == np.int32 else "", 
                 "Unique Kepler target identifier (KIC)", after="OBJECT")
        # gaia cards
        phdr.set("gaiaid", designation, "Gaia ID of the object", before="PMRA")
        pmra = phdr["PMRA"]
        phdr.set("pmra", pmra, "Gaia RA proper motion")
        pmdec = phdr["PMDEC"]
        phdr.set("pmdec", pmdec, "Gaia Dec proper motion")
        parallax = phdr["PARALLAX"]
        phdr.set("parallax", parallax, "Gaia parallax")
        gmag = phdr["GMAG"]
        phdr.set("gmag", gmag, "Gaia G magnitude")
        bpmag = phdr["BPMAG"]
        phdr.set("bpmag", bpmag, "Gaia BP magnitude")
        rpmag = phdr["RPMAG"]
        phdr.set("rpmag", rpmag, "Gaia RP magnitude")
        # org cards
        tpf_orgn = phdr["TPF_ORGN"]
        phdr.set("tpf_orgn", tpf_orgn, "TPF name of origin")
        kid_orgn = phdr["KID_ORGN"]
        phdr.set("kid_orgn", kid_orgn, "Kepler ID of file of origin")
        # location cards
        row = phdr["ROW"]
        phdr.set("row", row, "Object row location in TPF_ORGN")
        column = phdr["COLUMN"]
        phdr.set("column", column, "Object column location in TPF_ORGN")
        # aperture metrics
        for k, ap in enumerate(ap_names):
            if ap == "O":
                try:
                    per = phdr["PERCUTget_ipython().run_line_magic("s"", " % str(ap)]")
                except KeyError:
                    per = 75
            else:
                per = ap_percut[k]
            FLFRSAP = phdr["FLFRSAPget_ipython().run_line_magic("s"", " % str(ap)]")
            CRWDSAP = phdr["CRWDSAPget_ipython().run_line_magic("s"", " % str(ap)]")
            phdr.set("PERCUTget_ipython().run_line_magic("s"", " % str(k), int(per), ")
                     "Flux completeness metric for aperture get_ipython().run_line_magic("s"", " % (\"optimal\" if k==0 else str(k)),")
                    before="FLFRSAPget_ipython().run_line_magic("s"", " % str(k) if k>0 else None)")
            phdr.set("FLFRSAPget_ipython().run_line_magic("s"", " % str(k), np.round(FLFRSAP, decimals=4), ")
                     "Flux completeness metric for aperture get_ipython().run_line_magic("s"", " % (\"optimal\" if k==0 else str(k)))")
            phdr.set("CRWDSAPget_ipython().run_line_magic("s"", " % str(k), np.round(CRWDSAP, decimals=4), ")
                     "Flux crowding metric for aperture get_ipython().run_line_magic("s"", " % (\"optimal\" if k==0 else str(k)))")
        try:
            del phdr["PERCUTO"], phdr["FLFRSAPO"], phdr["CRWDSAPO"]
        except KeyError:
            del phdr["FLFRSAPO"], phdr["CRWDSAPO"]

        # hlsp keywords


        # remove SAP_QUALITY header card
        # del hdu[1].header["TTYPE20"], hdu[1].header["TFORM20"]

        # remove SAP_QUALITY from bintable
        new_table = Table(hdu[1].data)
        # new_table.remove_column("SAP_QUALITY")

        columns = ["TIME", "CADENCENO", "FLUX", "FLUX1", "FLUX2", "FLUX3", "FLUX4", "FLUX5", "FLUX6", "FLUX7", 
                   "FLUX_ERR", "FLUX_ERR1", "FLUX_ERR2", "FLUX_ERR3", "FLUX_ERR4", "FLUX_ERR5", "FLUX_ERR6", "FLUX_ERR7",
                   "QUALITY", "SAP_QUALITY"]

        for col in columns:
            if col.startswith("FLUX"):
                new_table[col].unit = 'e-/s'
            if col == "TIME":
                new_table[col].unit = 'jd'

        hdu[1] = fits.BinTableHDU(new_table[columns], name="LIGHTCURVE")

        # update extname for aperture mask
        hdu[2].header["EXTNAME"] = "APERTURE_OPT"
        for k in range(1, 8):
            hdu[k+2].header["EXTNAME"] = "APERTURE_get_ipython().run_line_magic("i"", " % k")

        # new dir tree
        out_dir = "../../EXBA_LCFs/data/qget_ipython().run_line_magic("02i/ch%02i"", " % (quarter, ch)")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        # new file name
        fname = "hlsp_get_ipython().run_line_magic("s_kepler_%s-q%02i_v%s_lc.fits.gz"", " % (")
                    project_name,
                    designation.replace(" ", "-"),
                    quarter,
                    "1.0",
                )
        hdu.writeto("get_ipython().run_line_magic("s/%s"", " % (out_dir, fname), overwrite=True, checksum=True)")
        # break


# out_dir = "../../EXBA_LCFs/data/qget_ipython().run_line_magic("02i/ch%02i"", " % (quarter, channel)")
# fname = "get_ipython().run_line_magic("s/hlsp_exba_kepler_Gaia-EDR3-2101136277956081920-q05_v1.0_lc.fits.gz"", " % out_dir")

test_hdu = fits.open("get_ipython().run_line_magic("s/%s"", " % (out_dir, fname))")


test_hdu.info()


test_hdu[0].header


test_hdu[1].header


test_hdu[3].header


lc = lk.LightCurve.read(fname, format="kepler")


lc.plot()


lck16 = lk.search_lightcurve('Kepler-16', mission='Kepler', 
                                 quarter=12, radius=1000, limit=100, 
                                 cadence='long').download_all(quality_bitmask=None)


lck16[0].meta


quarters = np.arange(5, 18)
channels = np.delete(np.arange(1, 85, 1), [4,5,6,7])
quarters, channels


del_kw = ["PMRA",
"PMDEC",
"PMTOTAL",
"PARALLAX",
"GLON",
"GLAT",
"GMAG",
"RMAG",
"IMAG",
"ZMAG",
"D51MAG",
"JMAG",
"HMAG",
"KMAG",
"KEPMAG",
"GRCOLOR",
"JKCOLOR",
"GKCOLOR",
"TEFF",
"LOGG",
"FEH",
"EBMINUSV",
"AV",
"RADIUS",
"TMINDEX",
"SCPID"]

for q in tqdm(quarters):
    for ch in channels:
        path = "../data/export/get_ipython().run_line_magic("i/%02i/image_q%i_ch%02i_gaiadr3.fits.gz"", " % (q, ch, q, ch)")
        hdu = fits.open(path)
        hdu[1].header["EXTNAME"] = "EXBA_mask"
        hdu[2].header["EXTNAME"] = "EXBA_sources"
        for k in del_kw:
            del hdu[0].header[k]
        # new dir tree
        out_dir = "../../EXBA_LCFs/data/qget_ipython().run_line_magic("02i/ch%02i"", " % (q, ch)")
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        # new file name
        fname = "hlsp_get_ipython().run_line_magic("s_kepler_%s-ch%02i-q%02i_v%s_sup.fits.gz"", " % (")
                    project_name,
                    "exba-mask",
                    ch,
                    q,
                    "1.0",
                )
        # print("get_ipython().run_line_magic("s/%s"", " % (out_dir, fname))")
        hdu.writeto("get_ipython().run_line_magic("s/%s"", " % (out_dir, fname), overwrite=True, checksum=True)")
    #     break
    # break


hdu.info()


hdu[0].header


hdu[1].header


hdu[2].header


cat = Table.read("../../EXBA_LCFs/data/hlsp_kbonus-apexba_kepler_v1.0_cat.fits", format="fits")


cat


hdu = fits.open("../../EXBA_LCFs/data/hlsp_kbonus-apexba_kepler_v1.0_cat.fits")


hdu.info()


phdr = hdu[0].header
phdr


hdu[1].header.set("TTYPE1", "Gaia_designation")


hdu[1].header


hdu.writeto("../../EXBA_LCFs/data/hlsp_kbonus-apexba_kepler_v1.0_cat.fits", overwrite=True, checksum=True)



