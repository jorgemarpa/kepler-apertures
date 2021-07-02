import os, glob
import numpy as np
import lightkurve as lk
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from tqdm.auto import tqdm
import argparse

parser = argparse.ArgumentParser(description="AutoEncoder")
parser.add_argument(
    "--quarter",
    dest="quarter",
    type=int,
    default=None,
    help="Which quarter.",
)
parser.add_argument(
    "--channel",
    dest="channel",
    type=int,
    default=1,
    help="List of files to be downloaded",
)
args = parser.parse_args()

project_name = "kBonus-ApEXBA"
ap_names = ["O", 1, 2, 3, 4, 5, 6, 7]
ap_percut = ["op", 0, 15, 30, 45, 60, 75, 90]

columns = [
    "TIME",
    "CADENCENO",
    "FLUX",
    "FLUX1",
    "FLUX2",
    "FLUX3",
    "FLUX4",
    "FLUX5",
    "FLUX6",
    "FLUX7",
    "FLUX_ERR",
    "FLUX_ERR1",
    "FLUX_ERR2",
    "FLUX_ERR3",
    "FLUX_ERR4",
    "FLUX_ERR5",
    "FLUX_ERR6",
    "FLUX_ERR7",
    "QUALITY",
    "SAP_QUALITY",
]


def run_code():
    quarter = args.quarter
    channel = args.channel
    files = np.sort(
        glob.glob(
            "../data/export/%i/%02i/lc_q%i_ch%02i_*.fits.gz"
            % (quarter, channel, quarter, channel)
        )
    )
    for i, f in tqdm(enumerate(files), total=len(files)):
        hdu = fits.open(f)
        # update header cards
        phdr = hdu[0].header
        designation = phdr["OBJECT"]

        phdr.set(
            "DOI",
            "",
            "Digital Object Identifier for the HLSP data collection",
            before="ORIGIN",
        )
        phdr.set(
            "HLSPID",
            "",
            "The identifier (acronym) for this HLSP collection",
            before="ORIGIN",
        )
        phdr.set(
            "HLSPLEAD",
            "Jorge Martinez-Palomera",
            "Full name of HLSP project lead",
            before="ORIGIN",
        )
        phdr.set("HLSPVER", "1.0", "HLSP version", before="ORIGIN")
        phdr.set("LICENSE", "CC BY 4.0", "", before="ORIGIN")
        phdr.set(
            "LICENURL",
            "https://creativecommons.org/licenses/by/4.0/",
            "",
            before="ORIGIN",
        )

        phdr.set(
            "MJD-BEG",
            hdu[1].data["time"][0],
            "Observation start time [MJD]",
            before="RADESYS",
        )
        phdr.set(
            "MJD-END",
            hdu[1].data["time"][-1],
            "Observation end time [MJD]",
            before="RADESYS",
        )
        phdr.set(
            "MJD-MED",
            hdu[1].data["time"][hdu[1].data["time"].shape[0] // 2],
            "Observation mid time [MJD]",
            before="RADESYS",
        )
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
        phdr.set("campaign", phdr["CAMPAIGN"], "", after="MISSION")
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
                per = phdr["PERCUT%s" % str(ap)]
            else:
                per = ap_percut[k]
            FLFRSAP = phdr["FLFRSAP%s" % str(ap)]
            CRWDSAP = phdr["CRWDSAP%s" % str(ap)]
            phdr.set(
                "PERCUT%s" % str(k),
                int(per),
                "Flux completeness metric for aperture %s"
                % ("optimal" if k == 0 else str(k)),
                before="FLFRSAP%s" % str(k) if k > 0 else None,
            )
            phdr.set(
                "FLFRSAP%s" % str(k),
                np.round(FLFRSAP, decimals=4),
                "Flux completeness metric for aperture %s"
                % ("optimal" if k == 0 else str(k)),
            )
            phdr.set(
                "CRWDSAP%s" % str(k),
                np.round(CRWDSAP, decimals=4),
                "Flux crowding metric for aperture %s"
                % ("optimal" if k == 0 else str(k)),
            )
        del phdr["PERCUTO"], phdr["FLFRSAPO"], phdr["CRWDSAPO"]

        # remove SAP_QUALITY from bintable
        new_table = Table(hdu[1].data)
        # new_table.remove_column("SAP_QUALITY")

        for col in columns:
            if col.startswith("FLUX"):
                new_table[col].unit = "e-/s"
            if col == "TIME":
                new_table[col].unit = "jd"

        hdu[1] = fits.BinTableHDU(new_table[columns], name="LIGHTCURVE")

        # update extname for aperture mask
        hdu[2].header["EXTNAME"] = "APERTURE_OPT"
        for k in range(1, 8):
            hdu[k + 2].header["EXTNAME"] = "APERTURE_%i" % k

        # new dir tree
        out_dir = "../../EXBA_LCFs/data/q%02i/ch%02i" % (quarter, channel)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        # new file name
        fname = "hlsp_%s_kepler_%s-q%02i_v%s_lc.fits.gz" % (
            project_name,
            designation.replace(" ", "-"),
            quarter,
            "1.0",
        )
        hdu.writeto("%s/%s" % (out_dir, fname), overwrite=True, checksum=True)
        # break
    return


if __name__ == "__main__":
    print("Quarter %i Channel %i" % (args.quarter, args.channel))
    run_code()
    print("Done!")
