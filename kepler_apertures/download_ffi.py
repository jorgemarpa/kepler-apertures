import glob, os
import wget
from tqdm import tqdm
from . import log

filenames = {
    10: ["kplr2011208112727_ffi-cal.fits", "kplr2011240181752_ffi-cal.fits"],
    11: ["kplr2011303191211_ffi-cal.fits", "kplr2011334181008_ffi-cal.fits"],
    12: ["kplr2012032101442_ffi-cal.fits", "kplr2012060123308_ffi-cal.fits"],
    13: ["kplr2012121122500_ffi-cal.fits", "kplr2012151105138_ffi-cal.fits"],
    14: ["kplr2012211123923_ffi-cal.fits", "kplr2012242195726_ffi-cal.fits"],
    15: ["kplr2012310200152_ffi-cal.fits", "kplr2012341215621_ffi-cal.fits"],
    16: ["kplr2013038133130_ffi-cal.fits", "kplr2013065115251_ffi-cal.fits"],
    5: ["kplr2010111125026_ffi-cal.fits", "kplr2010140101631_ffi-cal.fits"],
    6: ["kplr2010203012215_ffi-cal.fits", "kplr2010234192745_ffi-cal.fits"],
    7: ["kplr2010296192119_ffi-cal.fits", "kplr2010326181728_ffi-cal.fits"],
    8: ["kplr2011024134926_ffi-cal.fits", "kplr2011053174401_ffi-cal.fits"],
    9: ["kplr2011116104002_ffi-cal.fits", "kplr2011145152723_ffi-cal.fits"],
}
url = "https://archive.stsci.edu/missions/kepler/ffi"


def download_ffi(quarter: int = 5):
    """
    Download FFI fits file to a dedicated quarter directory
    Parameters
    ----------
    quarter : int
        Number of the quarter to download
    """
    try:
        fnames = filenames[quarter]
    except KeyError:
        log.error("Input Quarter not in FFI catalog")

    if not os.path.isdir("../data/fits/%s" % (str(quarter))):
        os.mkdir("../data/fits/%s" % (str(quarter)))
    for i, fn in enumerate(fnames):
        out = "../data/fits/%i/%s" % (quarter, fn)

        wget.download("%s/%s" % (url, fn), out=out)

    return
