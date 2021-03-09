from __future__ import absolute_import
import logging
from .KeplerFFI import KeplerPSF

# Configure logging
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())

__version__ = "0.1.0"
__all__ = ["KeplerPSF"]
