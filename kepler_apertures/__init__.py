from __future__ import absolute_import
import logging

# Configure logging
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())

from .KeplerFFI import KeplerPSF

__version__ = "0.1.0"
__all__ = ["KeplerPSF"]
