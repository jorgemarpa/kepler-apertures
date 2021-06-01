from __future__ import absolute_import
import os
import logging

PACKAGEDIR = os.path.abspath(os.path.dirname(__file__))
DATAOUTDIR = "/".join(PACKAGEDIR.split("/")[:-2])

from .KeplerFFI import KeplerFFI
from .KeplerPRF import KeplerPRF
from .EXBAMachine import EXBAMachine
from .version import *

# Configure logging
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler())

__all__ = ["KeplerFFI", "KeplerPRF", "EXBAMachine"]
