"""Module to loading magnet states and convert magnet values."""
from __future__ import print_function
import os
from collections import OrderedDict
import numpy as np

from orbit.utils.xml import XmlDataAdaptor

from btfsim.util.default import Default
from btfsim.util import utils


class MagnetConverter(object):
    """Convert between gradient and current based on coefficients in file."""

    def __init__(self, coef_filename=None):
        self.coef_filename = coef_filename
        if self.coef_filename is None:
            default = Default()
            self.coef_filename = os.path.join(
                default.defaultdict["HOMEDIR"], default.defaultdict["MAG_COEFF"]
            )
        print("coef_filename:", self.coef_filename)
        self.coeff = utils.file_to_dict(self.coef_filename)

    def c2gl(self, quadname, scaledAI):
        """Convert current to gradient.

        quadname : str
            Quadrupole name (i.e., 'QH01').
        scaledAI : float
            Current setpoint (corresponds with scaled AI in IOC) [A].
        """
        scaledAI = float(scaledAI)
        try:
            A = float(self.coeff[quadname][0])
            B = float(self.coeff[quadname][1])
            GL = A * scaledAI + B * scaledAI**2
        except KeyError:
            print(
                "Do not know conversion factor for element {}; gradient value not assigned."
                .format(quadname)
            )
            GL = []
        return GL

    def gl2c(self, quadname, GL):
        """Convert gradient to current.

        quadname : str
            Quadrupole name (i.e., 'QH01').
        GL : float
            Integrated gradient [T].
        """
        GL = float(GL)
        try:
            A = float(self.coeff[quadname][0])
            B = float(self.coeff[quadname][1])
            if B == 0 and A == 0:  # handle case of 0 coefficients
                scaledAI = 0
            elif B == 0 and A != 0:  # avoid division by 0 for quads with 0 offset
                scaledAI = GL / A
            else:
                scaledAI = 0.5 * (A / B) * (-1 + np.sqrt(1 + 4 * GL * B / A**2))
        except KeyError:
            print(
                "Do not know conversion factor for element {}; current set to zero."
                .format(quadname)
            )
            scaledAI = 0
        return scaledAI

    def igrad2current(self, inputdict):
        """inputdict has key = magnet name, value = integrated gradient GL [T]."""
        outputdict = OrderedDict.fromkeys(self.coeff.keys(), [])
        for name in inputdict:
            try:
                outputdict[name] = self.gl2c(name, inputdict[name])
            except:
                print("Something went wrong on element {}.".format(name))
        return outputdict

    def current2igrad(self, inputdict):
        """inputdict has key = magnet name, value = current setpoint [A]."""
        outputdict = OrderedDict.fromkeys(self.coeff.keys(), [])
        for name in inputdict:
            try:
                outputdict[name] = self.c2gl(name, inputdict[name])
            except:
                print("Something went wrong on element {}.".format(name))
        return outputdict


# Functions to load settings from SCORE (.mstate file)
# ------------------------------------------------------------------------------
def load_quad_setpoint(filename):
    """Retrieve quadrupole setpoints [A] from .mstate file.

    Returns dictionary matching quad name with current setpoint.
    """
    state_da = XmlDataAdaptor.adaptorForFile(filename)
    thisstate = state_da.data_adaptors[0]
    setpointdict = OrderedDict()
    for item in thisstate.data_adaptors:
        pvname = item.getParam("setpoint_pv").split(":")
        psname = pvname[1].split("_")
        magname = psname[1]
        setpointdict[magname] = float(item.getParam("setpoint"))
    return setpointdict


def loadQuadReadback(filename):
    """Retrieve quadrupole readbacks [T] from .mstate file.

    Returns dictionary matching quad name with field readback.
    """
    state_da = XmlDataAdaptor.adaptorForFile(filename)
    thisstate = state_da.data_adaptors[0]
    readbackdict = OrderedDict()
    for item in thisstate.data_adaptors:
        pvname = item.getParam("setpoint_pv").split(":")
        psname = pvname[1].split("_")
        magname = psname[1]
        readbackdict[magname] = float(item.getParam("readback"))
    return readbackdict
