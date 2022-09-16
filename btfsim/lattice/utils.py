"""
Module for loading magnet state and converting magnet values
"""
from btfsim.util import Defaults
from btfsim.util import Utilities
import numpy as np
from orbit.utils.xml import XmlDataAdaptor
from collections import OrderedDict


# functions for converting quad current to gradient (i2gl)
class magConvert(object):
    # converst gradients to currents and vice versa, based on cofficients in file
    def __init__(self, coeffilename=""):
        if not (coeffilename):
            print("no coeffilename specified")
            # -- get location of mag coefficients file
            defaults = Defaults.getDefaults()
            filename = (
                defaults.defaultdict["HOMEDIR"] + defaults.defaultdict["MAG_COEFF"]
            )
            self.coeff = Utilities.file2dict(filename)
        else:
            print(coeffilename)
            self.coeff = Utilities.file2dict(coeffilename)

    def c2gl(self, quadname, scaledAI):
        """
        arguments:
        1 - Name of quad (ie, QH01)
        2 - Current setpoint (corresponds with scaled AI in IOC), [A]
        """
        scaledAI = float(scaledAI)
        try:
            A = float(self.coeff[quadname][0])
            B = float(self.coeff[quadname][1])
            GL = A * scaledAI + B * scaledAI**2
        except KeyError:
            print(
                "Do not know conversion factor for element %s, gradient value not assigned"
                % quadname
            )
            GL = []
        return GL

    def gl2c(self, quadname, GL):
        """
        arguments:
        1 - Name of quad (ie, QH01)
        2 - Integrated gradient (GL), [T]
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
                "Do not know conversion factor for element %s, current set to 0"
                % quadname
            )
            scaledAI = 0
        return scaledAI

    def igrad2current(self, inputdict):
        """
        Input is dictionary where key is name of magnet, and value is integrated gradient GL [T]
        """
        outputdict = OrderedDict.fromkeys(self.coeff.keys(), [])
        for name in inputdict:
            try:
                outputdict[name] = self.gl2c(name, inputdict[name])
            except:
                print("something went wrong on element %s" % name)
        return outputdict

    def current2igrad(self, inputdict):
        """
        Input is dictionary where key is name of magnet, and value is current setpoint [A]
        """
        outputdict = OrderedDict.fromkeys(self.coeff.keys(), [])
        for name in inputdict:
            try:
                outputdict[name] = self.c2gl(name, inputdict[name])
            except:
                print("something went wrong on element %s" % name)
        return outputdict


# functions to load settings from SCORE .mstate file
def loadQuadSetpoint(filename):
    """
    This retrieves quadrupole setpoints (in Amps) from .mstate file
    Returns dictionary matching quad name with current setpoint
    """
    state_da = XmlDataAdaptor.adaptorForFile(filename)
    thisstate = state_da.data_adaptors[0]

    setpointdict = OrderedDict()
    for item in thisstate.data_adaptors:
        pvname = item.getParam("setpoint_pv").split(":")
        psname = pvname[1].split("_")
        magname = psname[1]
        # -- get current set-point
        setpoint = float(item.getParam("setpoint"))
        setpointdict[magname] = setpoint

    return setpointdict


def loadQuadReadback(filename):
    """
    This retrieves quadrupole readback (in Tesla) from .mstate file
    Returns dictionary matching quad name with field readback
    """

    state_da = XmlDataAdaptor.adaptorForFile(filename)
    thisstate = state_da.data_adaptors[0]

    readbackdict = OrderedDict()
    for item in thisstate.data_adaptors:
        # -- get magnet name from pvname
        pvname = item.getParam("setpoint_pv").split(":")
        psname = pvname[1].split("_")
        magname = psname[1]
        # -- get current set-point
        readback = float(item.getParam("readback"))
        readbackdict[magname] = readback
    return readbackdict
