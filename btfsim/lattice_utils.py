from __future__ import print_function
import collections
import math
import os
import sys

import numpy as np

from orbit.py_linac.linac_parsers import SNS_LinacLatticeFactory
from orbit.py_linac.lattice.LinacApertureNodes import LinacApertureNode
from orbit.py_linac.overlapping_fields.overlapping_quad_fields_lib import EngeFunction
from orbit.py_linac.overlapping_fields.overlapping_quad_fields_lib import SimpleQuadFieldFunc
from orbit.py_linac.overlapping_fields.overlapping_quad_fields_lib import PMQ_Trace3D_Function
from orbit.utils.xml import XmlDataAdaptor

from btfsim import utils
from btfsim.default import Default


def load_quad_setpoint(filename):
    """Load quadrupole setpoints from .mstate file.

    Returns dictionary matching quad name with current setpoint [A].
    """
    state_da = XmlDataAdaptor.adaptorForFile(filename)
    thisstate = state_da.data_adaptors[0]
    setpointdict = collections.OrderedDict()
    for item in thisstate.data_adaptors:
        pvname = item.getParam("setpoint_pv").split(":")
        psname = pvname[1].split("_")
        magname = psname[1]
        setpointdict[magname] = float(item.getParam("setpoint"))
    return setpointdict


def load_quad_readback(filename):
    """Load quadrupole readbacks from .mstate file.

    Returns dictionary matching quad name with field readback [T].
    """
    state_da = XmlDataAdaptor.adaptorForFile(filename)
    thisstate = state_da.data_adaptors[0]
    readbackdict = collections.OrderedDict()
    for item in thisstate.data_adaptors:
        pvname = item.getParam("setpoint_pv").split(":")
        psname = pvname[1].split("_")
        magname = psname[1]
        readbackdict[magname] = float(item.getParam("readback"))
    return readbackdict


def quad_func_factory(quad):
    """Generate Enge's Function for SNS quads.
    
    This is factory is specific to the BTF magnets. Some Enge's function parameters 
    are found by fitting the measured or calculated field distributions; others are
    generated from the quadrupole length and beam pipe diameter. The parameters 
    here are known for the BTF quads.
    
    Parameters
    ----------
    quad : AccNode
    
    Returns
    -------
    EngeFunction 
    """
    name = quad.getName()
    if name in ["MEBT:QV02"]:
        length_param = 0.066
        acceptance_diameter_param = 0.0363
        cutoff_level = 0.001
        func = EngeFunction(length_param, acceptance_diameter_param, cutoff_level)
        return func
    elif name in ["MEBT:QH01"]:
        length_param = 0.061
        acceptance_diameter_param = 0.029
        cutoff_level = 0.001
        func = EngeFunction(length_param, acceptance_diameter_param, cutoff_level)
        return func
    # added this for BTF PMQ's (arrangements of 2 pancakes per quad)
    elif name.find("FQ") >= 0:
        # number of pancakes comprising 1 quad
        npancakes = 2
        # inches to meters
        inch2meter = 0.0254
        # pole field [T] (from Menchov FEA simulation, field at inner radius 2.25 cm)
        Bpole = 0.574  # 1.2
        # inner radius (this is actually radius of inner aluminum housing, which is
        # slightly less than SmCo2 material inner radius)
        ri = 0.914 * inch2meter
        # outer radius (this is actually radius of through-holes, which is slightly
        # larger than SmCo2 material)
        ro = 1.605 * inch2meter
        # length of quad (this is length of n pancakes sancwhiched together)
        length_param = npancakes * 1.378 * inch2meter
        cutoff_level = 0.01
        func = PMQ_Trace3D_Function(length_param, ri, ro, cutoff_level=cutoff_level)
        return func
    # ----- general Enge's Function (for other quads with given aperture parameter)
    elif quad.hasParam("aperture"):
        length_param = quad.getLength()
        acceptance_diameter_param = quad.getParam("aperture")
        cutoff_level = 0.001
        func = EngeFunction(length_param, acceptance_diameter_param, cutoff_level)
        return func
    else:
        msg = "SNS_EngeFunctionFactory Python function. "
        msg += os.linesep
        msg += "Cannot create the EngeFunction for the quad!"
        msg += os.linesep
        msg = msg + "quad name = " + quad.getName()
        msg = msg + os.linesep
        msg = msg + "It does not have the aperture parameter!"
        msg = msg + os.linesep
        orbitFinalize(msg)
        return None


class MagnetConverter(object):
    """Class to convert gradient/current."""
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


class LatticeGenerator(MagnetConverter):
    """Class to generate BTF lattice from XML file.

    By default, loads lattice defined in '/data/lattice/BTF_lattice.xml'.

    Attributes
    ----------
    lattice : orbit.lattice.AccLattice
        The PyORBIT accelerator lattice instance.
    magnets : dict
        Quadrupole magnet names and strengths.
    """
    def __init__(
        self,
        xml=None,
        beamlines=["MEBT1", "MEBT2", "MEBT3"],
        maxdriftlen=0.012,
        coef_filename=None,
    ):
        """Constructor.

        filename : str
            Path to the XML file.
        beamlines : list[str]
            List of beamlines to include in the lattice construction.
        maxdriftlen : float
            Maximum drift length [m].
        coef_filename : str
            File name for magnet coefficients.
        """
        pipe_diameter = 0.04
        slit_widths = {
            "HZ04": 0.2,
            "HZ06": 0.2,
            "VT04": 0.2,
            "VT06": 0.2,
            "HZ34a": 0.2,
            "HZ34b": 0.2,
            "VT34a": 0.2,
            "VT34b": 0.2,
            "VS06": 0.2,  # larger slit is 0.8 mm
        }

        default = Default()
        default_lattice_filename = os.path.join(
            default.defaultdict["HOMEDIR"], default.defaultdict["XML_FILE"]
        )

        self.xml = xml
        if self.xml is None:
            self.xml = default_lattice_filename
        self.beamlines = beamlines

        print("xml file:", self.xml)
        print("beamlines:", self.beamlines)

        super(LatticeGenerator, self).__init__(coef_filename=coef_filename)

        # Create the factory instance.
        btf_linac_factory = SNS_LinacLatticeFactory()
        btf_linac_factory.setMaxDriftLength(maxdriftlen)

        # Make lattice from XML file.
        self.lattice = btf_linac_factory.getLinacAccLattice(self.beamlines, self.xml)

        # Make dictionary of quads.
        quads = self.lattice.getQuads()
        self.magnets = collections.OrderedDict()
        for quad in quads:
            qname = quad.getName().split(":")[1]  # split off beamline name
            self.magnets[qname] = dict()  # initialize sub-dictionary
            self.magnets[qname]["Node"] = quad
            # By convention, focusing quad has GL > 0, QV02 is always positive.
            GL = -quad.getParam("dB/dr") * quad.getLength()
            if qname == "QV02":
                GL = -GL 
            # Record coefficients and current if applicable (FODO quads do not have 
            # set current and are caught by try loop.)
            try:
                self.magnets[qname]["coeff"] = self.coeff[qname]
                self.magnets[qname]["current"] = self.gl2c(qname, GL)
            except:  
                # Catch quads that do not have PV names.
                if "FQ" in qname:  # FODO PMQs
                    self.magnets[qname]["coeff"] = [0, 0]
                    self.magnets[qname]["current"] = 0
                else:  
                    # Ignore other elements (not sure what these could be... probably nothing).
                    continue
        print("Lattice generation completed! L={:.3f} m"
              .format(self.lattice.getLength()))

    def update_quads(self, units='Amps', **spdict):
        """Update quadrupole gradients in lattice definition.
        
        **spdict : dict
            Keys are quadrupole names; values are currents. Names should not include
            beamline name ('QH01' instead of 'MEBT1:QH01'). 
        """        
        for element_name, value in spdict.items():
            if units == "Amps":
                GL = self.c2gl(element_name, float(value))
                newcurrent = float(value)
            elif units == "Tesla":
                GL = float(value)
                newcurrent = self.gl2c(element_name, float(value))
            else:
                raise (
                    TypeError,
                    "Do not understand unit {} for quadrupole setting".format(units),
                )
            try:
                self.magnets[element_name]["current"] = newcurrent
                # Update gradient in node definition. By convention, the 
                # focusing quad has GL > 0. (Special treatment for QV02 polarity: 
                # kappa + current, GL are always positive.)
                newkappa = -GL / self.magnets[element_name]["Node"].getLength()
                if element_name == "QV02":
                    newkappa = -newkappa
                self.magnets[element_name]["Node"].setParam("dB/dr", newkappa)
                print(
                    "Changed {} to {:.3f} [A] (dB/dr={:.3f} [T/m], GL={:.3f} [T])."
                    .format(element_name, float(newcurrent), newkappa, GL)
                )
            except KeyError:
                print("Element {} is not defined.".format(element_name))

    def default_quads(self):
        """Load info stored in default quad settings file."""
        key = "QUAD_SET"
        default = Default()
        filename = os.path.join(default.homedir, default.defaultdict[key])
        spdict = util.file2dict(filename)
        self.update_quads(**spdict)

    def load_quads(self, filename, units="Tesla"):
        if filename[-6:] == "mstate":
            if units == "Tesla":
                spdict = load_quad_readback(filename)
            elif units == "Amps":
                spdict = load_quad_setpoint(filename)
            self.update_quads(units=units, **spdict)
        else:
            raise NameError("Error loading file {}, expected extension .mstate".format(filename))

    def update_pmqs(self, field='GL', **spdict):
        """Update quadrupole gradients in lattice definition.

        Input is key-value pairs. Key is quadrupole name (ie, FQ01), value is field GL [Tesla].

        names should not include beamline name. (ie, refer to FQ01, not MEBT:FQ01)

        Example:
        >>> update_quads(FQ01=20, FQ04=-20) will change field for PMQs 1 and 4
        >>> update_quads(FQ01=20, FQ04=-20,field='GL') will do the same as above
        >>> update_quads(FQ01=20, FQ04=-20,field='length') change lengths for PMQs 1 and 4
        >>> update_quads(field='GL', **spdict) will change fields for all quads in dictionary
        where dictionary is key-value pair for field values of magnets
        """
        for element_name, value in spdict.items():
            if field == "GL":
                newGL = float(value)
                try:
                    L = self.magnets[element_name]["Node"].getLength()
                    newkappa = newGL / L  # convention: focusing quad has + GL
                    self.magnets[element_name]["Node"].setParam("dB/dr", newkappa)
                    print(
                        "Changed %s to GL = %.3f T (dB/dr=%.3f T/m, L=%.3f)"
                        % (element_name, float(newGL), newkappa, L)
                    )
                except KeyError:
                    print("Element %s is not defined" % (element_name))
            elif field == "Length":
                newL = float(value)
                try:
                    GL = (
                        self.magnets[element_name]["Node"].getParam("dB/dr")
                        * self.magnets[element_name]["Node"].getLength()
                    )
                    self.magnets[element_name]["Node"].setLength(newL)
                    # changing length but holding GL fixed changes effective strength kappa
                    newkappa = GL / self.magnets[element_name]["Node"].getLength()
                    self.magnets[element_name]["Node"].setParam("dB/dr", newkappa)
                    print(
                        "Changed %s to L = %.3f m (dB/dr=%.3f T/m, GL=%.3f T)"
                        % (element_name, float(newL), newkappa, GL)
                    )
                except KeyError:
                    print("Element %s is not defined" % (element_name))
            else:
                raise (TypeError, "Do not understand field=%s for PMQ element" % field)

    def add_slit(self, slit_name, pos=0.0, width=None):
        """Add a slit to the lattice.

        slit_name : str
            The name of slit, e.g., 'MEBT:HZ04'.
        pos : float
            Transverse position of slit [mm] (bunch center is at zero).
        width : float or None
            Width of slit [mm]. If None, uses lookup table.
        """
        if width is None:
            width = self.slit_widths[slit_name]

        # Determine if horizontal or vertical slit.
        if slit_name[0] == "V":
            dx = width * 1e-3
            dy = 1.1 * self.pipe_diameter
            c = pos * 1e-3
            d = 0.0
        elif slit_name[0] == "H":
            dy = width * 1e-3
            dx = 1.1 * self.pipe_diameter
            d = pos * 1e-3
            c = 0.0
        else:
            raise KeyError("Cannot determine plane for slit {}".format(slit_name))

        a = 0.5 * dx
        b = 0.5 * dy
        shape = 3  # rectangular

        # Create aperture node. In this call, pos is longitudinal position.
        slit_node = self.lattice.getNodeForName("MEBT:" + slit_name)
        apertureNode = LinacApertureNode(
            shape,
            a,
            b,
            c=c,
            d=d,
            pos=slit_node.getPosition(),
            name=slit_name,
        )

        # Add as child to slit marker node.
        apertureNode.setName(slit_node.getName() + ":Aprt")
        apertureNode.setSequence(slit_node.getSequence())
        slit_node.addChildNode(apertureNode, slit_node.ENTRANCE)
        print("Inserted {} at {:.3f} mm".format(slit_name, pos))

        return apertureNode