from __future__ import print_function
import sys
import os
from collections import OrderedDict

from orbit.py_linac.linac_parsers import SNS_LinacLatticeFactory
from orbit.py_linac.lattice.LinacApertureNodes import LinacApertureNode

import btfsim.lattice.utils as mutils
from btfsim.util.default import Default
import btfsim.util.utils as utils


class LatticeGenerator(mutils.MagnetConverter):
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
        defaultlatticeFileName = os.path.join(
            default.defaultdict["HOMEDIR"], default.defaultdict["XML_FILE"]
        )

        self.xml = xml
        if self.xml is None:
            self.xml = defaultlatticeFileName
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
        self.magnets = OrderedDict()
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
                spdict = mutils.loadQuadReadback(filename)
            elif units == "Amps":
                spdict = mutils.load_quad_setpoint(filename)
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
