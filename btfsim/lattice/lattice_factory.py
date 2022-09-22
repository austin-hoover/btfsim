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
        self.magdict = OrderedDict()
        for quad in quads:
            qname = quad.getName().split(":")[1]  # split off beamline name
            self.magdict[qname] = {}  # initialize sub-dictionary
            self.magdict[qname]["Node"] = quad
            GL = (
                -quad.getParam("dB/dr") * quad.getLength()
            )  # convention: focusing quad has + GL
            if qname == "QV02":
                GL = -GL  # QV02 polarity is always positive.
            # record coefficients and current if applicable
            # (FODO quads do not have set current and are caught by try loop)
            try:
                self.magdict[qname]["coeff"] = self.coeff[qname]
                self.magdict[qname]["current"] = self.gl2c(qname, GL)
            except:  # catch quads that do not have PV names
                if "FQ" in qname:  # catch FODO PMQs
                    self.magdict[qname]["coeff"] = [0, 0]
                    self.magdict[qname]["current"] = 0
                else:  # ignore other elements (not sure what these could be... probably nothing)
                    continue
        print(
            "Lattice generation completed! L={:.3f} m".format(self.lattice.getLength())
        )

    def update_quads(self, **kwargs):
        """Update quadrupole gradients in lattice definition.

        Input is key-value pairs. Key is quadrupole name (ie, QH01), value is current.
        Names should not include beamline name. (ie, refer to QH01, not MEBT:QH01)

        Example:
        >>> update_quads(QH01=10,QV04=-10) will change current for quads 1 and 4
        >>> update_quads(dist=spdict) will change currents for all quads in dictionary
        """
        spdict = kwargs.pop("dict", [])
        units = kwargs.pop("units", "Amps")
        print('spdict:', spdict)

        # -- change gradient for dictionary input
        for key in spdict:
            elementname = key
            if units == "Amps":
                GL = self.c2gl(elementname, float(spdict[key]))
                newcurrent = float(spdict[key])
            elif units == "Tesla":
                GL = float(spdict[key])
                newcurrent = self.gl2c(elementname, float(spdict[key]))
            else:
                raise (
                    TypeError,
                    "Do not understand unit %s for quadrupole setting" % units,
                )
            try:
                # -- update current
                self.magdict[elementname]["current"] = newcurrent
                # -- update gradient in node definition
                newkappa = (
                    -GL / self.magdict[elementname]["Node"].getLength()
                )  # convention: focusing quad has + GL
                if elementname == "QV02":
                    newkappa = (
                        -newkappa
                    )  # special treatment for QV02 poliarty. Kappa + current, GL are always positive
                self.magdict[elementname]["Node"].setParam("dB/dr", newkappa)
                print(
                    "Changed %s to %.3f A (dB/dr=%.3f T/m, GL=%.3f T)"
                    % (elementname, float(newcurrent), newkappa, GL)
                )
            except KeyError:
                print("Element %s is not defined" % (elementname))

        # -- change gradient for keyword input
        for key, value in kwargs.items():
            elementname = key
            if units == "Amps":
                GL = self.c2gl(elementname, float(value))
                newcurrent = float(value)
            elif units == "Tesla":
                GL = float(value)
                newcurrent = self.gl2c(elementname, float(value))
            else:
                raise (
                    TypeError,
                    "Do not understand unit %s for quadrupole setting" % units,
                )
            try:
                # -- update current
                self.magdict[elementname]["current"] = newcurrent
                # -- update gradient in node definition
                newkappa = (
                    -GL / self.magdict[elementname]["Node"].getLength()
                )  # convention: focusing quad has + GL
                if elementname == "QV02":
                    newkappa = (
                        -newkappa
                    )  # special treatment for QV02 poliarty. Kappa + current, GL are always positive
                self.magdict[elementname]["Node"].setParam("dB/dr", newkappa)
                print(
                    "Changed %s to %.3f A (dB/dr=%.3f T/m, GL=%.3f T)"
                    % (elementname, float(newcurrent), newkappa, GL)
                )
            except KeyError:
                print("Element %s is not in beamline" % (elementname))

    def default_quads(self):
        """Load info stored in default quad settings file."""
        key = "QUAD_SET"
        default = Default()
        filename = os.path.join(default.homedir, default.defaultdict[key])
        spdict = util.file2dict(filename)
        self.update_quads(dict=spdict)

    def load_quads(self, filename, units="Tesla"):
        # -- create setpointdict from mstate file
        if filename[-6:] == "mstate":
            if units == "Tesla":
                spdict = mutils.loadQuadReadback(filename)
            elif units == "Amps":
                spdict = mutils.load_quad_setpoint(filename)
            # -- update quad values
            self.update_quads(dict=spdict, units=units)
        else:
            raise NameError(
                "Error loading file %s, expected extension .mstate" % filename
            )

    def update_pmqs(self, **kwargs):
        """Update quadrupole gradients in lattice definition.

        Input is key-value pairs. Key is quadrupole name (ie, FQ01), value is field GL [Tesla].

        names should not include beamline name. (ie, refer to FQ01, not MEBT:FQ01)

        Example:
        >>> update_quads(FQ01=20,FQ04=-20) will change field for PMQs 1 and 4
        >>> update_quads(FQ01=20,FQ04=-20,field='GL') will do the same as above
        >>> update_quads(FQ01=20,FQ04=-20,field='length') change lengths for PMQs 1 and 4
        >>> update_quads(dist=spdict,field='GL') will change fields for all quads in dictionary
        where dictionary is key-value pair for field values of magnets
        """
        # -- if input includes dictionary
        spdict = kwargs.pop("dict", [])
        field = kwargs.pop("field", "GL")

        print(spdict)

        # -- change gradient for dictionary input
        for key in spdict:
            elementname = key
            if field == "GL":
                newGL = float(spdict[key])
                try:
                    L = self.magdict[elementname]["Node"].getLength()
                    newkappa = -newGL / L
                    self.magdict[elementname]["Node"].setParam("dB/dr", newkappa)
                    print(
                        "Changed %s to GL = %.3f T (dB/dr=%.3f T/m, L=%.3f)"
                        % (elementname, float(newGL), newkappa, L)
                    )
                except KeyError:
                    print("Element %s is not defined" % (elementname))
            elif field == "Length":
                newL = float(spdict[key])
                try:
                    GL = (
                        -self.magdict[elementname]["Node"].getParam("dB/dr")
                        * self.magdict[elementname]["Node"].getLength()
                    )
                    self.magdict[elementname]["Node"].setLength(newL)
                    # changing length but holding GL fixed changes effective strength kappa
                    newkappa = -GL / self.magdict[elementname]["Node"].getLength()
                    self.magdict[elementname]["Node"].setParam("dB/dr", newkappa)
                    print(
                        "Changed %s to L = %.3f m (dB/dr=%.3f T/m, GL=%.3f T)"
                        % (elementname, float(newL), newkappa, GL)
                    )
                except KeyError:
                    print("Element %s is not defined" % (elementname))
            else:
                raise (TypeError, "Do not understand field=%s for PMQ element" % field)

        # -- change gradient for keyword input
        for key, value in kwargs.items():
            elementname = key
            if field == "GL":
                newGL = float(value)
                try:
                    L = self.magdict[elementname]["Node"].getLength()
                    newkappa = newGL / L  # convention: focusing quad has + GL
                    self.magdict[elementname]["Node"].setParam("dB/dr", newkappa)
                    print(
                        "Changed %s to GL = %.3f T (dB/dr=%.3f T/m, L=%.3f)"
                        % (elementname, float(newGL), newkappa, L)
                    )
                except KeyError:
                    print("Element %s is not defined" % (elementname))
            elif field == "Length":
                newL = float(value)
                try:
                    GL = (
                        self.magdict[elementname]["Node"].getParam("dB/dr")
                        * self.magdict[elementname]["Node"].getLength()
                    )
                    self.magdict[elementname]["Node"].setLength(newL)
                    # changing length but holding GL fixed changes effective strength kappa
                    newkappa = GL / self.magdict[elementname]["Node"].getLength()
                    self.magdict[elementname]["Node"].setParam("dB/dr", newkappa)
                    print(
                        "Changed %s to L = %.3f m (dB/dr=%.3f T/m, GL=%.3f T)"
                        % (elementname, float(newL), newkappa, GL)
                    )
                except KeyError:
                    print("Element %s is not defined" % (elementname))
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
