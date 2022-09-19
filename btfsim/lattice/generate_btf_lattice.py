"""Module to generate BTF lattice based on XML file.

By default, loads lattice defined in defaults file: 
'/data/lattice/BTF_lattice.xml'

Optional keyword arguments:
filename = 'path/to/xml/file.xml'
beamlines = ['MEBT',], list of beamlines to use from XML file

Usage:
>>> import btfsim.lattice.genLattice as gl
>>> lat = gl.genLattice()
>>> lat.updateQuads(QH01=10)

lattice definition stored in lat.accLattice
quad params accessible in lat.quaddict
"""
import sys
import os
from collections import OrderedDict

from orbit.py_linac.linac_parsers import SNS_LinacLatticeFactory
from orbit.py_linac.lattice.LinacApertureNodes import LinacApertureNode

from btfsim.util.default import Default
import btfsim.util.utils as utils
import btfsim.lattice.utils as mutils


class GenLattice(mutils.magConvert):

    # -- BTF slit parameters
    pipediameter = 0.04
    slitwidth = {
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

    def __init__(self, **kwargs):
        # -- get default lattice file
        default = Default()
        defaultlatticeFileName = os.path.join(
            default.defaultdict["HOMEDIR"], default.defaultdict["XML_FILE"])

        # -- parse args
        self.filename = kwargs.get("xml", defaultlatticeFileName)
        kwargs.pop("xml")
        self.beamlines = kwargs.get("beamline", ["MEBT1", "MEBT2", "MEBT3"])
        kwargs.pop("beamline")
        maxdriftlen = kwargs.get("ds", 0.012)
        kwargs.pop("ds")

        # -- super (inherit methods from mutils):
        # c2gl, gl2c, current2kappa, kappa2current, and attribute coeff)
        super(GenLattice, self).__init__(**kwargs)

        print(self.filename)
        print(self.beamlines)

        # ---- create the factory instance
        btf_linac_factory = SNS_LinacLatticeFactory()
        btf_linac_factory.setMaxDriftLength(
            maxdriftlen
        )  # 0.00001   0.016  float(Drift_len)

        # ---- make lattice from XML file
        self.accLattice = btf_linac_factory.getLinacAccLattice(
            self.beamlines, self.filename
        )

        # -- make dictionary of quads
        quads = self.accLattice.getQuads()
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
        print("Lattice generation completed! L=%.3f m" % (self.accLattice.getLength()))

    def updateQuads(self, **kwargs):
        """
        updates quadrupole gradients in lattice definition
        Input is key-value pairs. Key is quadrupole name (ie, QH01), value is current.

        names should not include beamline name. (ie, refer to QH01, not MEBT:QH01)

        Some ideas for keyword arguments that might be good to use:
        beamline
        QH01
        QV02
        QH03
        QV04
        QH05
        QV06
        QV07
        QH08
        QV09

        Example:
        >>> updateQuads(QH01=10,QV04=-10) will change current for quads 1 and 4
        >>> updateQuads(dist=spdict) will change currents for all quads in dictionary
        """
        # -- if input includes dictionary
        spdict = kwargs.pop("dict", [])
        units = kwargs.pop("units", "Amps")

        print(spdict)

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

    def defaultQuads(self):
        """Load info stored in default quad settings file."""
        key = "QUAD_SET"
        default = Default()
        filename = os.path.join(default.homedir, default.defaultdict[key])
        spdict = util.file2dict(filename)
        self.updateQuads(dict=spdict)

    def loadQuads(self, filename, units="Tesla"):
        # -- create setpointdict from mstate file
        if filename[-6:] == "mstate":
            if units == "Tesla":
                spdict = mutils.loadQuadReadback(filename)
            elif units == "Amps":
                spdict = mutils.loadQuadSetpoint(filename)
            # -- update quad values
            self.updateQuads(dict=spdict, units=units)
        else:
            raise NameError(
                "Error loading file %s, expected extension .mstate" % filename
            )

    def updatePMQs(self, **kwargs):
        """
        updates quadrupole gradients in lattice definition
        Input is key-value pairs. Key is quadrupole name (ie, FQ01), value is field GL [Tesla].

        names should not include beamline name. (ie, refer to FQ01, not MEBT:FQ01)

        Example:
        >>> updateQuads(FQ01=20,FQ04=-20) will change field for PMQs 1 and 4
        >>> updateQuads(FQ01=20,FQ04=-20,field='GL') will do the same as above
        >>> updateQuads(FQ01=20,FQ04=-20,field='length') change lengths for PMQs 1 and 4
        >>> updateQuads(dist=spdict,field='GL') will change fields for all quads in dictionary
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

    def addSlit(self, slitname, pos=0.0, width=None):
        """
        slitname: name of slit. ex: "MEBT:HZ04"
        pos=0: transverse position of slit in mm. pos=0 is at bunch center
        width=None: width of slit in mm. if None, uses lookup table

        """

        # -- get longitudinal position of slitname
        slitnode = self.accLattice.getNodeForName("MEBT:" + slitname)
        zpos = slitnode.getPosition()
        node_name = slitnode.getName()

        # get slit width is not provided
        if not (width):
            width = self.slitwidth[slitname]

        # determine if horizontal or vertical slit
        if slitname[0] == "V":
            dx = width * 1e-3
            dy = 1.1 * self.pipediameter
            c = pos * 1e-3
            d = 0.0
        elif slitname[0] == "H":
            dy = width * 1e-3
            dx = 1.1 * self.pipediameter
            d = pos * 1e-3
            c = 0.0
        else:
            raise KeyError("Cannot determine plane for slit %s" % slitname)

        a = 0.5 * dx
        b = 0.5 * dy
        shape = 3  # rectangular

        # -- create aperture note
        # -- in this call, pos is longitudinal position
        apertureNode = LinacApertureNode(shape, a, b, pos=zpos, c=c, d=d, name=slitname)

        # -- add as child to slit Marker node
        apertureNode.setName(node_name + ":Aprt")
        apertureNode.setSequence(slitnode.getSequence())
        slitnode.addChildNode(apertureNode, slitnode.ENTRANCE)

        print("Inserted %s at %.3f mm" % (slitname, pos))
        return apertureNode
