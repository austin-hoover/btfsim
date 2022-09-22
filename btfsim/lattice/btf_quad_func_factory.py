"""Quadrupole factory.

This is a Enge Function Factory specific for the SNS. Some Enge's function 
parameters are found by fitting the measured or calculated field distributions.
Others are generated with quad's length and beam pipe diameter. These 
parameters for SNS with the specific quads' names. For your accelerator, you 
should create your own factory.
"""
import sys
import os
import math

from orbit.py_linac.overlapping_fields.overlapping_quad_fields_lib import EngeFunction
from orbit.py_linac.overlapping_fields.overlapping_quad_fields_lib import SimpleQuadFieldFunc
from orbit.py_linac.overlapping_fields.overlapping_quad_fields_lib import PMQ_Trace3D_Function


def btf_quad_func_factory(quad):
    """Generate Enge's Function for SNS quads. 
    
    For some of the quads in the SNS lattice we know the Enge's parameters. So, it 
    is an SNS specific function.
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
