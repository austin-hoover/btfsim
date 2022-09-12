import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(1, os.getcwd())
import btfsim.sim.simulation_main as main
import btfsim.lattice.magUtilities as magutil
import btfsim.bunch.bunchUtilities as butil
from btfsim.lattice.btf_quad_func_factory import BTF_QuadFunctionFactory as quadfunc
from orbit.py_linac.lattice_modifications import Replace_Quads_to_OverlappingQuads_Nodes


mstatefile = 'data/lattice/TransmissionBS34_04212022.mstate'
sim = main.simBTF()
sim.initLattice(beamline=["MEBT1", "MEBT2", "MEBT3"], mstatename=mstatefile)


lattice = sim.accLattice

for node in lattice.getNodes():
    print(node.getName(), node.getPosition())