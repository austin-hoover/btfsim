from __future__ import print_function
import sys
import os
import time
import shutil
from pathlib import Path
from pprint import pprint
from pathlib import Path 

import numpy as np

from orbit.py_linac.lattice_modifications import Replace_Quads_to_OverlappingQuads_Nodes

from btfsim.lattice import diagnostics
from btfsim.lattice.btf_quad_func_factory import btf_quad_func_factory
from btfsim.sim.sim import Sim
from btfsim.util import utils
import btfsim.bunch.utils as butils


# Setup
# ------------------------------------------------------------------------------
start = 0  # start node (name or index)
stop = 'QH03'  # stop node (name or index)
switches = {
    'space_charge': True,  # toggle space charge calculation
    'save_init_bunch': True,   # whether to save initial bunch to file
}
    
# Lattice
fio['in']['mstate'] = 'data/lattice/TransmissionBS34_04212022.mstate'

# Bunch
bunch_dec_factor = None
beam_current = 42.0  # current to use in simulation [mA]
beam_current_input = 42.0  # current specified in input bunch file [mA]

# Space charge
sclen = 0.01  # max distance between space charge nodes [m]
gridmult = 6  # grid resolution = 2**gridmult
n_bunches = 3  # number of bunches to model

# File paths (do not change)
fio = {'in': {}, 'out': {}}  # store input/output paths
script_name = Path(__file__).stem
datestamp = time.strftime('%Y-%m-%d')
timestamp = time.strftime('%y%m%d%H%M%S')
outdir = os.path.join('data/_output/', datestamp)
if not os.path.isdir(outdir):
    os.makedirs(outdir)

# Save the current git revision hash (do not change).
revision_hash = utils.git_revision_hash()
repo_url = utils.git_url()
if revision_hash and repo_url:
    _filename = '{}-{}-git_hash.txt'.format(timestamp, script_name)
    file = open(os.path.join(outdir, _filename), 'w')
    file.write('{}/-/tree/{}'.format(repo_url, revision_hash))
    file.close()
    
# Save time-stamped copy of this file (do not change).
filename = os.path.join(outdir, '{}-{}.py'.format(timestamp, script_name))
shutil.copy(__file__, filename)


# Initialize simulation
# ------------------------------------------------------------------------------
_base = '{}-{}-{}-{}'.format(timestamp, script_name, start, stop)
fio['out']['bunch'] = _base + '-bunch-{}.dat'.format(stop)
fio['out']['history'] = os.path.join(outdir, _base + '-history.dat')

sim = Sim(outdir=outdir)
sim.dispersion_flag = False
sim.init_lattice(
    beamlines=['MEBT1'], 
    mstatename=fio['in']['mstate'],
)
    
# Dump bunch coordinates at each quad entrance.
for node in sim.lattice.getNodes():
    if node.getName() in ['MEBT:QH01', 'MEBT:QV02', 'MEBT:QH03', 'MEBT:QV04']:
        filename = os.path.join(outdir, _base + '-bunch-{}.dat'.format(node.getName()))
        bunch_monitor_node = diagnostics.BunchMonitorNode(filename=filename)
        node.addChildNode(bunch_monitor_node, node.ENTRANCE)
for node in sim.lattice.getNodes():
    print(node.getName(), node.getPosition())
    
if switches['space_charge']:
    sim.init_sc_nodes(min_dist=sclen, solver='fft', gridmult=gridmult, n_bunches=n_bunches)
    
sim.init_bunch(
    gen_type='twiss', 
    dist='waterbag',
    n_parts=200000,
)
sim.attenuate_bunch(beam_current / beam_current_input)
if bunch_dec_factor is not None and bunch_dec_factor > 1:
    sim.decimate_bunch(bunch_dec_factor)


# Run simulation
# ------------------------------------------------------------------------------
if switches['save_init_bunch']:
    sim.dump_bunch(os.path.join(outdir, _base + '-bunch-init.dat'))

def process_start_stop_arg(arg):
    return "MEBT:{}".format(arg) if type(arg) is str else arg

start = process_start_stop_arg(start)
stop = process_start_stop_arg(stop)
sim.run(start=start, stop=stop, out=fio['out']['bunch'])
sim.tracker.write_hist(filename=fio['out']['history'])