"""Generic script to track a bunch through the BTF."""
from __future__ import print_function
import sys
import os
import time
from pathlib import Path

import numpy as np

from orbit.py_linac.lattice_modifications import Replace_Quads_to_OverlappingQuads_Nodes

from btfsim.lattice import diagnostics
from btfsim.lattice.btf_quad_func_factory import btf_quad_func_factory
import btfsim.sim.simulation_main as main
import btfsim.bunch.utils as butils


# Setup
# ------------------------------------------------------------------------------
start = 0  # start node (name or index)
stop = 'HZ04'  # stop node (name or index)
switches = {
    'overlapping_pmq': True,  # use overlapping PMQ field model
    'space_charge': True,  # toggle space charge calculation
    'decorrelate': False,  # decorrelate inital bunch
    'bunch_monitors': False  # bunch monitor nodes within lattice
}

# File paths
script_name = Path(__file__).stem
datestamp = time.strftime('%Y-%m-%d')
timestamp = time.strftime('%y%m%d%H%M%S')
outdir = os.path.join('data/_output/', datestamp)
if not os.path.isdir(outdir):
    os.makedirs(outdir)
fio = dict() 
fio['in'] = dict()
fio['out'] = dict()

# Lattice
fio['in']['mstate'] = 'data/lattice/TransmissionBS34_04212022.mstate'
dispersion_flag = False  # dispersion correction in Twiss calculation
switches['replace_quads'] = True  # use overlapping PMQ field model
n_pmq = 19  # number of permanent-magnet quadrupoles

# Bunch
fio['in']['bunch'] = 'data/bunch/realisticLEBT_50mA_42mA_8555k.dat'
bunch_dec_factor = None
x0 = 0.0  # [m]
y0 = 0.0  # [m]
xp0 = 0.0  # [rad]
yp0 = 0.0  # [rad]
beam_current = 42.0  # current to use in simulation [mA]
beam_current_input = 42.0  # current specified in input bunch file [mA]

# Space charge
sclen = 0.01  # max distance between space charge nodes [m]
gridmult = 6  # grid resolution = 2**gridmult
n_bunches = 3  # number of bunches to model


# Initialize simulation
# ------------------------------------------------------------------------------
_base = '{}-{}-{}-{}'.format(timestamp, script_name, start, stop)
fio['out']['bunch'] = _base + '_bunch_{}.dat'.format(stop)
fio['out']['history'] = os.path.join(outdir, _base + '_history.dat')

sim = main.Sim(outdir=outdir)
sim.dispersion_flag = int(dispersion_flag)
sim.init_lattice(beamlines=["MEBT1", "MEBT2", "MEBT3"], mstatename=fio['in']['mstate'])

# Add bunch monitor nodes at each of the first four quads.
if switches['bunch_monitors']:
    for node in sim.lattice.getNodes():
        if node.getName() in ['MEBT:QH01', 'MEBT:QV02', 'MEBT:QH03', 'MEBT:QV04']:
            filename = os.path.join(outdir, _base + '_bunch_{}.dat'.format(node.getName()))
            bunch_monitor_node = diagnostics.BunchMonitorNode(filename=filename)
            node.addChildNode(bunch_monitor_node, node.ENTRANCE)
for node in sim.lattice.getNodes():
    print(node.getName(), node.getPosition())
    
# Overlapping nodes: replace quads with analytic model. (Must come after 
# `lattice.init()` but before `lattice.init_sc_nodes()`.)
if switches['overlapping_pmq']:
    z_step = 0.001
    quad_names = []  # Leaving this empty will replace every quad.
    for j in range(n_pmq):
        pmq_id = j + 1 + 13
        qname = 'MEBT:FQ{:02.0f}'.format(pmq_id)
        quad_names.append(qname)
        Replace_Quads_to_OverlappingQuads_Nodes(
            sim.lattice,
            z_step, 
            accSeq_Names=["MEBT3"], 
            quad_Names=quad_names, 
            EngeFunctionFactory=btf_quad_func_factory,
        )
    
if switches['space_charge']:
    sim.init_sc_nodes(min_dist=sclen, solver='fft', gridmult=gridmult, n_bunches=n_bunches)
    
sim.init_bunch(
    gen_type='twiss',
    dist='waterbag',
    n_parts=200000,
)

if switches['decorrelate']:
    print('Initial covariance matrix:')
    print(butils.cov(sim.bunch_in))
    sim.decorrelate_bunch()
    print('New covariance matrix:')
    print(butils.cov(sim.bunch_in))

sim.attenuate_bunch(beam_current / beam_current_input)
if bunch_dec_factor is not None and bunch_dec_factor > 1:
    sim.decimate_bunch(bunch_dec_factor)
sim.shift_bunch(x0=x0, y0=y0, xp0=xp0, yp0=yp0)


# Run simulation
# ------------------------------------------------------------------------------
if start == 0:
    sim.dump_bunch(os.path.join(outdir, _base + '_bunch_init.dat'))

def process_start_stop_arg(arg):
    return "MEBT:{}".format(arg) if type(arg) is str else arg

start = process_start_stop_arg(start)
stop = process_start_stop_arg(stop)
sim.run(start=start, stop=stop, out=fio['out']['bunch'])
sim.tracker.write_hist(filename=fio['out']['history'])