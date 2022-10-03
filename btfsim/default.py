import inspect
import os

import numpy as np

from btfsim import utils


class Default:
    def __init__(self):
        self.default_dict = dict()
        self.read()
        self.home = self.defaultdict['HOMEDIR']
        self.out = os.path.join(self.home, self.defaultdict['OUTDIR'])
        self.magnet = os.path.join(self.home, self.defaultdict['MAGNETDIR'])
        self.bunch = os.path.join(self.home, self.defaultdict['BUNCHDIR'])
        self.lattice = os.path.join(self.home, self.defaultdict['LATTICEDIR'])

    def read(self):
        """Read default file defining locations of setpoints/bunches."""
        sim_dir = os.getcwd()
        self.defaultdict = utils.file_to_dict(
            os.path.join(sim_dir, 'data/default_settings.csv'))
        self.defaultdict['HOMEDIR'] = sim_dir