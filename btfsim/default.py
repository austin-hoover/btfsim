import inspect
import os

import numpy as np

from btfsim.util import utils


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
        simdir = os.path.dirname(os.path.dirname(os.path.dirname(inspect.getfile(utils))))
        self.defaultdict = utils.file_to_dict(
            os.path.join(simdir, 'data/default_settings.csv'))
        self.defaultdict['HOMEDIR'] = simdir