"""
This simple class defines the default settings, by pointing to files where defaults
are located.
For example, the default simulation bunch is defined here, as is the default magnet settings

2/12/19
K. Ruisard
"""
import os
import inspect
import numpy as np
import btfsim.util.Utilities as util

class getDefaults():

    def __init__(self):
        self.defaultdict = {}
        self.readDefaults()
        self.homedir = self.defaultdict["HOMEDIR"]
        self.outdir =  self.defaultdict["HOMEDIR"] + self.defaultdict["OUTDIR"]
        self.magnetdir = self.defaultdict["HOMEDIR"] + self.defaultdict["MAGNETDIR"]
        self.bunchdir = self.defaultdict["HOMEDIR"]  + self.defaultdict["BUNCHDIR"] 
        self.latticedir = self.defaultdict["HOMEDIR"]  + self.defaultdict["LATTICEDIR"] 

    def readDefaults(self):
        # Reads and interprets default file, which lists locations of files that define default setpoints/bunches
        simdir = os.path.dirname(os.path.dirname(os.path.dirname(inspect.getfile(util)))) + '/'
        if len(simdir) > 1:
            defaultfilename = simdir+'data/default_settings.csv'
        else:
            simdir = './'
            defaultfilename = 'data/default_settings.csv'
        self.defaultdict = util.file2dict(defaultfilename)
        self.defaultdict["HOMEDIR"] = simdir 
