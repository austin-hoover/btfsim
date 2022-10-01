from __future__ import print_function

from orbit.py_linac.lattice import BaseLinacNode


class BunchMonitorNode(DriftTEAPOT):
    def __init__(self, filename="bunch.dat", name="bunch_monitor"):
        BaseLinacNode.__init__(self, name)
        self.filename = filename
        self.active = True

    def track(self, params_dict):
        if self.active:
            bunch = params_dict["bunch"]
            bunch.dumpBunch(self.filename)