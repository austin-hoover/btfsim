from __future__ import print_function
from orbit.teapot import DriftTEAPOT


class BunchMonitorNode(DriftTEAPOT):
    def __init__(self, filename="bunch.dat", name="bunch_monitor"):
        DriftTEAPOT.__init__(self, name)
        self.setLength(0.0)
        self.filename = filename

    def track(self, params_dict):
        bunch = params_dict["bunch"]
        bunch.dumpBunch(self.filename)
