"""Plot a snapshot of the bunch.

Keep in mind that this is using python2-compatible matplotlib.
"""
import sys
import os

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


DIMS = ["x", "x'", "y", "y'", "z", "dE"]
UNITS = ["mm", "mrad", "mm", "mrad", "mm", "keV"]


def _histogram_bin_edges(X, bins=10, limits=None):
    d = X.shape[1]
    if type(bins) is not list:
        bins = d * [bins]
    if type(limits) is not list:
        limits = d * [limits] 
    edges = [np.histogram_bin_edges(X[:, i], bins[i], limits[i]) for i in range(d)]
    return edges


def _bin_centers(edges):
    """Compute bin centers from bin edges."""
    return 0.5 * (edges[:-1] + edges[1:])


def proj2d(
    data=None, 
    info=None, 
    axis=(0, 1),
    bins='auto', 
    limits=None, 
    fig_kws=None,
    **plot_kws
):
    """Plot the 2D projection onto the specified axis."""
    if fig_kws is None:
        fig_kws = dict()
    fig, ax = plt.subplots(**fig_kws)
    labels = True
    if labels:
        ax.set_xlabel("{} [{}]".format(DIMS[axis[0]], UNITS[axis[0]]))
        ax.set_ylabel("{} [{}]".format(DIMS[axis[1]], UNITS[axis[1]]))
    
    edges = _histogram_bin_edges(data[:, axis], bins=bins, limits=limits)
    image, _ = np.histogramdd(data[:, axis], edges)
    centers = [_bin_centers(e) for e in edges]            
    ax.pcolormesh(centers[0], centers[1], image.T, **plot_kws)
    return ax
    
    
class Plotter:
    
    def __init__(self, path='.', default_fig_kws=None, default_save_kws=None):
        self.path = path
        self.default_fig_kws = default_fig_kws
        self.default_save_kws = default_save_kws
        if self.default_fig_kws is None:
            self.default_fig_kws = dict()
        if self.default_save_kws is None:
            self.default_save_kws = dict()
        self.funcs = []
        self.fig_kws = []
        self.save_kws = []
        self.plot_kws = []
        self.names = []
        self.counter = 0
            
    def add_func(self, func, fig_kws=None, save_kws=None, name=None, **plot_kws):
        self.funcs.append(func)
        self.fig_kws.append(fig_kws if fig_kws else self.default_fig_kws)
        self.save_kws.append(save_kws if save_kws else self.default_save_kws)
        self.plot_kws.append(plot_kws)
        self.names.append(name if name else 'plot{}'.format(self.counter))
        self.counter += 1
        
    def plot(self, data=None, info=None, verbose=False):
        for i in range(len(self.funcs)):
            filename = self.names[i]
            for key in ['step', 'node']:
                if key in info:
                    filename = filename + '_{}'.format(info[key])
            filename = os.path.join(self.path, filename)
            if verbose:
                print("Calling plot function '{}' ({}).".format(self.names[i], self.funcs[i].__name__))
            self.funcs[i](data=data, info=info, fig_kws=self.fig_kws[i], **self.plot_kws[i])
            plt.savefig(filename, **self.save_kws[i])
            plt.close()
            