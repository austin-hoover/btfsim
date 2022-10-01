"""Within-simulation plotting routines.

Keep in mind that this is using python2-compatible matplotlib.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import colors
from matplotlib import pyplot as plt


DIMS = ["x", "x'", "y", "y'", "z", "dE"]
UNITS = ["mm", "mrad", "mm", "mrad", "mm", "keV"]


def truncate_cmap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


def histogram_bin_edges(X, bins=10, limits=None):
    d = X.shape[1]
    if type(bins) not in [list, tuple]:
        bins = d * [bins]
    if type(limits) not in [list, tuple]:
        limits = d * [limits] 
    edges = [np.histogram_bin_edges(X[:, i], bins[i], limits[i]) for i in range(d)]
    return edges


def get_bin_centers(edges):
    """Compute bin centers from bin edges."""
    return 0.5 * (edges[:-1] + edges[1:])


def prep_image_for_log(image, method="floor"):
    """Avoid zeros in image."""
    if np.all(image > 0):
        return image
    if method == "floor":
        floor = 1e-12
        if np.max(image) > 0:
            floor = np.min(image[image > 0])
        return image + floor
    elif method == "mask":
        return np.ma.masked_less_equal(image, 0)

    
def plot1d(x, y, ax=None, flipxy=False, kind="step", **kws):
    funcs = {
        "line": ax.plot,
        "bar": ax.bar,
        "step": ax.plot,
    }
    if kind == "step":
        kws.setdefault("drawstyle", "steps-mid")
    if flipxy:
        x, y = y, x
        funcs["bar"] = ax.barh
    return funcs[kind](x, y, **kws)


def profile(
    image,
    xcoords=None,
    ycoords=None,
    ax=None,
    profx=True,
    profy=True,
    kind="step",
    scale=0.12,
    **plot_kws
):
    """Plot 1D projection of image along axis."""
    if xcoords is None:
        xcoords = np.arange(image.shape[1])
    if ycoords is None:
        ycoords = np.arange(image.shape[0])
    plot_kws.setdefault("lw", 0.75)
    plot_kws.setdefault("color", "white")

    def _normalize(prof):
        pmax = np.max(prof)
        if pmax > 0:
            prof = prof / pmax
        return prof

    px, py = [_normalize(np.sum(image, axis=i)) for i in (1, 0)]
    yy = ycoords[0] + scale * np.abs(ycoords[-1] - ycoords[0]) * px
    xx = xcoords[0] + scale * np.abs(xcoords[-1] - xcoords[0]) * py
    for i, (x, y) in enumerate(zip([xcoords, ycoords], [yy, xx])):
        if i == 0 and not profx:
            continue
        if i == 1 and not profy:
            continue
        plot1d(x, y, ax=ax, flipxy=i, kind=kind, **plot_kws)
    return ax


def pcolor(
    image,
    x=None,
    y=None,
    ax=None,
    profx=False,
    profy=False,
    prof_kws=None,
    thresh=None,
    thresh_type="abs",  # {'abs', 'frac'}
    contour=False,
    contour_kws=None,
    handle_log="floor",
    fill_value=None,
    mask_zero=False,
    **plot_kws
):
    """Plot 2D image."""
    if fill_value is not None:
        image = np.ma.filled(image, fill_value=fill_value)
    if thresh is not None:
        if thresh_type == "frac":
            thresh = thresh * np.max(image)
        image[image < max(1e-12, thresh)] = 0
    if mask_zero:
        image = np.ma.masked_less_equal(image, 0)
    if contour_kws is None:
        contour_kws = dict()
    contour_kws.setdefault("color", "white")
    contour_kws.setdefault("lw", 1.0)
    contour_kws.setdefault("alpha", 0.5)
    if prof_kws is None:
        prof_kws = dict()
    if x is None:
        x = np.arange(image.shape[0])
    if y is None:
        y = np.arange(image.shape[1])
    if x.ndim == 2:
        x = x.T
    if y.ndim == 2:
        y = y.T
    mesh = ax.pcolormesh(x, y, image.T, **plot_kws)
    if contour:
        ax.contour(x, y, image.T, **contour_kws)
    profile(image, xcoords=x, ycoords=y, ax=ax, 
                 profx=profx, profy=profy, **prof_kws)
    return ax


def proj2d(
    data=None, 
    info=None, 
    axis=(0, 1),
    bins='auto', 
    limits=None, 
    units=True,
    fig_kws=None,
    text=None,
    **plot_kws
):
    """Plot the 2D projection onto the specified axis."""
    if fig_kws is None:
        fig_kws = dict()
    fig, ax = plt.subplots(**fig_kws)
    labels = True
    if labels:
        if units:
            ax.set_xlabel("{} [{}]".format(DIMS[axis[0]], UNITS[axis[0]]))
            ax.set_ylabel("{} [{}]".format(DIMS[axis[1]], UNITS[axis[1]]))
        else:
            ax.set_xlabel("{}".format(DIMS[axis[0]]))
            ax.set_ylabel("{}".format(DIMS[axis[1]]))
    
    edges = histogram_bin_edges(data[:, axis], bins=bins, limits=limits)
    image, _ = np.histogramdd(data[:, axis], edges)
    centers = [get_bin_centers(e) for e in edges]  
    pcolor(image, x=centers[0], y=centers[1], ax=ax, **plot_kws)
    
    if text is not None:
        if 's' in info:
            ax.set_title('s = {:.3f} [m]'.format(info['s']))
            
            
def corner():
    raise NotImplementedError
    
    
class Plotter:
    """Manage chains of plotting functions, arguments, and file saving."""
    def __init__(self, path='.', default_fig_kws=None, default_save_kws=None,
                 norm=False, scale_emittance=False):
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