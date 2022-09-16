# -- draw lattice
from matplotlib.patches import Polygon

maglist = []
maglist.append([0.1306, 0.061, 1])
maglist.append([0.3139, 0.066, -1])
maglist.append([0.5751, 0.096, 1])
maglist.append([0.7709, 0.096, -1])
maglist.append([1.8517, 0.106, -1])
maglist.append([2.863 + 0.324, 0.55858, 0])
maglist.append([2.863 + 0.789, 0.096, -1])
maglist.append([2.863 + 1.075, 0.146, 1])


def magparse(magval):
    s0 = magval[0] - 0.5 * magval[1]
    s1 = magval[0] + 0.5 * magval[1]
    i = magval[2]
    return (s0, s1, i)


def drawmag(magval, h):

    if not (type(h) == list):
        h = [-h, h]

    s0, s1, i = magparse(magval)
    c = magval[0]

    if i == 1:  # Positive quad
        pts = np.array([[s0, np.mean(h)], [c, h[1]], [s1, np.mean(h)], [c, h[0]]])
    elif i == -1:  # negative quad
        pts = np.array(
            [
                [c, np.mean(h)],
                [s0, h[1]],
                [s1, h[1]],
                [c, np.mean(h)],
                [s1, h[0]],
                [s0, h[0]],
            ]
        )
    elif i == 0:  # dipole
        pts = np.array([[s0, h[1]], [s1, h[1]], [s1, h[0]], [s0, h[0]]])
        plt.text(s0 + 0.01, np.mean(h) + 0.01, "dipole", color="w")

    p = Polygon(pts, fc="blue", ec="black")
    ax = plt.gca()
    ax.add_patch(p)


h = [-0.5, 0]
plt.figure()
for mag in maglist:
    drawmag(mag, h)


plt.ylim([-1, 1])
plt.xlim([0, 4])
