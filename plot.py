import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck

from multiple import multiple_formatter
import couplings


#
#
# Choose contours and colours
#
#
levels = [0, 0.025, 0.05, 0.1]
colours = [(0.05, 0.3, 0.6), (0.12, 0.47, 0.706), (0.5, 0.7, 0.8)]

opts_contourlabels = dict(
    colors='k',
    fmt='%1.3f',
    fontsize=12,
    manual=[(1.2, 1.4), (1.4, 1.3), (0.4, 1.3),
            (0.7, 1.4), (1.4, 1), (1.4, 1.2)]
)

plot_font = {'size': 16}
plot_size = (6, 6)
plt.rc('font', **plot_font)


#
#
# Supernova constraint
#
#
npoints = 1000
x = np.linspace(0, np.pi / 2, npoints)
y = np.linspace(0, np.pi / 2, npoints)
z = couplings.SNconstraint(*np.meshgrid(x, y))

fig, ax = plt.subplots(figsize=plot_size)
con = ax.contour(x, y, z, levels, colors='k', linewidths=1)
conf = ax.contourf(x, y, z, levels, colors=colours)


#
#
# Perturbativity bounds
#
#
npoints = 100
xpert = np.linspace(0.001, np.pi / 2, npoints)
ypert = np.linspace(0.001, np.pi / 2, npoints)

zpert = couplings.yukawa_perturbative(*np.meshgrid(xpert, ypert))
ax.contourf(xpert, ypert, zpert, [0, 0.5], hatches=['///',''], colors='w', alpha=0)

#
#
# Final plot options
#
#
ax.clabel(con, **opts_contourlabels)
ax.set(xlim=(0, np.pi / 2), ylim=(0, np.pi / 2))
ax.set_xlabel(r'$\beta_{1}$')
ax.set_ylabel(r'$\beta_{2}$', rotation=0)
ax.tick_params(direction='in', which='both')

for ti, t in enumerate([ax.xaxis, ax.yaxis]):
    d = (8, 8)[ti]
    t.set_major_locator(tck.MultipleLocator(np.pi / d))
    t.set_major_formatter(plt.FuncFormatter(multiple_formatter(denominator=d)))
    t.set_minor_locator(tck.AutoMinorLocator(4))
    t.set_ticks_position('both')

plt.tight_layout()
plt.show()
