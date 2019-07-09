import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# import matplotlib.ticker as tck

# from multiple import multiple_formatter
import couplings


save = True
filename = 'images/plotb1b2-leg.pdf'

#
# Choose contours and colours
#

levels = [0, 0.025, 0.05, 0.1, 0.2, 0.5, 1, 10]

fillcolours = [(0.05, 0.2, 0.5), (0.15, 0.35, 0.6),
               (0.4, 0.6, 0.8),
               # (0.65, 0.75, 0.9),
               (0.75, 0.85, 1.0),
               (1, 1, 1), (1, 1, 1), (1, 1, 1)]

sncolour = 'k'
ecolour = 'gold'
hatchcolour = '0.3'


opts_sncontours = dict(
    colors=sncolour,
    linewidths=0.5
)

opts_sncontourlabels = dict(
    colors=sncolour,
    fmt='%g',
    fontsize=12,
    manual=[(0.4, 0.2), (0.4, 0.7), (0.67, 1.05),
            (1.4, 1.3), (0.85, 1.1),
            (0.8, 1.1), (1.1, 1.4),
            (1.4, 1.1), (1.4, 1.2), (1.4, 0.9), (1.4, 0.2)]
)

opts_econtours = dict(
    colors=ecolour,
    linewidths=1.5,
    linestyles='dashed',
    levels=[0, 0.001, 0.01, 0.1]
)

opts_econtourlabels = dict(
    colors=ecolour,
    fmt='%g',
    fontsize=14,
    manual=[(0.5, 1.5), (0.3, 1.4), (0.75, 0.85),
            (1.2, 1.5), (1.2, 1.35), (1.25, 0.85)]
)

opts_pedge = dict(
    colors=hatchcolour,
    linewidths=0.5
)

opts_phatch = dict(
    hatches=['//', ''],
    alpha=0
)

xtext, ytext = 0.15, 0.8
opts_text = dict(
    size=16,
    color=hatchcolour,
    ha='center',
    va='center',
    rotation=90
)

xmin, xmax, ymin, ymax = 0, np.pi / 2, 0, np.pi / 2
npoints = 2000
npoints_pert = 1000

plot_size = (8, 8)
plt.rc('font', size=16)
plt.rc('hatch', linewidth=0.3, color=hatchcolour)

#
# Supernova constraint
#

x = np.linspace(xmin, xmax, npoints)
y = np.linspace(ymin, ymax, npoints)
z = couplings.SNconstraint(*np.meshgrid(x, y))

fig, ax = plt.subplots(figsize=plot_size)
con = ax.contour(x, y, z, levels, **opts_sncontours)
conf = ax.contourf(x, y, z, levels, colors=fillcolours)
ax.clabel(con, **opts_sncontourlabels)

#
# Axion-electron coupling lines
#

z = np.abs(couplings.Ce(couplings.chi_3HDM(*np.meshgrid(x, y))))

cone = ax.contour(x, y, z, **opts_econtours)
ax.clabel(cone, **opts_econtourlabels)

#
# Perturbativity bounds
#


def not_edge(x, n):
    if x == 0.0:
        return 0.1 / n
    elif x == np.pi / 2:
        return (n - 1) / n * np.pi / 2
    else:
        return x


lims_pert = [not_edge(t, npoints_pert) for t in [xmin, xmax, ymin, ymax]]
xpert = np.linspace(lims_pert[0], lims_pert[1], npoints)
ypert = np.linspace(lims_pert[2], lims_pert[3], npoints)
zpert = couplings.yukawa_perturbative(*np.meshgrid(xpert, ypert))

ax.contour(xpert, ypert, zpert, [-0.5, 0.5], **opts_pedge)
ax.contourf(xpert, ypert, zpert, [-0.5, 0.5], **opts_phatch)
ax.text(xtext, ytext, 'Perturbative unitarity', **opts_text)

#
# Legend
#

leg_Ce = mpl.lines.Line2D([], [], color=ecolour, linestyle='--',
                          label=r'$C_{e}$')
leg_CSN = mpl.lines.Line2D([], [], color=sncolour,
                           label=r'$\sqrt{C_p^2+C_n^2}$')
# leg_CSNpatch = mpl.patches.Patch(facecolor=(0.75, 0.85, 1.0))
ax.legend(handles=[leg_CSN, leg_Ce],
          loc='lower left', bbox_to_anchor=(0.25, 0),
          fontsize=14, framealpha=0.9)

#
# Final plot options
#

ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))
ax.set_xlabel(r'$\beta_{1}$')
ax.set_ylabel(r'$\beta_{2}$', rotation=0, labelpad=15)
ax.tick_params(direction='in', which='both')

for ti, t in enumerate([ax.xaxis, ax.yaxis]):
    t.set_ticks_position('both')
    # d = (8, 8)[ti]
    # t.set_major_locator(tck.MultipleLocator(np.pi / d))
    # t.set_major_formatter(plt.FuncFormatter(multiple_formatter(denominator=d)))
    # t.set_minor_locator(tck.AutoMinorLocator(4))
    ticks = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, np.pi / 2]
    lab = ['0', '0.2', '0.4', '0.6', '0.8',
           '1', '1.2', '1.4', r'$\frac{\pi}{2}$']
    t.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, np.pi / 2])
    t.set_ticklabels(lab)

plt.tight_layout()

if save is True:
    plt.savefig(filename)
else:
    plt.show()
