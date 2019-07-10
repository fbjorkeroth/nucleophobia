import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# import matplotlib.ticker as tck

# from multiple import multiple_formatter
import couplings

print('Making plot...')

save = False
filename = 'images/Allowed_new.pdf'

npoints = 200
npoints_pert = 100

#
# Choose contours and colours

snlevels = [0, 0.025, 0.05, 0.1, 10]
elevels = [0, 0.001, 0.01, 0.1, 10]

fillcolours = [(0.1, 0.25, 0.55),
               (0.2, 0.45, 0.6),
               (0.4, 0.7, 0.8),
               (1, 1, 1)]

sncolour = 'k'
ecolour = 'gold'
hatchcolour = '0.3'

opts_sncontours = dict(
    colors=sncolour,
    linewidths=1,
    levels=snlevels
)

opts_sncontourlabels = dict(
    colors=sncolour,
    fmt='%g',
    fontsize=12,
    manual=[(0.75, 1.15),  # 0.1 left
            (0.85, 1.16),  # 0.05 left
            (1.1, 1.4),  # 0.025 left
            (1.4, 1.1),  # 0.1 right
            (1.4, 1.2),  # 0.05 right
            (1.4, 1.3)]  # 0.025 right
)

opts_econtours = dict(
    colors=ecolour,
    linewidths=1,
    # linestyles='dashed',
    levels=elevels
)

opts_econtourlabels = dict(
    colors=ecolour,
    fmt='%g',
    fontsize=12,
    manual=[
        (0.2, 1.2),  # 0.1 left
        (0.3, 1.4),  # 0.01 left
        (0.5, 1.5),  # 0.001 left
        (1.2, 1.35),  # 0.01 right
        (1.2, 1.5),  # 0.001 right
        (1.4, 1)]  # 0.1 right
)

opts_pedge = dict(
    colors=hatchcolour,
    linewidths=0.5
)

opts_phatch = dict(
    hatches=['///', ''],
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

opts_canonical_contours = dict(
    colors='k',
    linestyles=':'
)

label_ksvz = (r'$C^\mathrm{KSVZ}_{N}$', [(0.45, 0.75)])
label_dfszmin = (r'$C^\mathrm{DFSZ}_{N, \mathrm{min}}$', [(0.6, 1.0)])
label_dfszmax = (r'$C^\mathrm{DFSZ}_{N, \mathrm{max}}$', [(0.4, 0.6)])


def opts_canonical_labels(lab, pos):
    return dict(
        colors='k',
        fmt=lab,
        fontsize=12,
        manual=pos
    )


xmin, xmax, ymin, ymax = 0, np.pi / 2, 0, np.pi / 2
plot_size = (8, 8)
plt.rc('font', size=16)
plt.rc('hatch', linewidth=0.3, color=hatchcolour)

fig, ax = plt.subplots(figsize=plot_size)

#
# Supernova constraint

x = np.linspace(xmin, xmax, npoints)
y = np.linspace(ymin, ymax, npoints)
z = couplings.SNconstraint(*np.meshgrid(x, y))

sncon = ax.contour(x, y, z, snlevels, **opts_sncontours)
snconf = ax.contourf(x, y, z, snlevels, colors=fillcolours)
ax.clabel(sncon, **opts_sncontourlabels)

#
# Canonical KSVZ and DFSZ (C_N)

ksvz = ax.contour(x, y, z, [0, 0.48, 10], **opts_canonical_contours)
ax.clabel(ksvz, **opts_canonical_labels(*label_ksvz))

dfszmin = ax.contour(x, y, z, [0, 0.24, 10], **opts_canonical_contours)
ax.clabel(dfszmin, **opts_canonical_labels(*label_dfszmin))
dfszmax = ax.contour(x, y, z, [0, 0.66, 10], **opts_canonical_contours)
ax.clabel(dfszmax, **opts_canonical_labels(*label_dfszmax))
dfszfill = ax.contourf(x, y, z, [0, 0.24, 0.66, 10],
                       colors=['1', '0.1', '1'],
                       alpha=0.05)

#
# Axion-electron coupling lines

z = np.abs(couplings.Ce(couplings.chi_3HDM(*np.meshgrid(x, y))))

econ = ax.contour(x, y, z, **opts_econtours)
ax.clabel(econ, **opts_econtourlabels)

# Canonical DFSZ (C_e)
dfsze = ax.contour(x, y, z, [0, 1.0 / 6, 10], colors=ecolour, linestyles=':')
ax.clabel(dfsze, manual=[(0.7, 1.1)],
          colors=ecolour,
          fmt=r'$C^\mathrm{DFSZ}_{e, \mathrm{PDG}}$',
          fontsize=12)

#
# Perturbativity bounds


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

leg_Ce = mpl.lines.Line2D([], [], color=ecolour, linewidth=1,
                          label=r'$C_{e}$')
leg_CN = mpl.lines.Line2D([], [], color=sncolour, linewidth=1,
                          label=r'$C_{N}$')
# leg_CNpatch = mpl.patches.Patch(facecolor=(0.75, 0.85, 1.0))
ax.legend(handles=[leg_CN, leg_Ce],
          loc='center', bbox_to_anchor=(0.9, 0.08),
          fontsize=16, framealpha=0.9)

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
           '1', '1.2', '1.4', r'$\pi/2$']
    t.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, np.pi / 2])
    t.set_ticklabels(lab)

plt.tight_layout()

if save is True:
    plt.savefig(filename)
else:
    plt.show()
