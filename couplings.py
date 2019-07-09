import numpy as np


def scalefactor(units):
    switcher = {
        'eV': 1,
        'keV': 1e3,
        'MeV': 1e6,
        'GeV': 1e9
    }
    return switcher.get(units)


def Csupernova(Cp, Cn):
    ''' SNcostraint in Federico's notes'''
    return np.sqrt(np.power(Cp, 2) + np.power(Cn, 2))


def Gsupernova(Cp, Cn, fa, units='eV'):
    mn = 9.38e8
    f = fa * scalefactor(units)
    return mn * Csupernova(Cp, Cn) / f


def Cp(Cu, Cd, Cc, Cs, Cb, Ct):
    heavy = - 0.038 * Cs - 0.012 * Cc - 0.009 * Cb - 0.0035 * Ct
    return - 0.47 + 0.88 * Cu - 0.39 * Cd + heavy


def Cn(Cu, Cd, Cc, Cs, Cb, Ct):
    heavy = - 0.038 * Cs - 0.012 * Cc - 0.009 * Cb - 0.0035 * Ct
    return - 0.02 - 0.39 * Cu + 0.88 * Cd + heavy


def Cq_3HDM(chi):
    chi1, chi2 = chi[0], chi[1]
    couplings = [x / (chi1 - chi2)
                 for x in [chi1, -chi2, chi1, -chi2, chi2, -chi1]]
    return couplings


def Ce(chi):
    return chi[2] / 3


def Ge(Ce, fa, units='eV'):
    me = 5.11e5
    f = fa * scalefactor(units)
    return me * Ce / f


def chi_3HDM(beta1, beta2):
    [cc1, cc2] = [np.power(np.cos(b), 2) for b in [beta1, beta2]]
    [ss1, ss2] = [np.power(np.sin(b), 2) for b in [beta1, beta2]]

    chi1 = 3 * ss1 * cc2 + 2 * ss2
    chi2 = -3 * cc1 * cc2 - ss2
    chi3 = cc2 * (ss1 - 2 * cc1)

    return [chi1, chi2, chi3]


def SNconstraint(beta1, beta2):
    Cquarks = Cq_3HDM(chi_3HDM(beta1, beta2))
    Cprot, Cneut = [func(*Cquarks) for func in [Cp, Cn]]
    return Csupernova(Cprot, Cneut)


@np.vectorize
def yukawa_perturbative(beta1, beta2):
    ytau = 0.0102 / np.sin(beta2)
    ytop = 0.9951 / (np.sin(beta1) * np.cos(beta2))
    ybot = 0.0240 / (np.cos(beta1) * np.cos(beta2))

    y_unitarity = np.array([2 * np.sqrt(2) * ytau, ytop, ybot])

    alltrue = np.all([y**2 < 16 * np.pi / 3 for y in y_unitarity])
    # alltrue = np.all([y < 4 * np.pi for y in np.array([ytau, ytop, ybot])])

    return alltrue
