import math
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def DensityModel(poro, rhob_mat, rhob_fld):
    rhob = (1 - poro)*rhob_mat + poro*rhob_fld
    return rhob


def SoftsandModel(poro, rhob, Kmat, Gmat, Kfl, critporo, coordnum, press):
    Poisson = (3 * Kmat - 2 * Gmat) / (6 * Kmat + 2 * Gmat)
    KHM = ((coordnum ** 2 * (1 - critporo) ** 2 * Gmat ** 2 * press) / (18 * math.pi ** 2 * (1 - Poisson) ** 2)) ** (
                1 / 3)
    GHM = (5 - 4 * Poisson) / (10 - 5 * Poisson) * ((3 * coordnum ** 2 * (1 - critporo) ** 2 * Gmat ** 2 * press) / (
                2 * math.pi ** 2 * (1 - Poisson) ** 2)) ** (1 / 3)

    # Modified Hashin-Shtrikmann lower bounds
    Kdry = 1. / ((poro / critporo) / (KHM + 4 / 3 * GHM) + (1 - poro / critporo) / (Kmat + 4 / 3 * GHM)) - 4 / 3 * GHM
    psi = (9 * KHM + 8 * GHM) / (KHM + 2 * GHM)
    Gdry = 1. / ((poro / critporo) / (GHM + 1 / 6 * psi * GHM) + (1 - poro / critporo) / (
                Gmat + 1 / 6 * psi * GHM)) - 1 / 6 * psi * GHM

    # Gassmann
    [Ksat, Gsat] = GassmannModel(poro, Kdry, Gdry, Kmat, Kfl)

    # Velocities
    Vp = torch.sqrt((Ksat + 4 / 3 * Gsat) / rhob)
    Vs = torch.sqrt(Gsat / rhob)

    return Vp, Vs


def GassmannModel(poro, Kdry, Gdry, Kmat, Kfl):
    # Bulk modulus of saturated rock
    Ksat = Kdry + ((1 - Kdry / Kmat) ** 2) / (poro / Kfl + (1 - poro) / Kmat - Kdry / (Kmat ** 2))
    # Shear modulus of saturated rock
    Gsat = Gdry

    return Ksat, Gsat


def mineral_mix(Kminc, Gminc, rhob_minc, Vminc):
    nmin = Kminc.shape[0]
    KV = torch.zeros(Vminc.shape[1:])
    GV = torch.zeros(Vminc.shape[1:])
    KR = torch.zeros(Vminc.shape[1:])
    GR = torch.zeros(Vminc.shape[1:])
    rhob = torch.zeros(Vminc.shape[1:])
    for i in range(nmin):
        KV = KV + Kminc[i]*Vminc[i]
        GV = GV + Gminc[i]*Vminc[i]
        KR = KR + 1.0/Kminc[i]*Vminc[i]
        GR = GR + 1.0/Gminc[i]*Vminc[i]
        rhob = rhob + rhob_minc[i]*Vminc[i]

    KR = 1.0 / KR
    GR = 1.0 / GR

    K = (KV + KR) / 2.0
    G = (GV + GR) / 2.0

    return K, G, rhob


def fluid_mix(Kfld, rhob_fld, Sfld, patchy):
    nfld = Sfld.shape[0]
    K = torch.zeros(Sfld.shape[1:]).to(device)
    rhob = torch.zeros(Sfld.shape[1:]).to(device)
    if patchy == 0:
        # Reuss average for fluid
        for i in range(nfld):
            K = K + 1.0/Kfld[i]*Sfld[i]
        K = 1.0 / K
    else:
        # Voigt average for fluid
        for i in range(nfld):
            K = K + Kfld[i]*Sfld[i]
    # linear average for fluid density
    for i in range(nfld):
        rhob = rhob + rhob_fld[i] * Sfld[i]

    return K, rhob