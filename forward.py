import torch
from utils import shuey
from rpm import *
from params import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def pred_seismic(poro, Sc, critporo, coordnum, thetas, wavelets):
    Sw = 1 - Sc
    Sflc = torch.stack([Sw, Sc]).to(device)
    Kmat, Gmat, rhob_mat = mineral_mix(Kminc, Gminc, Rhominc, Volminc)
    Kfl, rhob_fl = fluid_mix(Kflc, Rhoflc, Sflc, patchy)
    rhob = DensityModel(poro, rhob_mat, rhob_fl)
    Vp, Vs = SoftsandModel(poro, rhob, Kmat, Gmat, Kfl, critporo, coordnum, press)

    nx, ny, nz = Vp.shape
    nthetas = len(thetas)
    seismic = torch.zeros((nthetas, nx, ny, nz-1)).to(device)
    refl_coef = shuey(Vp[:, :, :-1], Vs[:, :, :-1], rhob[:, :, :-1],
                      Vp[:, :, 1:], Vs[:, :, 1:], rhob[:, :, 1:], thetas)
    for i in range(nthetas):
        seismic[i] = torch.einsum('kji,li->kjl', refl_coef[i], wavelets[i])

    return seismic


def pred_resistivity(poro, Sc, m, n):
    Rw = 0.17
    Rs = poro**(-m)*Rw*(1.0-Sc)**(-n)

    return torch.log10(Rs)
