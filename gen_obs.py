import time
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter
import torch
from rpm import *
from forward import *
from utils import *
from params import *
device = 'cuda' if torch.cuda.is_available() else 'cpu'


start_t = time.time()
# monitoring years
critporo = 0.4
coordnum = 7
Rw = 0.17
m = 1.3
n = 2.0

years = ['1996', '2000', '2003', '2006']

# sleipner reservoir simulation data
simdata  = sio.loadmat('data/sleipner_ressim_data_time.mat')
poro = simdata['poro'].astype(np.float32)
Scs = simdata['Sc'].astype(np.float32)
Scs[Scs<0] = 0
poro = torch.tensor(poro).to(device)
Scs = torch.tensor(Scs).to(device)

for i, year in enumerate(years):
    print(year)
    Sc = Scs[:, :, :, i]
    Sw = 1 - Sc

    Sflc = torch.stack([Sw, Sc]).to(device)
    Kmat, Gmat, rhob_mat = mineral_mix(Kminc, Gminc, Rhominc, Volminc)
    Kfl, rhob_fl = fluid_mix(Kflc, Rhoflc, Sflc, patchy)
    rhob = DensityModel(poro, rhob_mat, rhob_fl)
    Vp, Vs = SoftsandModel(poro, rhob, Kmat, Gmat, Kfl, critporo, coordnum, press)

    nx, ny, nz = Vp.shape
    nthetas = len(thetas)
    d_seis = torch.zeros((nthetas, nx, ny, nz - 1)).to(device)
    refl_coef = shuey(Vp[:, :, :-1], Vs[:, :, :-1], rhob[:, :, :-1],
                      Vp[:, :, 1:], Vs[:, :, 1:], rhob[:, :, 1:], thetas)
    refl_coef += 0.02*torch.randn(*refl_coef.shape, device=device)
    for i in range(nthetas):
        d_seis[i] = torch.einsum('kji,li->kjl', refl_coef[i], wavelets[i])

    d_res = torch.log10(poro**(-m)*Rw*(1.0-Sc)**(-n))

    # d_seis += 1.0*d_seis*torch.randn(*d_seis.shape, device=device)
    d_res += 0.5*torch.randn(*d_res.shape, device=device)
    Vp = Vp.cpu().numpy()
    Vs = Vs.cpu().numpy()
    rhob = rhob.cpu().numpy()
    d_seis = d_seis.cpu().numpy()
    d_res = d_res.cpu().numpy()
    d_res = gaussian_filter(d_res, sigma=3)


    sio.savemat(f'data/d_true_{year}.mat', mdict={'Vp': Vp,
                                                  'Vs': Vs,
                                                  'rhob': rhob,
                                                  'd_seis': d_seis,
                                                  'd_res': d_res})


print('Done.')
