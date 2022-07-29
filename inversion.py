import time
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
from forward import *
from rpm import *
from utils import *
from params import *
from net import Net
np.random.seed(2022)
torch.manual_seed(2022)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_t = time.time()
nx = 120
ny = 120
nt = 224

n_filters = 16
model = Net(n_filters=n_filters, nm=4).to(device)
gm_prior_min = torch.Tensor([0.38, 5.0, 1.0, 1.8]).to(device)
gm_prior_range = torch.Tensor([0.05, 4.0, 0.5, 0.5]).to(device)

lr = 1e-3
criterion = nn.MSELoss()
n_epochs = 1000
alpha_seis = 50
alpha_res = 1
alpha_kl = 0.1
alpha_reg = 0.1

years = ['1996', '2000', '2003', '2006']
z0 = torch.randn((n_filters * 4 * 15 * 15 * 28)).to(device)
y0 = torch.randn((4, )).to(device)
Sc_base = torch.zeros((nx, ny, nt)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

total_loss = []
data_loss = []

for epoch in range(n_epochs):
    optimizer.zero_grad()
    m, gm, kl = model(z0, y0)
    gm = gm_prior_range*gm + gm_prior_min
    poro_inv = 0.4 * m[0, 0] # poro: prior range [0, 0.4]
    Sc_inv = m[0, 1:]
    loss = alpha_kl * kl
    d_loss = 0
    for i, year in enumerate(years):
        if i == 0:
            seis_pred = pred_seismic(poro_inv, Sc_base, gm[0], gm[1], thetas, wavelets)
            res_pred = pred_resistivity(poro_inv, Sc_base, gm[2], gm[3])
            reg = 0
        else:
            seis_pred = pred_seismic(poro_inv, Sc_inv[i-1], gm[0], gm[1], thetas, wavelets)
            res_pred = pred_resistivity(poro_inv, Sc_inv[i-1], gm[2], gm[3])
            reg = torch.mean(Sc_inv[i - 1] * (1 - Sc_inv[i - 1]))

        d_true = sio.loadmat(f'./data/d_true_{year}.mat')
        seis_true = d_true['d_seis']
        res_true = d_true['d_res']

        seis_true = torch.tensor(seis_true).to(device)
        res_true = torch.tensor(res_true).to(device)

        loss_seis = criterion(seis_pred, seis_true)
        loss_res = criterion(res_pred, res_true)


        d_loss += alpha_seis*loss_seis + alpha_res*loss_res
        loss += alpha_seis*loss_seis + alpha_res*loss_res + alpha_reg*reg

    loss.backward()
    optimizer.step()
    print(f'epoch: {epoch} | total loss: {loss:.6f}, data loss: {d_loss:.6f}')
    total_loss.append(loss.item())
    data_loss.append(d_loss.item())


# save data
sio.savemat('data/m_inv.mat',
            mdict={'poro_inv': poro_inv.cpu().detach().numpy(),
                   'Sc_inv': Sc_inv.cpu().detach().numpy(),
                   'z0': z0.cpu().numpy(),
                   'y0': y0.cpu().numpy(),
                   'gm': gm.cpu().detach().numpy()})

sio.savemat('data/training_history.mat', mdict={'total_loss': np.array(total_loss),
                                                'data_loss': np.array(data_loss)})
torch.save(model.state_dict(), 'saved_models/model.pth')

end_t = time.time()
print('Done. Time elapsed: ', end_t - start_t)