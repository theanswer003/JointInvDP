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

lr = 1e-3
criterion = nn.MSELoss()
n_epochs = 300
alpha_seis = 50
alpha_res = 1
alpha_kl = 0.1
alpha_reg = 0.1

years = ['1996', '2000', '2003', '2006']
m = 0.5*torch.ones((1, 4, 120, 120, 224))
m = torch.tensor(m, requires_grad=True, device='cuda')
m = torch.nn.Parameter(m)
Sc_base = torch.zeros((nx, ny, nt)).to(device)
optimizer = torch.optim.Adam([m,], lr=lr)

total_loss = []
data_loss = []

for epoch in range(n_epochs):
    optimizer.zero_grad()
    poro_inv = 0.4*m[0, 0]  # poro: prior range [0, 0.4]
    Sc_inv = m[0, 1:]
    Sw_inv = 1 - Sc_inv
    loss = 0
    d_loss = 0
    for i, year in enumerate(years):
        if i == 0:
            seis_pred = pred_seismic(poro_inv, Sc_base, 0.4, 7, thetas, wavelets)
            res_pred = pred_resistivity(poro_inv, Sc_base, 1.3, 2.0)
            reg = 0
        else:
            seis_pred = pred_seismic(poro_inv, Sc_inv[i-1], 0.4, 7, thetas, wavelets)
            res_pred = pred_resistivity(poro_inv, Sc_inv[i-1], 1.3, 2.0)
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
sio.savemat('data/m_inv_nonet.mat',
            mdict={'poro_inv': poro_inv.cpu().detach().numpy(),
                   'Sc_inv': Sc_inv.cpu().detach().numpy()})

sio.savemat('data/training_history_nonet.mat', mdict={'total_loss': np.array(total_loss),
                                                'data_loss': np.array(data_loss)})
torch.save(model.state_dict(), 'saved_models/model_nonet.pth')

end_t = time.time()
print('Done. Time elapsed: ', end_t - start_t)