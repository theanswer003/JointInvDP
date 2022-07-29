import numpy as np
import scipy.io as sio
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from net import Net
device = 'cuda' if torch.cuda.is_available() else 'cpu'


n_filters = 16
model = Net(n_filters=n_filters, nm=4).to(device)
model.load_state_dict(torch.load('saved_models/model.pth', map_location=device))
res = sio.loadmat('data/m_inv.mat')
z0 = res['z0']
y0 = res['y0']
z0 = torch.Tensor(z0).to(device)
y0 = torch.Tensor(y0).to(device)

gm_prior_min = torch.Tensor([0.38, 5.0, 1.0, 1.8]).to(device)
gm_prior_range = torch.Tensor([0.05, 4.0, 0.5, 0.5]).to(device)

n = 1000
gms = np.zeros([n, 4])
with torch.no_grad():
    for i in range(n):
        _, gm, _ = model(z0, y0)
        gm = gm_prior_range*gm + gm_prior_min
        gms[i] = gm.cpu().numpy()

data = pd.DataFrame({'Critical poro.': gms[:, 0], 'Coord. num.': gms[:, 1], 'm': gms[:, 2], 'n': gms[:, 3]})
# draw jointplot with kde kind
kdeplot = sns.jointplot(x='Critical poro.', y='Coord. num.', kind='kde', data=data, fill=True, levels=100)
kdeplot.ax_joint.set_xlim([0.38, 0.43])
kdeplot.ax_joint.set_ylim([5, 9])
plt.show()

kdeplot = sns.jointplot(x='m', y='n', kind='kde', data=data, fill=True, levels=100)
kdeplot.ax_joint.set_xlim([1.0, 1.5])
kdeplot.ax_joint.set_ylim([1.8, 2.3])
plt.show()