import numpy as np
import torch
from utils import ricker, convmtx
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# rock physics parameters
Kqtz = 36.6
Gqtz = 44
rhob_qtz = 2.65
Kcly = 21
Gcly = 9
rhob_cly = 2.5

Kw = 3.06
rhob_w = 1.08
Kg = 0.10
rhob_g = 0.72

Vcly = 0.1
Vqtz = 1 - Vcly
patchy = 0
press = 0.02


Kminc = torch.tensor([Kqtz, Kcly]).to(device)
Gminc = torch.tensor([Gqtz, Gcly]).to(device)
Rhominc = torch.tensor([rhob_qtz, rhob_cly]).to(device)
Volminc = torch.tensor([Vqtz, Vcly]).to(device)
Kflc = torch.tensor([Kw, Kg]).to(device)
Rhoflc = torch.tensor([rhob_w, rhob_g]).to(device)


# wavelet
thetas = [12, 24, 36]
wavelets = []
# near stack
wavelet, _ = ricker(30, 0.001)
W = convmtx(wavelet, 223)
nsW = len(wavelet) // 2
W = W[:, nsW:-nsW].T
wavelets.append(W)

# mid stack
wavelet, _ = ricker(25, 0.001)
W = convmtx(wavelet, 223)
nsW = len(wavelet) // 2
W = W[:, nsW:-nsW].T
wavelets.append(W)

# far stack
wavelet, _ = ricker(20, 0.001)
W = convmtx(wavelet, 223)
nsW = len(wavelet) // 2
W = W[:, nsW:-nsW].T
wavelets.append(W)

wavelets = np.array(wavelets)
wavelets = torch.tensor(wavelets).to(device)