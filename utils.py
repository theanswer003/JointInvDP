from functools import wraps
import math
import numbers
import numpy as np
import scipy.linalg
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def ricker(f, dt):
    nw = 2.2 / f / dt
    nw = 2 * np.math.floor(nw/2) + 1
    nc = np.math.floor(nw/2)
    k = np.arange(1, nw+1)
    alpha = (nc - k + 1) * f * dt * np.pi
    beta = alpha**2
    w = (1.0 - 2.0*beta) * np.exp(-beta)
    tw = -(nc + 1 - np.arange(1, nw+1)) * dt

    return w, tw


def convmtx(h, n):
    col_1 = np.r_[h[0], np.zeros(n - 1)]
    row_1 = np.r_[h, np.zeros(n - 1)]

    return scipy.linalg.toeplitz(col_1, row_1)


def vectorize(func):
    """
    Decorator to make sure the inputs are arrays. We also add a dimension
    to theta1 to make the functions work in an 'outer product' way.

    Takes a reflectivity function requiring Vp, Vs, and RHOB for 2 rocks
    (upper and lower), plus incidence angle theta1, plus kwargs. Returns
    that function with the arguments transformed to ndarrays.
    """
    @wraps(func)
    def wrapper(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, **kwargs):

        # vp1 = torch.tensor(vp1, dtype=float)
        # vs1 = np.asanyarray(vs1, dtype=float) + 1e-12  # Prevent singular matrix.
        # rho1 = np.asanyarray(rho1, dtype=float)
        # vp2 = np.asanyarray(vp2, dtype=float)
        # vs2 = np.asanyarray(vs2, dtype=float) + 1e-12  # Prevent singular matrix.
        # rho2 = np.asanyarray(rho2, dtype=float)

        new_shape = [-1] + vp1.ndim * [1]
        theta1 = theta1.reshape(*new_shape)
        if (np.nan_to_num(theta1) > np.pi/2.).any():
            raise ValueError("Incidence angle theta1 must be less than 90 deg.")

        return func(vp1, vs1, rho1, vp2, vs2, rho2, theta1, **kwargs)
    return wrapper


def preprocess(func):
    """
    Decorator to preprocess arguments for the reflectivity equations.

    Takes a reflectivity function requiring Vp, Vs, and RHOB for 2 rocks
    (upper and lower), plus incidence angle theta1, plus kwargs. Returns
    that function with some arguments transformed.
    """
    @wraps(func)
    def wrapper(vp1, vs1, rho1, vp2, vs2, rho2, theta1=0, **kwargs):

        # Interpret tuple for theta1 as a linspace.
        if isinstance(theta1, tuple):
            if len(theta1) == 2:
                start, stop = theta1
                theta1 = np.linspace(start, stop, num=stop+1)
            elif len(theta1) == 3:
                start, stop, step = theta1
                steps = (stop / step) + 1
                theta1 = np.linspace(start, stop, num=steps)
            else:
                raise TypeError("Expected 2 or 3 parameters for theta1 expressed as range.")

        # Convert theta1 to radians and complex numbers.
        theta1 = np.radians(theta1).astype(complex)

        return func(vp1, vs1, rho1, vp2, vs2, rho2, theta1, **kwargs)
    return wrapper


@preprocess
@vectorize
def shuey(vp1, vs1, rho1, vp2, vs2, rho2, theta1):

    theta1 = torch.real(torch.tensor(theta1)).to(device)
    drho = rho2-rho1
    dvp = vp2-vp1
    dvs = vs2-vs1
    rho = (rho1+rho2)/2.0
    vp = (vp1+vp2)/2.0
    vs = (vs1+vs2)/2.0

    # Compute three-term reflectivity
    r0 = 0.5 * (dvp/vp + drho/rho)
    g = 0.5 * dvp/vp - 2 * (vs**2/vp**2) * (drho/rho + 2 * dvs/vs)
    f = 0.5 * dvp/vp

    term1 = r0
    term2 = g * torch.sin(theta1)**2
    term3 = f * (torch.tan(theta1)**2 - torch.sin(theta1)**2)

    return term1 + term2 + term3
