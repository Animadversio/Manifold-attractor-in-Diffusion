# load MNIST
import torch
import numpy as np
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
#%%
dataset = MNIST(root='~/Datasets', download=True, transform=transforms.ToTensor())
dataset_test = MNIST(root='~/Datasets', train=False, download=True, transform=transforms.ToTensor())
#%%
dataset[0][0].shape
Xtsr = torch.stack([dataset[i][0] for i in
                    range(len(dataset))], dim=0)
ytsr = torch.tensor([dataset[i][1] for i in
                     range(len(dataset))], dtype=torch.long)
Xtsr_test = torch.stack([dataset_test[i][0] for i in
                            range(len(dataset_test))], dim=0)
ytsr_test = torch.tensor([dataset_test[i][1] for i in
                            range(len(dataset_test))], dtype=torch.long)
#%%
from gmm_special_diffusion_lib import GMM_density_torch, GMM_logprob_torch, GMM_scores_torch
#%%
Xmat = Xtsr.reshape(Xtsr.shape[0], -1)
Xmat_test = Xtsr_test.reshape(Xtsr_test.shape[0], -1)
#%%
# scores = GMM_scores_torch(Xmat, sigma, pnts + Delta * perturbvec)
#%%
import torch.nn.functional as F
def GMM_scores_torch(mus, sigma, x):
    sigma2 = sigma**2
    res = x[:, None, :] - mus[None, :, :]  # [x batch, mu, space dim]
    dist2 = torch.sum(res ** 2, dim=-1)  # [x batch, mu]
    participance = F.softmax(- dist2 / sigma2 / 2, dim=1)  # [x batch, mu]
    scores = - torch.einsum("ij,ijk->ik", participance, res) / sigma2  # [x batch, space dim]
    return scores


mus_ssq = torch.sum(Xmat ** 2, dim=1)  # [mu,]
def GMM_scores_torch_eff(mus, sigma, x, mus_ssq=None):
    """

    :param mus:  P x d
    :param sigma: scalar
    :param x: B x d ( normally B = 1)
    :param mus_ssq: P
    :return:
    """
    if mus_ssq is None:
        mus_ssq = torch.sum(mus ** 2, dim=1)
    sigma2 = sigma**2
    dotprod = x @ mus.t()  # [x batch, P pnts]
    participance = F.softmax((dotprod - mus_ssq[None, :] / 2) / sigma2, dim=1)  # [x batch, P pnts]
    scores = - (x - (participance @ mus)) / sigma2  # [x batch, space dim]
    # scores = - torch.einsum("BP,Pd->Bd", participance, mus) / sigma2  # [x batch, space dim]
    return scores


def GMM_exp_x_torch_eff(mus, sigma, x, mus_ssq=None):
    """

    :param mus:  P x d
    :param sigma: scalar
    :param x: B x d ( normally B = 1)
    :param mus_ssq: P
    :return:
    """
    if mus_ssq is None:
        mus_ssq = torch.sum(mus ** 2, dim=1)
    sigma2 = sigma**2
    dotprod = x @ mus.t()  # [x batch, P pnts]
    participance = F.softmax((dotprod - mus_ssq[None, :] / 2) / sigma2, dim=1)  # [x batch, P pnts]
    expexted_x = (participance @ mus)  # [x batch, space dim]
    # scores = - torch.einsum("BP,Pd->Bd", participance, mus) / sigma2  # [x batch, space dim]
    return expexted_x

from scipy.special import softmax
def GMM_scores_eff(mus, sigma, x, mus_ssq=None):
    """

    :param mus:  P x d
    :param sigma: scalar
    :param x: B x d ( normally B = 1)
    :param mus_ssq: P
    :return:
    """
    if mus_ssq is None:
        mus_ssq = np.sum(mus ** 2, axis=1)
    sigma2 = sigma**2
    dotprod = x @ mus.T  # [x batch, P pnts]
    participance = softmax((dotprod - mus_ssq[None, :] / 2) / sigma2, axis=1)  # [x batch, P pnts]
    scores = - (x - (participance @ mus)) / sigma2  # [x batch, space dim]
    # scores = - torch.einsum("BP,Pd->Bd", participance, mus) / sigma2  # [x batch, space dim]
    return scores

def f_edm_ode(sigma, x):
    return - sigma * GMM_scores_torch_eff(Xmat, sigma, x.view(-1, ndim), mus_ssq=mus_ssq)

def f_edm_ode_np(sigma, x):
    return - sigma * GMM_scores_eff(Xmat_np, sigma, x.T, mus_ssq=mus_ssq_np).T

def f_edm_ode_smooth(sigma, x, Delta=1, nquad=50):
    quad_perturb = torch.randn(nquad, ndim)
    quad_perturb /= torch.norm(quad_perturb, dim=1, keepdim=True)
    quad_perturb *= Delta
    perturb_x = x.view(-1, ndim)[:, None] + quad_perturb[None, :, :]
    score_perturb = GMM_scores_torch_eff(Xmat, sigma, perturb_x.view(-1, ndim), mus_ssq=mus_ssq)
    score_smooth = score_perturb.view(-1, nquad, ndim).mean(dim=1)
    return - sigma * score_smooth

def f_edm_ode_smooth_particip(sigma, x, Delta=1, nquad=50):
    quad_perturb = torch.randn(nquad, ndim)
    quad_perturb /= torch.norm(quad_perturb, dim=1, keepdim=True)
    quad_perturb *= Delta
    perturb_x = x.view(-1, ndim)[:, None] + quad_perturb[None, :, :]
    exp_x_perturb = GMM_exp_x_torch_eff(Xmat, sigma, perturb_x.view(-1, ndim), mus_ssq=mus_ssq)
    exp_x_smooth = exp_x_perturb.view(-1, nquad, ndim).mean(dim=1)
    return  (x - exp_x_smooth) / sigma
#%%
Xmat_np = Xmat.numpy()
mus_ssq_np = (Xmat_np ** 2).sum(axis=1)
#%%
import math
from torchdiffeq import odeint
from scipy.integrate import solve_ivp
sigma_max, sigma_min = 80, 0.002
t_eval = np.logspace(np.log10(sigma_max-0.01),
                     np.log10(sigma_min+0.0001), 30)
ndim = 784
xT = np.random.randn(ndim) * sigma_max
sol = solve_ivp(lambda sigma, x: f_edm_ode_np(sigma, x),
            (sigma_max, sigma_min), xT, method="RK45",
            vectorized=True, t_eval=t_eval)
sol.y[:, -1], sol
#%%
plt.figure(figsize=(8, 8))
plt.imshow(sol.y[:, -1].reshape(28, 28))
plt.show()

#%%
import math
from torchdiffeq import odeint
sigma_max, sigma_min = 80, 0.002
t_eval = np.logspace(np.log10(sigma_max-0.01),
                     np.log10(sigma_min+0.0001), 20)
t_eval = torch.tensor(t_eval)
# t_space = np.linspace(sigma_max, sigma_min, 200)
# t_eval = torch.tensor(t_space)
ndim = 784
nbatch = 25
xT = torch.randn(ndim)[None] * sigma_max
xT = torch.randn(nbatch, ndim) * sigma_max
sol = odeint(f_edm_ode, xT, t=t_eval, method="euler")
#%%
mtg_tsr = make_grid(sol[-1, :].\
    reshape(nbatch, 1, 28, 28).clamp(0,1), nrow=5)
plt.figure(figsize=(8, 8))
plt.imshow(mtg_tsr.permute(1, 2, 0))
plt.show()
#%%

sigma_max, sigma_min = 80, 0.002
t_eval = np.logspace(np.log10(sigma_max-0.01),
                     np.log10(sigma_min+0.0001), 20)
t_eval = torch.tensor(t_eval)
# t_space = np.linspace(sigma_max, sigma_min, 200)
# t_eval = torch.tensor(t_space)
ndim = 784
nbatch = 9
# xT = torch.randn(ndim)[None] * sigma_max
xT = torch.randn(nbatch, ndim) * sigma_max
#%%
Delta = 100
nquad = 64
sol_smooth = odeint(lambda t, x:
        f_edm_ode_smooth(t, x, Delta=t * Delta, nquad=nquad),
              xT, t=t_eval, method="rk4")
sol_smooth_particip = odeint(lambda t, x:
        f_edm_ode_smooth_particip(t, x, Delta=t * Delta, nquad=nquad),
              xT, t=t_eval, method="rk4")
sol = odeint(lambda t, x: f_edm_ode(t, x, ),
             xT, t=t_eval, method="rk4")
mtg_tsr = make_grid(sol[-1, :].\
    reshape(nbatch, 1, 28, 28).clamp(0,1), nrow=int(math.sqrt(nbatch)))
mtg_smooth_tsr = make_grid(sol_smooth[-1, :].\
    reshape(nbatch, 1, 28, 28).clamp(0,1), nrow=int(math.sqrt(nbatch)))
mtg_smooth_particip_tsr = make_grid(sol_smooth_particip[-1, :].\
    reshape(nbatch, 1, 28, 28).clamp(0,1), nrow=int(math.sqrt(nbatch)))
plt.subplots(1, 3, figsize=(20, 8))
plt.subplot(131)
plt.imshow(mtg_tsr.permute(1, 2, 0))
plt.title("DeltaGMM score")
plt.subplot(132)
plt.imshow(mtg_smooth_tsr.permute(1, 2, 0))
plt.title("Quadrature smoothed DeltaGMM score")
plt.subplot(133)
plt.imshow(mtg_smooth_particip_tsr.permute(1, 2, 0))
plt.title("Quadrature smoothed DeltaGMM score (participance)")
plt.suptitle(f"Random Smoothing Quadrature Delta={Delta}, nquad={nquad}", fontsize=16)
plt.show()


#%%
import seaborn as sns
import pandas as pd
def vectsr2imgrid(vectsr):
    nbatch = vectsr.shape[0]
    mtg_tsr = make_grid(vectsr. \
                        reshape(nbatch, 1, 28, 28).clamp(0, 1), nrow=int(math.sqrt(nbatch)))
    return mtg_tsr.permute(1, 2, 0)


def dist2nearest(Xmat, x):
    inprod = Xmat @ x.T
    dist2 = torch.sum(Xmat ** 2, dim=1)[:, None] - 2 * inprod + torch.sum(x ** 2, dim=1)[None, :]
    mindist, minidx = dist2.min(dim=0)
    return mindist, minidx
#%%
ndim = 784
nbatch = 16
# xT = torch.randn(ndim)[None] * sigma_max
xT = torch.randn(nbatch, ndim) * sigma_max
#%%
Delta = 15
nquad = 64
sol = odeint(lambda t, x: f_edm_ode(t, x, ),
             xT, t=t_eval, method="rk4")
sol_smooth = odeint(lambda t, x:
        f_edm_ode_smooth(t, x, Delta=Delta, nquad=nquad),
              xT, t=t_eval, method="rk4")
sol_smooth_particip = odeint(lambda t, x:
        f_edm_ode_smooth_particip(t, x, Delta=Delta, nquad=nquad),
              xT, t=t_eval, method="rk4")
#%%
Delta = 100
nquad = 64
sol = odeint(lambda t, x: f_edm_ode(t, x, ),
             xT, t=t_eval, method="rk4")
sol_smooth = odeint(lambda t, x:
        f_edm_ode_smooth(t, x, Delta=np.sqrt(t) * Delta, nquad=nquad),
              xT, t=t_eval, method="rk4")
sol_smooth_particip = odeint(lambda t, x:
        f_edm_ode_smooth_particip(t, x, Delta=np.sqrt(t) * Delta, nquad=nquad),
              xT, t=t_eval, method="rk4")
#%%
plt.subplots(1, 3, figsize=(20, 8))
plt.subplot(131)
plt.imshow(vectsr2imgrid(sol[-1, :]))
plt.title("DeltaGMM score")
plt.subplot(132)
plt.imshow(vectsr2imgrid(sol_smooth[-1, :]))
plt.title("Quadrature smoothed DeltaGMM score")
plt.subplot(133)
plt.imshow(vectsr2imgrid(sol_smooth_particip[-1, :]))
plt.title("Quadrature smoothed DeltaGMM score (participance)")
plt.suptitle(f"Random Smoothing Quadrature Delta={Delta}, nquad={nquad}", fontsize=16)
plt.tight_layout()
plt.show()
#%%
# scatter / strip plot of distance to nearest
df_col = []
for label, vec in zip(["delta", "smooth", "smooth_particip"],
        (sol[-1, :], sol_smooth[-1, :], sol_smooth_particip[-1, :])):
    mindist, minidx = dist2nearest(Xmat, vec)
    mindist_test, minidx_test = dist2nearest(Xmat_test, vec)
    df_col.append(pd.DataFrame({"mindist": mindist, "minidx": minidx, "label": label, "split": "train"}))
    df_col.append(pd.DataFrame({"mindist": mindist_test, "minidx": minidx_test, "label": label, "split": "test"}))
df = pd.concat(df_col, axis=0)

plt.subplots(1,2,figsize=(8, 8))
ax = plt.subplot(121)
sns.stripplot(x="label", y="mindist", color="blue",
              data=df[df.split == "train"], alpha=0.9, jitter=0.15, ax=ax)
plt.title("Distance to training samples")
ax = plt.subplot(122)
sns.stripplot(x="label", y="mindist", color="red",
                data=df[df.split == "test"], alpha=0.9, jitter=0.15, ax=ax)
plt.title("Distance to test samples")
plt.show()

#%%
print(dist2nearest(Xmat, sol[-1, :]))
print(dist2nearest(Xmat, sol_smooth[-1, :]))
print(dist2nearest(Xmat, sol_smooth_particip[-1, :]))
#%%
print(dist2nearest(Xmat_test, sol[-1, :]))
print(dist2nearest(Xmat_test, sol_smooth[-1, :]))
print(dist2nearest(Xmat_test, sol_smooth_particip[-1, :]))
#%%
plt.figure(figsize=(8, 8))
plt.imshow(sol[-1, 1, :].reshape(28, 28).clamp(0,1),
           cmap="gray")
# plt.colorbar()
plt.show()
#%%%
# xT = torch.randn(ndim) * sigma_max
# xt = xT
# for t, t_next in zip(t_eval[:-1], t_eval[1:]):
#     xt = xt + f_edm_ode(t, xt) * (t_next - t)
#
# plt.figure(figsize=(8, 8))
# plt.imshow(xt.reshape(28, 28))
# plt.colorbar()
# plt.show()

#%%
nquad = 100
sigma = 0.1
Delta = 30
x = torch.randn(nbatch, ndim) * sigma
quad_perturb = torch.randn(nquad, ndim)
quad_perturb /= torch.norm(quad_perturb, dim=1, keepdim=True)
quad_perturb *= Delta
perturb_x = x.view(-1, ndim)[:, None] + quad_perturb[None, :, :]
score_perturb = GMM_scores_torch_eff(Xmat, sigma, perturb_x.view(-1, ndim), mus_ssq=mus_ssq)
score_smooth = score_perturb.view(-1, nquad, ndim).mean(dim=1)
#%%

# mtg_tsr = make_grid(sol[-1, :].\
#     reshape(nbatch, 1, 28, 28).clamp(0,1), nrow=int(math.sqrt(nbatch)))
# mtg_smooth_tsr = make_grid(sol_smooth[-1, :].\
#     reshape(nbatch, 1, 28, 28).clamp(0,1), nrow=int(math.sqrt(nbatch)))
# mtg_smooth_particip_tsr = make_grid(sol_smooth_particip[-1, :].\
#     reshape(nbatch, 1, 28, 28).clamp(0,1), nrow=int(math.sqrt(nbatch)))