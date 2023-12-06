

import torch
import torch.nn.functional as F
from torch.optim import Adam, SGD
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# from core.utils.plot_utils import saveallforms
R = 1
angles = np.linspace(0, 2*np.pi, 3, endpoint=False)
angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
pnts = R * np.stack([np.cos(angles), np.sin(angles)], axis=1)
#%%
def participants_func(pnts, xy_query, sigma):
    log_weights = (pnts @ xy_query.T / sigma ** 2)
    log_weights -= log_weights.max(axis=0)
    weights_normed = np.exp(log_weights)
    participants = weights_normed / weights_normed.sum(axis=0)
    return participants.T


xx, yy = np.mgrid[-3:3:.02, -3:3:.02]
xy_query = np.stack((xx.flatten(), yy.flatten())).T
sigma = 0.1
participants = participants_func(pnts, xy_query, sigma)
participants_maps = participants.reshape(xx.shape[0], xx.shape[1], -1)
plt.figure(figsize=(8, 8))
plt.imshow(participants_maps[:,:,:3])
plt.show()
#%%
# convolve with a gaussian
from scipy.ndimage import gaussian_filter
conv_sigma = 1.8 / .02
participants_maps_filtered = gaussian_filter(participants_maps, sigma=[conv_sigma,conv_sigma,0])
plt.figure(figsize=(8, 8))
plt.imshow(participants_maps_filtered)
plt.show()
#%%
# quadrature rule for the disk
# https://people.math.sc.edu/Burkardt/py_src/disk_rule/disk_rule.html
# https://pypi.org/project/quadpy/#n-ball-sn

#%% Naive quadrature, summing over the spherical shell
sigma = 0.05
Delta = 1.5
perturb_angle = np.linspace(0, 2*np.pi, 100, endpoint=False)
perturb_vecs = Delta * np.stack([np.cos(perturb_angle), np.sin(perturb_angle)], axis=1)
participants_quadrature = participants_func(pnts, xy_query, sigma)
for perturb_vec in perturb_vecs:
    participants_batch = participants_func(pnts, xy_query + perturb_vec, sigma)
    participants_quadrature += participants_batch
participants_quadrature /= (1 + len(perturb_vecs))
participants_maps_quadrature = participants_quadrature.reshape(xx.shape[0], xx.shape[1], 3)
plt.figure(figsize=(8, 8))
plt.imshow(participants_maps_quadrature)
plt.show()
#%% Using developed quadrature rule
import quadpy
scheme = quadpy.s2.get_good_scheme(8)
scheme.show()
#%%

scheme = quadpy.s2._peirce_1957.peirce_1957(4)
# scheme = quadpy.e2r2.get_good_scheme(15)
print(scheme.points.shape)
scheme.show()
#%%
sigma = 0.05
Delta = 1.5
# scheme = quadpy.s2.get_good_scheme(19)
scheme = quadpy.s2._peirce_1957.peirce_1957(4)
# scheme = quadpy.e2r2.get_good_scheme(15)
scheme.show()
quad_pnts = scheme.points.T
quad_weights = scheme.weights
participants_quadrature = np.zeros((xy_query.shape[0], 3))
for quad_pnt, quad_weight in zip(quad_pnts, quad_weights):
    participants_batch = participants_func(pnts,
                   xy_query + quad_pnt * Delta, sigma)
    participants_quadrature += quad_weight * participants_batch
# participants_quadrature /= (1 + len(perturb_vecs))
participants_maps_quadrature = participants_quadrature.reshape(xx.shape[0], xx.shape[1], 3)
plt.figure(figsize=(8, 8))
plt.imshow(participants_maps_quadrature)
plt.show()
#%%
plt.figure(figsize=(8, 8))
plt.imshow(participants_maps[0], cmap="Reds", alpha=0.3)
plt.imshow(participants_maps[1], cmap="Greens", alpha=0.3)
plt.imshow(participants_maps[2], cmap="Blues", alpha=0.3)
plt.show()

#%%
from tqdm import tqdm
from scipy.integrate import solve_ivp
def trimodal_edm_ode_quadrature_smooth(sigma, x,
                   sigma0=0.1, Delta=0.9,
                   quad_scheme=quadpy.s2._peirce_1957.peirce_1957(4)):
    # if quad_scheme is None:
    #     quad_scheme = quadpy.s2._peirce_1957.peirce_1957(4)
    sigma_eff2 = sigma0 ** 2 + sigma ** 2
    quad_pnts = quad_scheme.points.T
    quad_weights = quad_scheme.weights
    participants_all = participants_func(pnts,
             x + quad_pnts * Delta, np.sqrt(sigma_eff2))
    # VERY SLOW
    # participants_all *= quad_weights[:,None]
    # participants = participants_all.sum(axis=0)
    # FAST
    # participants = np.einsum("qP,q->P",
    #           participants_all, quad_weights)
    # FASTER dot product
    participants = quad_weights @ participants_all
    # avg_pnt = np.einsum("P,Pd->d",
    #           participants, pnts)
    avg_pnt = participants @ pnts
    return sigma / sigma_eff2 * (x - avg_pnt)


sigma_max = 80
sigma_min = 0.002
# evaluate at equal distance log space of sigma
t_eval = np.logspace(np.log10(sigma_max-0.01),
                     np.log10(sigma_min+0.0001), 30)
y0_col = np.random.randn(2000, 2) * sigma_max
ysol_col = []
for y0 in tqdm((y0_col)):
    ysol = solve_ivp(trimodal_edm_ode_quadrature_smooth,
                     t_span=[sigma_max, sigma_min],
                     y0=y0, tfirst=True, t_eval=t_eval)
    ysol_col.append(ysol)

y_traj_col = np.stack([ysol.y for ysol in ysol_col], axis=0)
#%%
plt.figure(figsize=(8, 8))
plt.hist2d(y_traj_col[:, 0, -1], y_traj_col[:, 1, -1], bins=40)
plt.scatter(pnts[:, 0], pnts[:, 1], s=180, color="r")
plt.show()
#%%
plt.figure(figsize=(8, 8))
sns.jointplot(x=y_traj_col[:, 0, -1], y=y_traj_col[:, 1, -1],
              kind="hex", joint_kws= {"gridsize": 25})
plt.scatter(pnts[:, 0], pnts[:, 1], s=180, color="r")
plt.show()

#%%
plt.figure(figsize=(8, 8))
plt.plot(y_traj_col[:, 0, -15:].T, y_traj_col[:, 1, -15:].T,
         color="gray", alpha=0.9)
plt.scatter(pnts[:, 0], pnts[:, 1], s=180, color="r")
plt.show()



#%%
sigma = 0.4
log_weights = (pnts @ xy_query.T / sigma ** 2)
log_weights -= log_weights.max(axis=0)
weights_normed = np.exp(log_weights)
participants = weights_normed / weights_normed.sum(axis=0)
participants_maps = participants.T.reshape(xx.shape[0], xx.shape[1], 3)
