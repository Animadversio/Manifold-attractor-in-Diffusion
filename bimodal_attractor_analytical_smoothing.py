

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm, trange

#%% 1d case
def bimodal_edm_ode(sigma, x, sigma0=0.3, v=1):
    return sigma / (sigma ** 2 + sigma0 ** 2) * \
            (x - v * np.tanh(v * x / (sigma ** 2 + sigma0 ** 2)))


sigma0 = 0.2
v = 1

sigma_max = 80
sigma_min = 0.002
# evaluate at equal distance log space of sigma
t_eval = np.logspace(np.log10(sigma_max-0.01),
                     np.log10(sigma_min+0.0001), 30)
y0_col = np.random.randn(2000, 1) * sigma_max
ysol_col = []
for y0 in tqdm((y0_col)):
    ysol = solve_ivp(bimodal_edm_ode,
                     t_span=[sigma_max, sigma_min],
                     y0=y0, tfirst=True, t_eval=t_eval)
    ysol_col.append(ysol)

y_traj_col = np.stack([ysol.y for ysol in ysol_col], axis=0)
#%%
plt.figure(figsize=(8, 8))
plt.hist(y_traj_col[:, 0, -2], bins=50)
plt.show()
#%%
plt.figure(figsize=(8, 8))
plt.plot(y_traj_col[::5, 0, -20:].T, color="gray", alpha=0.1)
plt.show()
#%%
def bimodal_edm_ode_smooth(sigma, x, sigma0=0.2, v=1, Delta=2.5):
    return sigma / (sigma ** 2 + sigma0 ** 2) * \
            x - sigma / (Delta) * \
            np.log(np.cosh(v / (sigma ** 2 + sigma0 ** 2) * (x + Delta / 2)) / \
                   np.cosh(v / (sigma ** 2 + sigma0 ** 2) * (x - Delta / 2)) )
sigma0 = 0.2
v = 1


sigma_max = 20
sigma_min = 0.002
# evaluate at equal distance log space of sigma
t_eval = np.logspace(np.log10(sigma_max-0.01),
                     np.log10(sigma_min+0.0001), 30)
y0_col = np.random.randn(2000, 1) * sigma_max
ysol_col = []
for y0 in tqdm((y0_col)):
    ysol = solve_ivp(bimodal_edm_ode_smooth,
                     t_span=[sigma_max, sigma_min],
                     y0=y0, tfirst=True, t_eval=t_eval)
    ysol_col.append(ysol)

y_traj_col = np.stack([ysol.y for ysol in ysol_col], axis=0)
#%%
plt.figure(figsize=(8, 8))
plt.hist(y_traj_col[:, 0, -2], bins=50)
plt.show()
#%%
x_query = np.linspace(-2.5, 2.5, 100)
plt.figure(figsize=(8, 8))
for sigma in t_eval[::5]:
    plt.plot(x_query,
             bimodal_edm_ode(sigma, x_query),
             alpha=0.9, label=f"sigma={sigma:.2f}", lw=3)
plt.legend(fontsize=16)
plt.show()
#%%
x_query = np.linspace(-2.5, 2.5, 100)
plt.figure(figsize=(8, 8))
for sigma in t_eval[::-5]:
    plt.plot(x_query,
             bimodal_edm_ode_smooth(sigma, x_query, Delta=2),
             alpha=0.9, label=f"sigma={sigma:.2f}", lw=3)
plt.legend(fontsize=16)
plt.show()
#%%
plt.figure(figsize=(8, 8))
plt.plot(y_traj_col[:, 0, -10:].T, color="gray", alpha=0.1)
plt.show()

#%%
def bimodal_edm_ode_smooth_uneven(sigma, x, sigma0=0.2, v=1, Delta=1.9, alpha=0.3):
    beta = np.log(alpha / (1 - alpha)) / 2
    return sigma / (sigma ** 2 + sigma0 ** 2) * \
            x - sigma / (Delta) * \
            np.log(np.cosh(beta + v / (sigma ** 2 + sigma0 ** 2) * (x + Delta / 2)) / \
                   np.cosh(beta + v / (sigma ** 2 + sigma0 ** 2) * (x - Delta / 2)) )

sigma_max = 20
sigma_min = 0.002
# evaluate at equal distance log space of sigma
t_eval = np.logspace(np.log10(sigma_max-0.01),
                     np.log10(sigma_min+0.0001), 30)
y0_col = np.random.randn(2000, 1) * sigma_max
ysol_col = []
for y0 in tqdm((y0_col)):
    ysol = solve_ivp(bimodal_edm_ode_smooth_uneven,
                     t_span=[sigma_max, sigma_min],
                     y0=y0, tfirst=True, t_eval=t_eval)
    ysol_col.append(ysol)

y_traj_col = np.stack([ysol.y for ysol in ysol_col], axis=0)
#%%
plt.figure(figsize=(8, 8))
plt.hist(y_traj_col[:, 0, -2], bins=50)
plt.show()
#%%
x_query = np.linspace(-2.5, 2.5, 100)
plt.figure(figsize=(8, 8))
for sigma in t_eval[::5]:
    plt.plot(x_query,
             bimodal_edm_ode(sigma, x_query),
             alpha=0.9, label=f"sigma={sigma:.2f}", lw=3)
plt.legend(fontsize=16)
plt.show()
#%%
x_query = np.linspace(-2.5, 2.5, 100)
plt.figure(figsize=(8, 8))
for sigma in t_eval[::-5]:
    plt.plot(x_query,
             bimodal_edm_ode_smooth(sigma, x_query, Delta=2),
             alpha=0.9, label=f"sigma={sigma:.2f}", lw=3)
plt.legend(fontsize=16)
plt.show()
#%%
plt.figure(figsize=(8, 8))
plt.plot(y_traj_col[:, 0, -10:].T, color="gray", alpha=0.1)
plt.show()
