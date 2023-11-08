import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.special import i0, i1, i0e, i1e
from scipy.integrate import odeint, solve_ivp
from tqdm import tqdm

# Note using i1e / i0e is more stable than i1 / i0
gamma = 1  # strength for non uniform bump
R = 1  # radius of the ring.
psi_peak = np.pi / 4
psi_star = psi_peak - np.pi


def rho_fun_polar(r, theta, sigma, R, psi_star, gamma):
    return np.sqrt(
        r ** 2 + (gamma * sigma**2 / R) **2
        - 2 * r * gamma * sigma**2 / R * np.cos(theta - psi_star)
    )


def dtheta_ode(sigma, r, theta):
    rho_var = rho_fun_polar(r, theta, sigma, R, psi_star, gamma)
    return (- 1 / r * sigma * gamma / rho_var * \
            i1e(R / sigma ** 2 * rho_var) / \
            i0e(R / sigma ** 2 * rho_var) * \
            np.sin(theta - psi_star) )


def dr_ode(sigma, r, theta):
    rho_var = rho_fun_polar(r, theta, sigma, R, psi_star, gamma)
    return (r / sigma -
            R / sigma / rho_var *
                i1e(R / sigma ** 2 * rho_var) / \
                i0e(R / sigma ** 2 * rho_var) * \
                (r - gamma * sigma ** 2 / R * np.cos(theta - psi_star)))


def polar_ode(sigma, y):
    r = y[0]
    theta = y[1]
    dy = np.stack([dr_ode(sigma, r, theta),
                         dtheta_ode(sigma, r, theta)])
    return dy

#%% Main experiment
from circuit_toolkit.plot_utils import saveallforms

#%% Use ode45 to solve the ODE, without fixed time steps
# tsteps = np.linspace(100, 0.005, 500)
figdir = r"/Users/binxuwang/Library/CloudStorage/OneDrive-HarvardUniversity/HaimDiffusionRNNProj/Ring_Nonunif/dynamics_rinit"
sigma_max = 100
sigma_min = 0.005
"""
Solve the diffusion ode with different initial radius, but uniform angular dist.
see the dynamics a long angular and radial direction. 
See the dynamics of the angular distribution change over the course of diffusion.  
"""
theta_dist = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
# evaluate at equal distance log space of sigma
t_eval = np.logspace(np.log10(sigma_max-0.01),
                     np.log10(sigma_min+0.0001), 20)
# r_init_norm = 1  # scipy.stats.norm(0, 1).rvs(10000)
# r_chi_init = r_init_norm #100 * scipy.stats.chi(2).rvs(1000)
for r_init_norm in [100, 50, 25, 12.5, 6, 3, 1.5, 1, 0.5, 0.2, 0.1, 0.04]:
    ysol_col = []
    for thete_init, in tqdm(zip(theta_dist)):
        ysol = solve_ivp(polar_ode, t_span=[sigma_max, sigma_min],
                         y0=np.array([r_init_norm, thete_init]),
                         tfirst=True, t_eval=t_eval)
        ysol_col.append(ysol)

    ttraj = ysol.t
    ytraj_tsr = np.stack([ysol.y for ysol in ysol_col])
    y_final_dist = np.stack([ysol.y[:,-1] for ysol in ysol_col])
    #%%
    plt.figure()
    for ysol in ysol_col[::4]:
        # plt.plot(ysol.t[-10:], ysol.y[-1,-10:], color="gray", alpha=0.3)
        plt.plot(ysol.t[:], ysol.y[-1, :], color="gray", alpha=0.2)
    plt.axhline(psi_peak, color="r", linestyle="--")
    plt.xlim([sigma_min-0.2, sigma_max+0.2])
    plt.xlabel("sigma")
    plt.ylabel("theta")
    plt.title(f"Angular Trajectories for init r={r_init_norm}\n psi*={psi_peak / np.pi:.2f}$\pi$ gamma={gamma} R={R}")
    saveallforms(figdir, f"angular_traj_rinit{r_init_norm:.1f}_sigmamax{sigma_max}")
    plt.show()
    #%%
    plt.figure()
    for ysol in ysol_col[::4]:
        plt.plot(ysol.t[:], ysol.y[0, :], color="gray", alpha=0.2)
    plt.axhline(R, color="r", linestyle="--")
    plt.xlim([sigma_min-0.2, sigma_max+0.2])
    plt.xlabel("sigma")
    plt.ylabel("r")
    plt.title(f"Radial Trajectories for init r={r_init_norm}\n psi*={psi_peak / np.pi:.2f}$\pi$ gamma={gamma} R={R}")
    saveallforms(figdir, f"radial_traj_rinit{r_init_norm:.1f}_sigmamax{sigma_max}")
    plt.show()
    #%%
    plt.figure(figsize=[6, 6])
    for ysol in ysol_col[::4]:
        plt.plot(ysol.y[0, :] * np.cos(ysol.y[-1, :]),
                       ysol.y[0, :] * np.sin(ysol.y[-1, :]), color="gray", alpha=0.2)
    # plot unit circle
    plt.plot(np.cos(np.linspace(0, 2 * np.pi, 100)),
                np.sin(np.linspace(0, 2 * np.pi, 100)), color="r", linestyle="--")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.axis("image")
    plt.title(f"Planar Trajectories for init r={r_init_norm}\n psi*={psi_peak / np.pi:.2f}$\pi$ gamma={gamma} R={R}")
    saveallforms(figdir, f"planar_traj_rinit{r_init_norm:.1f}_sigmamax{sigma_max}")
    plt.show()
    #%% modulo 2pi
    plt.figure()
    for time_idx in range(0, len(t_eval), 3):
        sigma = t_eval[time_idx]
        plt.hist(ytraj_tsr[:, 1, time_idx] % (2*np.pi), bins=40, alpha=0.5, label=f"sigma={sigma:.2f}")
    plt.legend()
    plt.xticks([i * 2 * np.pi / 8 for i in range(9)],
               [f"{i}$\pi$/4" for i in range(9)],)
    plt.xlim([0, 2 * np.pi])
    plt.xlabel("theta")
    plt.ylabel("count")
    plt.title(f"Evolution of theta distribution for init r={r_init_norm}\n psi*={psi_peak / np.pi:.2f}$\pi$ gamma={gamma} R={R}")
    saveallforms(figdir, f"theta_dist_evolution_rinit{r_init_norm:.1f}_sigmamax{sigma_max}")
    plt.show()








#%% Reparametrize time in different ways and plot time again.
plt.figure()
for ysol in ysol_col[::3]:
    # plt.plot(ysol.t[-10:], ysol.y[-1,-10:], color="gray", alpha=0.3)
    sigma_sy = ysol.t[:] / np.sqrt(1 + ysol.t[:]**2)
    plt.plot(sigma_sy, ysol.y[-1, :], color="gray", alpha=0.2)

plt.xlabel("sigma (VP sigma)")
plt.ylabel("theta")
plt.title(f"Trajectories for init r={r_init_norm}, psi*={psi_peak / np.pi:.2f}$\pi$")
plt.show()
#%%
plt.figure()
plt.hist(ytraj_tsr_chi[:, -1, -1] % (2 * np.pi), bins=50)
plt.show()
#%%
tsteps = np.linspace(100, 0.005, 500)
r_chi_init = 100 * scipy.stats.chi(2).rvs(10000)
theta_dist = np.linspace(0, 2 * np.pi, 10000, endpoint=False)
ysol_col = []
for r_init, thete_init in tqdm(zip(r_chi_init, theta_dist)):
    ysol = solve_ivp(polar_ode, t_span=[100, 0.005],
                     y0=np.array([r_init, thete_init]),
                     tfirst=True)
    ysol_col.append(ysol)

#%%
y_final_dist_chi = np.stack([ysol.y[:,-1] for ysol in ysol_col])

#%%
plt.figure()
for ysol in ysol_col[::3]:
    # plt.plot(ysol.t[-10:], ysol.y[-1,-10:], color="gray", alpha=0.3)
    plt.plot(ysol.t[:], ysol.y[-1, :], color="gray", alpha=0.02)
plt.xlim([0, 5])
plt.show()
#%%
plt.hist(y_final_dist_chi[:,1], bins=50)
plt.show()

#%% modulo 2pi
plt.hist(y_final_dist_chi[:, 1] % (2*np.pi), bins=40)
plt.xticks([i * 2 * np.pi / 8 for i in range(9)],
           [f"{i}$\pi$/4" for i in range(9)],)
plt.show()

#%% Scratch space


#%% Testing
polar_ode(80, np.random.randn(2) * 0.1)
dtheta_ode(5.1, 2,  np.pi/3)

#%%
tsteps = np.linspace(80, 0.002, 100)
ysol = odeint(polar_ode, np.array([80.0, np.pi/2]),
       t=tsteps, tfirst=True)
#%%
tsteps = np.linspace(80, 0.002, 100)
r_fix_init = 80.1
theta_dist = np.linspace(0, 2 * np.pi, 1000, endpoint=False)
ysol_col = []
for thete_init in theta_dist:
    ysol = odeint(polar_ode, np.array([r_fix_init, thete_init]),
                  t=tsteps, tfirst=True)
    ysol_col.append(ysol)

ytraj_tsr = np.stack(ysol_col)

#%%
plt.figure()
for ysol in ysol_col:
    plt.plot(tsteps[-10:], ysol[-10:,-1], color="gray", alpha=0.3)
plt.show()

#%%
plt.figure()
plt.hist(ytraj_tsr[:, -1, -1], bins=50)
plt.show()

#%%
tmpx = np.linspace(0,10,100)
plt.scatter(i1(tmpx) / i0(tmpx), i1e(tmpx) / i0e(tmpx), )
plt.show()

#%%
tsteps = np.linspace(100, 0.005, 500)
r_chi_init = 100 * scipy.stats.chi(2).rvs(10000)
theta_dist = np.linspace(0, 2 * np.pi, 10000, endpoint=False)
ysol_col = []
for r_init, thete_init in tqdm(zip(r_chi_init, theta_dist)):
    ysol = odeint(polar_ode, np.array([r_init, thete_init]),
                  t=tsteps, tfirst=True)
    ysol_col.append(ysol)

ytraj_tsr_chi = np.stack(ysol_col)
