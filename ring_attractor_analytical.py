import scipy.special as spc
import numpy as np
import matplotlib.pyplot as plt
# modified Bessel function of the 1st kind of of 0 order
#%%
sigma2 = 0.5
r = np.linspace(-5, 5, 1000)
fun = spc.i1(r / sigma2) / spc.i0(r / sigma2) - r
plt.plot(r, fun)
plt.axhline(0, color="k", linestyle="--")
plt.show()
#%%
sigma2 = 0.1
r = np.linspace(-5, 5, 1000)
fun = r - spc.i1(r / sigma2) / spc.i0(r / sigma2)
plt.plot(r, fun)
plt.axhline(0, color="k", linestyle="--")
plt.show()
#%%
r = np.linspace(0, 10, 1000)
func = np.log(spc.i0(r))
plt.plot(r, func)
plt.show()


#%%
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
# from scipy.special import i0, i1
# Define the ODE as a function
def dr_dsigma(r, sigma):
    if sigma == 0:
        return 0  # To avoid division by zero
    return (r - spc.i1(r / sigma**2) /
                spc.i0(r / sigma**2) ) / sigma

# Initial condition
r0 = 5  # You can set this to whatever initial condition you have
# Define the range of sigma values
sigma_values = np.linspace(5, 0, 500)
# Integrate the ODE using odeint
r_values = odeint(dr_dsigma, r0, sigma_values)
# Plot the results
plt.figure(figsize=(6, 6))
plt.plot(sigma_values, r_values, label='$r(\sigma)$')
plt.title('Numerical Solution of the ODE')
plt.xlabel('noise scale / time $\sigma$')
plt.ylabel('Radius $r$')
plt.legend()
plt.grid(True)
plt.show()
#%%
def r_ODE_traj(r0, sigma_values):
    """ Solution to r dynamics ODE

    :param r0: e.g.  80
    :param sigma_values:
                e.g. linspace(80,0,100)
    :return:
    """
    def dr_dsigma(r, sigma):
        if sigma == 0:
            return 0  # To avoid division by zero
        return (r - spc.i1(r / sigma ** 2) /
                spc.i0(r / sigma ** 2)) / sigma

    r_values = odeint(dr_dsigma, r0, sigma_values)
    return r_values

sigma_values = np.linspace(5, 0, 100)
plt.figure(figsize=(6, 6))
plt.plot(sigma_values, r_ODE_traj(5, sigma_values), label='$r(\sigma)$')
plt.title('Numerical Solution of the ODE')
plt.xlabel('noise scale / time $\sigma$')
plt.ylabel('Radius $r$')
plt.legend()
plt.grid(True)
plt.show()

#%%

def drsigma_dsigma(rsigma, sigma):
    if sigma == 0:
        return 0  # To avoid division by zero
    return ( - spc.i1(rsigma / sigma) /
               spc.i0(rsigma / sigma) ) / sigma**2


# Initial condition
rsigma0 = 0.05  # You can set this to whatever initial condition you have
# Define the range of sigma values
sigma_values = np.linspace(5, 0, 500)
# Integrate the ODE using odeint
rsigma_values = odeint(drsigma_dsigma, rsigma0, sigma_values)
# Plot the results
plt.figure(figsize=(6, 6))
plt.plot(sigma_values, sigma_values*rsigma_values[:,0], label='$r(\sigma)$')
plt.title('Numerical Solution of the ODE')
plt.xlabel('noise scale / time $\sigma$')
plt.ylabel('Radius $r$')
plt.legend()
plt.grid(True)
plt.show()
