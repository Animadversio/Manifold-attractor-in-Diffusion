"""
A demo of the principles of flow matching.
https://arxiv.org/abs/2210.02747
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
#%%
def gen_sample(batch_size, ):
    x_up = torch.rand(batch_size // 2)
    y_up = torch.rand(batch_size // 2)
    x_down = - torch.rand(batch_size // 2)
    y_down = - torch.rand(batch_size // 2)
    x = torch.cat([x_up, x_down])
    y = torch.cat([y_up, y_down])
    return torch.stack([x, y], dim=1)


def generate_spiral_samples_torch(n_points, a=1, b=0.2):
    """
    Generate points along a spiral using PyTorch.

    Parameters:
    - n_points (int): Number of points to generate.
    - n_batches (int): Number of batches to generate.
    - a, b (float): Parameters that define the spiral shape.

    Returns:
    - torch.Tensor: Batch of vectors representing points on the spiral.
    """
    theta = torch.linspace(0, 4 * np.pi, n_points)  # angle theta
    r = a + b * theta  # radius
    x = r * torch.cos(theta)  # x = r * cos(theta)
    y = r * torch.sin(theta)  # y = r * sin(theta)
    spiral_batch = torch.stack((x, y), dim=1)
    return spiral_batch


def generate_heart_samples_torch(n_points, ):
    """
    Generate points along a heart shape using PyTorch.

    Parameters:
    - n_points (int): Number of points to generate.
    - n_batches (int): Number of batches to generate.

    Returns:
    - torch.Tensor: Batch of vectors representing points on the heart shape.
    """
    theta = torch.linspace(0, 2 * np.pi, n_points)  # angle theta
    x = 16 * torch.sin(theta)**3  # x coordinate formula for heart shape
    y = 13 * torch.cos(theta) - 5 * torch.cos(2 * theta) - 2 * torch.cos(3 * theta) - torch.cos(4 * theta)  # y coordinate
    heart_batch = torch.stack((x, y), dim=1) / 8
    return heart_batch


def generate_checkerboard_samples_vectorized(n_samples, grid_size=4,
                                             xlim=(-2, 2), ylim=(-2, 2)):
    """
    Vectorized version to generate samples with uniform distribution on the black squares of a checkerboard.

    Parameters:
    - n_samples (int): Number of samples to generate.
    - grid_size (int): Size of the checkerboard grid. Default is 8x8.

    Returns:
    - torch.Tensor: Samples uniformly distributed on the black squares of the checkerboard.
    """
    # Calculate the size of each square in the checkerboard
    square_size = 1.0 / grid_size

    # Generate random positions
    positions = torch.rand(n_samples, 2)

    # Scale to the grid size
    scaled_positions = torch.floor(positions * grid_size)

    # Determine if a square is black (row+col is even)
    is_black_square = (scaled_positions.sum(axis=1) % 2 == 0)

    # Filter only the points that fall on black squares
    black_square_positions = positions[is_black_square]

    # If we don't have enough points, repeat the process
    while black_square_positions.shape[0] < n_samples:
        additional_positions = torch.rand(n_samples, 2)
        scaled_additional_positions = torch.floor(additional_positions * grid_size)
        is_black_square_additional = (scaled_additional_positions.sum(axis=1) % 2 == 0)
        additional_black_square_positions = additional_positions[is_black_square_additional]
        black_square_positions = torch.cat([black_square_positions, additional_black_square_positions], dim=0)

    # Take only the required number of samples
    black_square_positions = black_square_positions[:n_samples]
    black_square_positions[:, 0] = xlim[0] + (xlim[1] - xlim[0]) * black_square_positions[:, 0]
    black_square_positions[:, 1] = ylim[0] + (ylim[1] - ylim[0]) * black_square_positions[:, 1]

    return black_square_positions
#%%
batch_size = 1000
noise = torch.randn(batch_size, 2)
samples = gen_sample(batch_size)
#%%
plt.scatter(samples[:,0], samples[:,1])
plt.show()
#%%
class TimeProjection(nn.Module):
    def __init__(self, dim_out):
        super().__init__()
        self.dim_out = dim_out
        self.weight = nn.Parameter(torch.randn(1, dim_out // 2))

    def forward(self, x):
        proj = x @ self.weight
        return torch.concatenate([torch.cos(proj),
                                  torch.sin(proj)], dim=1)


class VectorField(nn.Module):
    def __init__(self, hidden_dim=32, space_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(space_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, space_dim),
        )

    def forward(self, x):
        return self.model(x)


class VectorField_Timedep(nn.Module):
    def __init__(self, hidden_dim=32, space_dim=2, time_dim=20):
        super().__init__()
        self.time_proj = TimeProjection(time_dim)
        self.model = nn.Sequential(
            nn.Linear(space_dim + time_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, space_dim),
        )

    def forward(self, x, t):
        t_proj = self.time_proj(t.view(-1, 1))
        return self.model(torch.concatenate([x, t_proj], dim=1))

#%%
vec_td = VectorField_Timedep()
vec_td(samples, torch.rand(batch_size))
#%% Training with static model
batch_size = 10000
sigma_min = 0.0002
epochs = 1000
device = "cpu"
vec_net = VectorField()
vec_net.to(device)
optimizer = Adam(vec_net.parameters(), lr=0.01)
for i in range(epochs):
    optimizer.zero_grad()
    noise = torch.randn(batch_size, 2)
    samples = gen_sample(batch_size)
    t_vec = torch.rand(batch_size,)
    x_t = (t_vec[:, None] * samples +
          (1 - (1 - sigma_min) * t_vec)[:, None] * noise)
    vec_t = samples - (1 - sigma_min) * noise
    loss = ((vec_net(x_t.to(device)) - vec_t.to(device)) ** 2).mean()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"epoch {i}: {loss.item()}")


#%% Training with time dependence model
batch_size = 10000
sigma_min = 0.0002
epochs = 2500
device = "cpu"

vec_td = VectorField_Timedep(hidden_dim=64, time_dim=20)
vec_td.to(device)
vec_td.train()
optimizer = Adam(vec_td.parameters(), lr=0.01)
for i in range(epochs):
    optimizer.zero_grad()
    noise = torch.randn(batch_size, 2)
    samples = gen_sample(batch_size)
    t_vec = torch.rand(batch_size,)
    x_t = (t_vec[:, None] * samples +
          (1 - (1 - sigma_min) * t_vec)[:, None] * noise)
    vec_t = samples - (1 - sigma_min) * noise
    loss = ((vec_td(x_t.to(device), t_vec.to(device)) - vec_t.to(device)) ** 2).mean()
    loss.backward()
    optimizer.step()
    if i % 100 == 0:
        print(f"epoch {i}: {loss.item()}")


#%% Sample
vec_td.eval()
n_sample = 1000
x_sample = torch.randn(n_sample, 2)
dt = 0.005
for t in torch.arange(0, 1, dt):
    with torch.no_grad():
        v_xt = vec_td(x_sample, torch.ones(n_sample) * t)
    x_sample = x_sample + dt * v_xt

plt.scatter(x_sample[:,0], x_sample[:, 1])
plt.axis("equal")
plt.show()
#%%
def train_flow_matching(generating_func,
                        vec_td_net=None,
                        batch_size=10000,
                        sigma_min=0.0002,
                        epochs=2500,
                        lr=0.01,
                        device="cpu"):
    if vec_td_net is None:
        vec_td_net = VectorField_Timedep(hidden_dim=64,
                                         time_dim=20)
    vec_td_net.to(device)
    vec_td_net.train()
    optimizer = Adam(vec_td_net.parameters(), lr=lr)
    for i in range(epochs):
        optimizer.zero_grad()
        noise = torch.randn(batch_size, 2)
        samples = generating_func(batch_size)
        t_vec = torch.rand(batch_size, )
        x_t = (t_vec[:, None] * samples +
               (1 - (1 - sigma_min) * t_vec)[:, None] * noise)
        vec_t = samples - (1 - sigma_min) * noise
        loss = ((vec_td_net(x_t.to(device), t_vec.to(device)) - vec_t.to(device)) ** 2).mean()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"epoch {i}: {loss.item()}")
    return vec_td_net


def sample_flow_matching(vec_td_net,
                        n_sample=1000,
                        dt=0.005,
                        device="cpu"):
    vec_td_net.eval().to(device)
    x_sample = torch.randn(n_sample, 2).to(device)
    for t in torch.arange(0, 1, dt):
        with torch.no_grad():
            v_xt = vec_td_net(x_sample, torch.ones(n_sample) * t)
        x_sample = x_sample + dt * v_xt.to(device)
    return x_sample


def sample_flow_matching_traj(vec_td_net,
                        n_sample=1000,
                        dt=0.005,
                        device="cpu"):
    vec_td_net.eval().to(device)
    x_sample = torch.randn(n_sample, 2).to(device)
    x_sample_traj = [x_sample.clone()]
    for t in torch.arange(0, 1, dt):
        with torch.no_grad():
            v_xt = vec_td_net(x_sample, torch.ones(n_sample) * t)
        x_sample = x_sample + dt * v_xt.to(device)
        x_sample_traj.append(x_sample.clone())
    x_sample_traj = torch.stack(x_sample_traj, dim=0)
    return x_sample_traj

def plot_flow_traj(x_sample_traj,):
    figh = plt.figure(figsize=[10, 10])
    plt.plot(x_sample_traj[:,:,0].numpy(),
             x_sample_traj[:,:,1].numpy(),
             color="k", alpha=0.25, lw=0.5)
    plt.scatter(x_sample_traj[-1,:,0].numpy(),
                x_sample_traj[-1,:,1].numpy(), marker="x",
                alpha=0.4, color="blue")
    plt.scatter(x_sample_traj[0,:,0].numpy(),
                x_sample_traj[0,:,1].numpy(), marker="o",
                alpha=0.4, color="r")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
    return figh


    pass




#%% Spiral
vec_td_spiral = train_flow_matching(generate_spiral_samples_torch,)
samples_spiral = sample_flow_matching(vec_td_spiral)
plt.scatter(samples_spiral[:,0], samples_spiral[:, 1])
plt.axis("equal")
plt.show()
#%%
samples_spiral_traj = sample_flow_matching_traj(
                    vec_td_spiral, n_sample=500,)
plot_flow_traj(samples_spiral_traj)
#%% Heart
vec_td_heart = train_flow_matching(generate_heart_samples_torch,)
samples_heart = sample_flow_matching(vec_td_heart)
plt.scatter(samples_heart[:,0], samples_heart[:, 1])
plt.axis("equal")
plt.show()
#%% sample trajectory
samples_heart_traj = sample_flow_matching_traj(
                    vec_td_heart, n_sample=500,)
plot_flow_traj(samples_heart_traj)
#%%
vec_td_checker = train_flow_matching(generate_checkerboard_samples_vectorized,
                                     VectorField_Timedep(64, time_dim=64),
                                     epochs=5000)
#%%
samples_checker = sample_flow_matching(vec_td_checker, n_sample=1000, dt=0.01)
plt.scatter(samples_checker[:,0], samples_checker[:, 1])
plt.axis("equal")
plt.show()
#%%
samples_checker_traj = sample_flow_matching_traj(
                    vec_td_checker, n_sample=500,)
plot_flow_traj(samples_checker_traj)

#%%
train_samples_checker = generate_checkerboard_samples_vectorized(10000, grid_size=4)
plt.scatter(train_samples_checker[:,0], train_samples_checker[:, 1])
plt.axis("equal")
plt.show()



#%% Dev zone
xx, yy = np.meshgrid(np.linspace(-3, 3, 50),
                     np.linspace(-3, 3, 50))
xx = torch.tensor(xx).float()
yy = torch.tensor(yy).float()
grid = torch.stack([xx, yy], dim=2)
grid = grid.reshape(-1, 2)
with torch.no_grad():
    vec = vec_net(grid)
vec = vec.reshape(*xx.shape, 2)
plt.quiver(xx.numpy(), yy.numpy(), vec[:,:,0].numpy(), vec[:,:,1].numpy())
plt.show()
#%%
#%%
# plt.figure(figsize=[10, 10])
# plt.plot(samples_heart_traj[:,:,0].numpy(),
#          samples_heart_traj[:,:,1].numpy(),
#          color="k", alpha=0.25, lw=0.5)
# plt.scatter(samples_heart_traj[-1,:,0].numpy(),
#             samples_heart_traj[-1,:,1].numpy(), marker="x",
#             alpha=0.4, color="blue")
# plt.scatter(samples_heart_traj[0,:,0].numpy(),
#             samples_heart_traj[0,:,1].numpy(), marker="o",
#             alpha=0.4, color="r")
# plt.axis("equal")
# plt.show()

