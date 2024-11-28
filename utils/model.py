import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

# naive implementation of compute_frontiers
# def compute_frontiers(map):
#     """
#     Input:
#         `map` FloatTensor(4 h, w)
#     Output:
#         `frontiers` FloatTensor(1, h, w)
#     """
#     # kernel = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]],
#     #                        dtype=torch.float32, device=global_input.device)
#     # kernel = kernel.view(1, 1, 3, 3)
#     # frontiers = F.conv2d(global_input, kernel, padding=1)
#     # frontiers = (frontiers > 0).float()
#     # return frontiers

#     # free spaces:
#     obstacle_map = map[0, :, :]
#     explored_map = map[1, :, :]
#     explored_free_space = (obstacle_map <= 0.2).float() * (explored_map >= 0.8).float()
#     # import pdb; pdb.set_trace()

#     # frontiers:
#     # frontier is a grid cell that is adjacent to an unknown cell, and itself is free
#     frontiers_map = torch.zeros_like(explored_free_space)
#     for i in range(0, explored_free_space.size(0) - 1):
#         for j in range(0, explored_free_space.size(1) - 1):
#             if explored_free_space[ i, j] == 1:
#                 # for neightbors
#                 for ni, nj in get_grid_neighbours(i, j):
#                     if explored_map[ni, nj] < 0.2:
#                         frontiers_map[ i, j] = 1.0
#                         break
#     # import pdb; pdb.set_trace()
#     return frontiers_map

# vectorized implementation of compute_frontiers
def compute_frontiers(map):
    """
    Input:
        `map` FloatTensor(4, h, w)
    Output:
        `frontiers` FloatTensor(1, h, w)
    """
    # free spaces:
    obstacle_map = map[0, :, :]
    explored_map = map[1, :, :]
    explored_free_space = (obstacle_map <= 0.2).float() * (explored_map >= 0.8).float()

    # Create a padded version of the explored map to handle edge cases
    padded_explored_map = F.pad(explored_map, (1, 1, 1, 1), mode='constant', value=0)
    padded_explored_free_space = F.pad(explored_free_space, (1, 1, 1, 1), mode='constant', value=0)

    # Create a kernel to check for adjacent unknown cells
    kernel = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32, device=map.device).view(1, 1, 3, 3)

    # Convolve the padded explored map with the kernel
    adjacent_unknowns = F.conv2d(padded_explored_map.unsqueeze(0).unsqueeze(0), kernel, padding=0).squeeze()

    # Identify frontiers: free cells adjacent to unknown cells
    frontiers_map = (padded_explored_free_space[1:-1, 1:-1] == 1) & (adjacent_unknowns[1:-1, 1:-1] > 0)
    frontiers_map = frontiers_map.float()

    return frontiers_map

def get_grid_neighbours( i, j):
    return [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]

def get_grid(pose, grid_size, device):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)

    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    t = t * np.pi / 180.
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta12 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta1 = torch.stack([theta11, theta12], 1)

    theta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1)

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size))
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size))

    return rot_grid, trans_grid


class ChannelPool(nn.MaxPool1d):
    def forward(self, x):
        n, c, w, h = x.size()
        x = x.view(n, c, w * h).permute(0, 2, 1)
        x = x.contiguous()
        pooled = F.max_pool1d(x, c, 1)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/utils.py#L32
class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L10
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L82
class NNBase(nn.Module):

    def __init__(self, recurrent, recurrent_input_size, hidden_size):

        super(NNBase, self).__init__()
        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRUCell(recurrent_input_size, hidden_size)
            nn.init.orthogonal_(self.gru.weight_ih.data)
            nn.init.orthogonal_(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def rec_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x = hxs = self.gru(x, hxs * masks[:, None])
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N, 1)

            outputs = []
            for i in range(T):
                hx = hxs = self.gru(x[i], hxs * masks[i])
                outputs.append(hx)

            # x is a (T, N, -1) tensor
            x = torch.stack(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)

        return x, hxs
