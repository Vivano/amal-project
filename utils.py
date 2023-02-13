import torch
import torch.nn as nn
import torch.nn.functional as func



class TimeseriesDataset(torch.utils.data.Dataset):   
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.__len__() 

    def __getitem__(self, index):
        return self.X[index], self.y[index]


###################################################################


def HiPPO(N):
    P = torch.sqrt(1 + 2 * torch.arange(N))
    A = P[:, None] * P[None, :]
    A = torch.tril(A) - torch.diag(torch.arange(N))
    return -A
def Abar(A, step):
    N = A.shape[0]
    I = torch.eye(N, N)
    inv1 = torch.linalg.inv(I - step/2 * A)
    inv2 = torch.linalg.inv(I + step/2 * A)
    return inv1 @ inv2
    

def bar_matrix(M, A, hidden_dim, step):
    I = torch.eye(hidden_dim, hidden_dim)
    inv = torch.linalg.inv(I - step/2 * A)
    return step * inv @ M


def init_params(A, input_dim, hidden_dim, latent_dim, output_dim, step):
    B = torch.rand(hidden_dim, input_dim)
    C = torch.rand(output_dim, hidden_dim)
    E = torch.rand(hidden_dim, latent_dim)
    F = torch.rand(output_dim, latent_dim)
    Bbar = bar_matrix(B, A, hidden_dim, step)
    Ebar = bar_matrix(E, A, hidden_dim, step)
    return Bbar, C, Ebar, F


#################################################


""" Function creating the kernels
"""
def materialize_kernel(A, M, z, length):
    N = A.shape[0]
    res = torch.zeros(N, 1)
    for k in range(length):
        tmp = z[length-1-k]
        tmp = tmp[:, None]
        # print("shape tmp z : ", tmp.shape)
        res += torch.matrix_power(A, k) @ M @ tmp
    return A @ res


def materialize_kernel_generative(A, B, E1, E2, x, z, length):
    # z: (B, L, z_dim) , x: (B, L - 1, input_dim)
    N = A.shape[0]
    res = torch.zeros(N, 1)
    for k in range(length-1):
        tmp_x = x[length-2-k]
        tmp_x = tmp_x[:, None]
        tmp_z = z[length-2-k]
        tmp_z = tmp_z[:, None]
        Ak = torch.matrix_power(A, k)
        res += Ak @ B @ tmp_x + Ak @ E1 @ tmp_z
    zz = E2 @ z[-1]
    zz = zz[:, None]
    return A @ res + zz


# def beta_parameter(A, input_dim, hidden_dim, latent_dim, nlatent, nprior, step):
#     B, E = init_beta(A, input_dim, hidden_dim, latent_dim, nlatent, nprior, step)
#     Bparam, Eparam = nn.ParameterList([nn.Parameter(b) for b in B]), nn.ParameterList([nn.ParameterList([nn.Parameter(e) for e in elist]) for elist in E])
#     return Bparam, Eparam


################################################################


# class mySequential(nn.Sequential):
#     def forward(self, *inputs):
#         for module in self._modules.values():
#             if type(inputs) == tuple:
#                 inputs = module(*inputs)
#             else:
#                 inputs = module(inputs)
#         return inputs



# class inheriting nn.Sequential to handle multi inputs forward
class mySequential(nn.Sequential):
    def forward(self, input1, input2):
        for module in self._modules.values():
            input1, input2 = module(input1, input2)
        return input1, input2



# class performing two linear layers in parallel for two different inputs
class twoLinear(nn.Module):
    def __init__(self, n, nlatent, hidden_dim, latent_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear((2**(nlatent-n))*hidden_dim, (2**(nlatent-n-1))*hidden_dim)
        self.fc2 = nn.Linear((2**(nlatent-n))*hidden_dim, (2**(nlatent-n-1))*hidden_dim)
    def forward(self,x,z):
        x = self.fc1(x)
        z = self.fc2(z)
        return x, z
