import torch
import torch.nn as nn
import torch.nn.functional as func


def HiPPO(N):
    P = torch.sqrt(1 + 2 * torch.arange(N))
    A = P[:, None] * P[None, :]
    A = torch.tril(A) - torch.diag(torch.arange(N))
    return -A

def bar_matrix(M, A, hidden_dim, step):
    I = torch.eye(hidden_dim, hidden_dim)
    inv = torch.linalg.inv(I - step/2 * A)
    return step * inv @ M

def init_simple_matrices(input_dim, hidden_dim, output_dim, step, rng):
    A = HiPPO(hidden_dim)
    I = torch.eye(hidden_dim, hidden_dim)
    inv1 = torch.linalg.inv(I - step/2 * A)
    inv2 = torch.linalg.inv(I + step/2 * A)
    B = torch.rand(hidden_dim, input_dim, generator=rng)
    C = torch.rand(output_dim, hidden_dim, generator=rng)
    Abar = inv1 @ inv2
    Bbar = bar_matrix(B, A, hidden_dim, step)
    return Abar, Bbar, C

def init_stack_matrices(Nlatent, hidden_dim, latent_dim, output_dim, A, step):
    Elist, Flist = [], []
    for k in range(Nlatent):
        E = torch.randn(hidden_dim, (k+1)*latent_dim)
        F = torch.randn(output_dim, (k+1)*latent_dim)
        Ebar = bar_matrix(E, A, hidden_dim, step)
        Fbar = bar_matrix(F, A, hidden_dim, step)
        Elist.append(Ebar)
        Flist.append(Fbar)
    return Elist, Flist


def init_matrices(input_dim, hidden_dim, latent_dim, output_dim, step, rng):
    A = HiPPO(hidden_dim)
    I = torch.eye(hidden_dim, hidden_dim)
    inv1 = torch.linalg.inv(I - step/2 * A)
    inv2 = torch.linalg.inv(I + step/2 * A)
    B = torch.rand(hidden_dim, input_dim, generator=rng)
    C = torch.rand(output_dim, hidden_dim, generator=rng)
    E = torch.rand(hidden_dim, latent_dim, generator=rng)
    F = torch.rand(output_dim, latent_dim, generator=rng)
    Abar = inv1 @ inv2
    Bbar = bar_matrix(B, A, hidden_dim, step)
    Ebar = bar_matrix(E, A, hidden_dim, step)
    return Abar, Bbar, C, Ebar, F



def discretize(A, B, C, E, step):   # A adapter dans notre cas
    I = torch.eye(A.shape[0])
    BL = torch.linalg.inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    Eb = (BL * step) @ E
    return Ab, Bb, C, Eb


""" Function creating the kernels
"""
# def materialize_kernel(z, A, M, L):
#         # L = z.shape[1]     # z: (Batch_size, L, 1)
#         kernel =  torch.stack(
#             [(torch.matrix_power(A, l) @ M) for l in range(L)]
#         )
#         return kernel
def materialize_kernel(A, M, z, length):
    N = A.shape[0]
    H = M.shape[1]
    res = torch.zeros(N, 1)
    for k in range(length):
        tmp = z[length-1-k]
        tmp = tmp[:, None]
        # print("shape tmp z : ", tmp.shape)
        res += torch.matrix_power(A, k) @ M @ tmp
    return A @ res





def append_ascent(A, nA, B, nB, nlayers):
    res = []
    for i in range(nlayers):
        for iA in range(nA):
            res.append(A)
        res.append(B)
    return res
