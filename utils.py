import torch
import torch.nn as nn
import torch.nn.functional as func


def HiPPO(N):
    P = torch.sqrt(1 + 2 * torch.arange(N))
    A = P[:, None] * P[None, :]
    A = torch.tril(A) - torch.diag(torch.arange(N))
    return -A
def Abar(A, hidden_dim, step):
    I = torch.eye(hidden_dim, hidden_dim)
    inv1 = torch.linalg.inv(I - step/2 * A)
    inv2 = torch.linalg.inv(I + step/2 * A)
    return inv1 @ inv2
    

def bar_matrix(M, A, hidden_dim, step):
    I = torch.eye(hidden_dim, hidden_dim)
    inv = torch.linalg.inv(I - step/2 * A)
    return step * inv @ M

def init_beta(A, input_dim, hidden_dim, Nlatent, Nprior, step):
    B = [torch.rand(hidden_dim, input_dim) for n in range(Nprior)]
    Bbar = [bar_matrix(b, A, hidden_dim, step) for b in B]
    Earray = []
    for n in range(Nprior):
        Earray.append([bar_matrix(torch.randn(hidden_dim, 2**k * hidden_dim), A, hidden_dim, step) for k in range(Nlatent)])
    return Bbar, Earray


# def init_simple_matrices(input_dim, hidden_dim, output_dim, step):
#     A = HiPPO(hidden_dim)
#     I = torch.eye(hidden_dim, hidden_dim)
#     inv1 = torch.linalg.inv(I - step/2 * A)
#     inv2 = torch.linalg.inv(I + step/2 * A)
#     B = torch.rand(hidden_dim, input_dim)
#     C = torch.rand(output_dim, hidden_dim)
#     Abar = inv1 @ inv2
#     Bbar = bar_matrix(B, A, hidden_dim, step)
#     return Abar, Bbar, C

# def init_stack_matrices(Nlatent, hidden_dim, latent_dim, output_dim, A, step):
#     Elist, Flist = [], []
#     for k in range(Nlatent):
#         E = torch.randn(hidden_dim, (k+1)*latent_dim)
#         F = torch.randn(output_dim, (k+1)*latent_dim)
#         Ebar = bar_matrix(E, A, hidden_dim, step)
#         Fbar = bar_matrix(F, A, hidden_dim, step)
#         Elist.append(Ebar)
#         Flist.append(Fbar)
#     return Elist, Flist


def init_matrices(input_dim, hidden_dim, latent_dim, output_dim, step):
    A = HiPPO(hidden_dim)
    I = torch.eye(hidden_dim, hidden_dim)
    inv1 = torch.linalg.inv(I - step/2 * A)
    inv2 = torch.linalg.inv(I + step/2 * A)
    B = torch.rand(hidden_dim, input_dim)
    C = torch.rand(output_dim, hidden_dim)
    E = torch.rand(hidden_dim, latent_dim)
    F = torch.rand(output_dim, latent_dim)
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
    res = torch.zeros(N, 1)
    for k in range(length):
        tmp = z[length-1-k]
        tmp = tmp[:, None]
        # print("shape tmp z : ", tmp.shape)
        res += torch.matrix_power(A, k) @ M @ tmp
    return A @ res


def materialize_kernel_generative(A, B, E, E2, x, z, length):
    # z: (B, L, z_dim) , x: (B, L - 1, input_dim)
    N = A.shape[0]
    res = torch.zeros(N, 1)
    for k in range(length-1):
        tmp = z[length-1-k]
        tmp = tmp[:, None]
        Ak = torch.matrix_power(A, k)
        res += Ak @ B @ tmp + Ak @ E @ tmp
    return A @ res + E2 @ z[-1]



