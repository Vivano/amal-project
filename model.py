from utils import *

""" LS4 Prior Layer
INPUTS
    * input_dim
    * output_dim
    * hidden_dim
    * latent_dim
    * time_step
    * randomness
OUPUT
    * y
"""
class LS4PriorLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, latent_dim, time_step, randomness) -> None:
        super().__init__()
        self.p = input_dim
        self.p_ = output_dim
        self.N = hidden_dim
        self.H = latent_dim
        self.step = time_step
        self.rng = randomness
        self.A, B, C, E, F = init_simple_matrices(self.p, self.N, self.p_, self.step, self.rng)
        self.B, self.C = nn.Parameter(B), nn.Parameter(C)
        self.E, self.F = nn.Parameter(E), nn.Parameter(F)
        self.gelu = nn.GELU()

    def forward(self, z):
        batch_size = z.shape[0]
        y = torch.randn(batch_size, self.p_)
        for i in range(batch_size):
            K = self.C @ materialize_kernel(z[i], self.A, self.E)
            y[i,:] = self.gelu(K + self.F @ z[i][-1])
        return y



""" LS4 Prior block
"""
class LS4PriorBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, latent_dim, time_step, randomness) -> None:
        super().__init__()
        self.ls4 = LS4PriorLayer(input_dim, output_dim, hidden_dim, latent_dim, time_step, randomness)
        self.lin = nn.Linear(latent_dim, output_dim)
        self.norm = nn.LayerNorm(latent_dim)
    def forward(self, z):
        tmp = self.norm(self.lin(self.ls4(z))) + z
        return tmp


def append_ascent(nprior, nlatent, input_dim, output_dim, hidden_dim, latent_dim, time_step, randomness):
    res = []
    for n in range(nlatent):
        for i in range(nprior):
            res.append(LS4PriorBlock(input_dim, output_dim, hidden_dim, (2**(n+1))*latent_dim, time_step, randomness))
        res.append(nn.Linear((2**(n+1))*latent_dim, (2**n)*latent_dim))
    return res


""" LS4 Prior Network 
"""
class LS4PriorNet(nn.Module):
    def __init__(self, Nlatent, Nprior, input_dim, output_dim, hidden_dim, latent_dim, time_step, length, randomness):
        super().__init__()
        self.nlatent = Nlatent
        self.nprior = Nprior
        layers_descent = [nn.Linear((2**n) * latent_dim, (2**(n+1)) * latent_dim) for n in range(self.nlatent)] # -1 ?
        layers_ascent = append_ascent(self.nprior, self.nlatent, input_dim, output_dim, hidden_dim, latent_dim, time_step, randomness)
        self.descent = nn.Sequential(*layers_descent)
        self.ascent = nn.Sequential(*layers_ascent)
        self.reparam = LS4PriorBlock(input_dim, output_dim, hidden_dim, latent_dim, time_step, randomness)
        self.mu = nn.Parameter(torch.randn(latent_dim))
        self.sigma = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.length = length
    def forward(self):
        eps = torch.randn(self.hidden)
        # compléter
        z = self.mu + self.sigma @ eps
        return z



""" LS4 Prior block for multidimensional variables
"""
class LS4PriorMulti(nn.Module):
    def __init__(self, C, psi):
        super(LS4PriorMulti, self).__init__()
        self.C = C
        self.LS4_params = psi.LS4_params
        self.linear = nn.Linear(C, C)
        # conv_layer = torch.nn.Conv2d(3, 3, kernel_size=1) # to replace the linear 
        self.LS4_prior_layer = LS4PriorNet(*self.LS4_params)
        
    def forward(self, z):
        # z: (B, L, C)
        for c in range(self.C):
            z[:,:,c] = self.LS4_prior_layer(z[:,:,c])
        z = self.linear(z) # (B, L, C) channel-wise mixing
        return z



""" LS4 Generative block
"""
class GenerativeModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GenerativeModel, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, z):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x
    # c'est pas fini
