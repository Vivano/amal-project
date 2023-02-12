from utils import *


##################################### PRIOR NETWORK #######################################################

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
    def __init__(self, A, input_dim, output_dim, hidden_dim, latent_dim, step) -> None:
        super().__init__()
        self.A = A
        self.Abar = Abar(self.A, step)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.step = step
        B, C, E, F = init_params(self.A, self.input_dim, self.hidden_dim, self.latent_dim, self.output_dim, self.step)
        self.B, self.C = nn.Parameter(B), nn.Parameter(C)
        self.E, self.F = nn.Parameter(E), nn.Parameter(F)
        self.gelu = nn.GELU()

    def forward(self, z):
        batch_size = z.shape[0]
        y = torch.randn(batch_size, self.output_dim)
        for i in range(batch_size):
            K = self.C @ materialize_kernel(self.Abar, self.E, z[i], z.shape[1])
            y[i,:] = self.gelu(K + self.F @ z[i][-1])
        return y


 

""" LS4 Prior block
"""
class LS4PriorResBlock(nn.Module):
    def __init__(self, A, input_dim, output_dim, hidden_dim, latent_dim, step) -> None:
        super().__init__()
        self.ls4 = LS4PriorLayer(A, input_dim, output_dim, hidden_dim, latent_dim, step)
        self.lin = nn.Linear(output_dim,latent_dim)
        self.norm = nn.LayerNorm([latent_dim])

    def forward(self, z):
        tmp = self.lin(self.ls4(z))
        tmp = self.norm(tmp) + z[:,-1,:]
        znew = tmp.unsqueeze(dim=1) 
        return z + znew

class LS4PriorBlock(nn.Module):
    def __init__(self, A, input_dim, output_dim, hidden_dim, latent_dim, step) -> None:
        super().__init__()
        self.ls4 = LS4PriorLayer(A, input_dim, output_dim, hidden_dim, latent_dim, step)
        self.lin = nn.Linear(output_dim,latent_dim)
        self.norm = nn.LayerNorm([latent_dim])

    def forward(self, z):
        tmp = self.lin(self.ls4(z))
        ztilde = self.norm(tmp) + z[:,-1,:]
        return ztilde 


def append_ascent(nprior, nlatent, A, input_dim, output_dim, hidden_dim, latent_dim, step):
    res = []
    for n in range(nlatent):
        for i in range(nprior):
            res.append(LS4PriorResBlock(A, input_dim, output_dim, hidden_dim, (2**(nlatent-n))*hidden_dim, step))
        res.append(nn.Linear(2**(nlatent-n)*hidden_dim, 2**(nlatent-n-1)*hidden_dim))
    return res




""" LS4 Prior Network 
"""
class LS4PriorNet(nn.Module):

    def __init__(self, Nlatent, Nprior, A, input_dim, output_dim, hidden_dim, latent_dim, step):
        super().__init__()
        self.nlatent = Nlatent
        self.nprior = Nprior
        self.latent_dim = latent_dim
        self.lintransfo = nn.Linear(latent_dim, hidden_dim)
        self.lintransfoinv = nn.Linear(hidden_dim,latent_dim)
        layers_descent = [nn.Linear((2**n) * hidden_dim, (2**(n+1)) * hidden_dim) for n in range(self.nlatent)] # -1 ?
        layers_ascent = append_ascent(self.nprior, self.nlatent, A, input_dim, output_dim, hidden_dim, latent_dim, step)
        self.descent = nn.Sequential(*layers_descent)
        self.ascent = nn.Sequential(*layers_ascent)
        # self.reparam = LS4PriorBlock(A, input_dim, output_dim, hidden_dim, latent_dim, step)
        self.reparam_mu = LS4PriorBlock(A, input_dim, output_dim, hidden_dim, latent_dim, step)
        self.reparam_sigma = LS4PriorBlock(A, input_dim, output_dim, hidden_dim, latent_dim, step)
    
    def forward(self, z):
        zh = self.lintransfo(z)
        zNlatent = self.descent(zh)
        ztilde = self.ascent(zNlatent)
        ztilde = self.lintransfoinv(ztilde)
        # zgen = self.reparam(ztilde)
        # znew = zgen.unsqueeze(dim=1)
        # return torch.cat((z,znew), dim=1)
        mu_z = self.reparam_mu(ztilde)
        sigma_z = self.reparam_sigma(ztilde)
        return mu_z, sigma_z
