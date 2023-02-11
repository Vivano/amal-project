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
    def __init__(self, input_dim, output_dim, hidden_dim, latent_dim, time_step, A) -> None:
        super().__init__()
        self.p = input_dim
        self.p_ = output_dim
        self.N = hidden_dim
        self.H = latent_dim
        self.step = time_step
        self.A = A
        self.A, B, C, E, F = init_matrices(self.p, self.N, self.H, self.p_, self.step)
        self.B, self.C = nn.Parameter(B), nn.Parameter(C)
        self.E, self.F = nn.Parameter(E), nn.Parameter(F)
        self.gelu = nn.GELU()

    def forward(self, z):
        batch_size = z.shape[0]
        length = z.shape[1]
        y = torch.randn(batch_size, self.p_)
        for i in range(batch_size):
            CK = self.C @ materialize_kernel(self.A, self.E, z[i], length)
            y[i,:] = self.gelu(CK + self.F @ z[i][-1])
        return y



""" LS4 Prior block
"""
class LS4PriorBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, latent_dim, time_step) -> None:
        super().__init__()
        self.ls4 = LS4PriorLayer(input_dim, output_dim, hidden_dim, latent_dim, time_step)
        self.lin
        self.lin = nn.Linear(latent_dim, output_dim)
        self.norm = nn.LayerNorm(latent_dim)
    def forward(self, z):
        tmp = self.lin(self.ls4(z))
        tmp = self.norm(tmp) + z[:,-1,:]
        return tmp


def append_ascent(nprior, nlatent, input_dim, output_dim, hidden_dim, latent_dim, time_step):
    res = []
    for n in range(1, nlatent):
        for i in range(nprior):
            res.append(LS4PriorLayer(input_dim, output_dim, hidden_dim, (2**(nlatent-(n+1)))*hidden_dim, time_step))
        res.append(nn.Linear((2**(nlatent-(n+1)))*hidden_dim, (2**(nlatent-n))*hidden_dim))
    return res


""" LS4 Prior Network 
"""
class LS4PriorNet(nn.Module):

    def __init__(self, Nlatent, Nprior, input_dim, output_dim, hidden_dim, latent_dim, time_step):
        super().__init__()
        self.nlatent = Nlatent
        self.nprior = Nprior
        self.latent_dim = latent_dim
        self.lintransfo = nn.Linear()
        layers_descent = [nn.Linear((2**n) * hidden_dim, (2**(n+1)) * hidden_dim) for n in range(self.nlatent)] # -1 ?
        layers_ascent = append_ascent(self.nprior, self.nlatent, input_dim, output_dim, hidden_dim, latent_dim, time_step)
        self.descent = nn.Sequential(*layers_descent)
        self.ascent = nn.Sequential(*layers_ascent)
        self.reparam = LS4PriorBlock(input_dim, output_dim, hidden_dim, latent_dim, time_step)
        # self.mu = nn.Parameter(torch.randn(latent_dim))
        # self.sigma = nn.Parameter(torch.randn(latent_dim, latent_dim))
        # self.length = length
    
    def forward(self, z):
        zh = nn.lintransfo(z)
        zNlatent = self.descent(zh)
        ztilde = self.ascent(zNlatent)
        zgen = self.reparam(ztilde)
        znew = zgen.unsqueeze(dim=1)
        return torch.cat((z,znew), dim=1)



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
class GenerativeNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mu = nn.Parameter(torch.tensor(input_dim))

    def forward(self, x, z):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x
    # c'est pas fini


def beta_parameter(A, input_dim, hidden_dim, latent_dim, nlatent, nprior, step):
    B, E = init_beta(A, input_dim, hidden_dim, latent_dim, nlatent, nprior, step)
    Bparam, Eparam = nn.ParameterList([nn.Parameter(b) for b in B]), nn.ParameterList([nn.ParameterList([nn.Parameter(e) for e in elist]) for elist in E])
    return Bparam, Eparam

class ModelLS4(nn.Module):

    def __init__(self, NumLatent, NumPrior, input_dim, hidden_dim, latent_dim, output_dim, step, length) -> None:
        super().__init__()
        # self.step = time_step
        # self.length = length
        # number of latent and prior layers
        self.nlatent = NumLatent
        self.nprior = NumPrior
        # dimensions
        # self.input_dim = input_dim
        # self.hidden_dim = hidden_dim
        # self.latent_dim = latent_dim
        # self.output_dim = output_dim
        # invariant matrices
        self.A = HiPPO(hidden_dim)
        self.Abar = Abar(self.A, hidden_dim, step)
        # PARAMETER MATRICES
        # prior block
        self.B1, self.E1 = None, None
        self.B2, self.E2 = beta_parameter(self.A, input_dim, hidden_dim, self.nlatent, self.nprior, step)
        self.Cprior = nn.ParameterList([nn.Parameter(torch.randn(output_dim, hidden_dim))] for n in range(self.nprior))
        self.Fprior = nn.ParameterList([nn.ParameterList([nn.Parameter(torch.randn(output_dim, 2**k * hidden_dim)) for k in range(self.nlatent)]) for n in range(self.nprior)])
        # generative block
        self.B3, self.E3 = beta_parameter(self.A, input_dim, hidden_dim, self.nlatent, self.nprior, step)
        self.B4, self.E4 = beta_parameter(self.A, input_dim, hidden_dim, self.nlatent, self.nprior, step)
        self.Cgenx, self.Cgenz = nn.ParameterList([nn.Parameter(torch.randn(output_dim, hidden_dim))] for n in range(self.nprior)), nn.ParameterList([nn.Parameter(torch.randn(output_dim, hidden_dim))] for n in range(self.nprior))
        self.Dgenx, self.Dgenz = nn.ParameterList([nn.ParameterList([nn.Parameter(torch.randn(output_dim, 2**n * hidden_dim)) for n in range(self.nlatent)]) for i in range(self.nprior)])
        # inference block
        self.Binf = None # to change
        self.Cinf = None # to change
        self.Dinf = None # to change


        def PriorModel(self):
            # TO DO
            return None
        
        def GenerativeModel(self):
            # TO DO
            # pas oublier de mapper
            return None

        def InferenceModel(self):
            # TO DO
            return None
