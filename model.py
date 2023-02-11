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
        self.Abar = Abar(A, hidden_dim, step)
        B, C, E, F = init_params(self.A, input_dim, hidden_dim, latent_dim, output_dim, step)
        self.B, self.C = nn.Parameter(B), nn.Parameter(C)
        self.E, self.F = nn.Parameter(E), nn.Parameter(F)
        self.gelu = nn.GELU()
    def forward(self, z):
        batch_size = z.shape[0]
        length = z.shape[1]
        y = torch.randn(batch_size, self.output_dim)
        for i in range(batch_size):
            print(i,z[i].shape)
            K = self.C @ materialize_kernel(self.Abar, self.E, z[i], length)
            y[i,:] = self.gelu(K + self.F @ z[i][-1])
        return 



""" LS4 Prior block
"""
class LS4PriorBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, latent_dim, time_step) -> None:
        super().__init__()
        self.ls4 = LS4PriorLayer(input_dim, output_dim, hidden_dim, latent_dim, time_step)
        self.lin
        self.lin = nn.Linear(output_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
    def forward(self, z):
        tmp = self.lin(self.ls4(z))
        ztilde = self.norm(tmp) + z[:,-1,:]
        return ztilde 
class LS4PriorResBlock(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, latent_dim, time_step) -> None:
        super().__init__()
        self.ls4 = LS4PriorLayer(input_dim, output_dim, hidden_dim, latent_dim, time_step)
        self.lin = nn.Linear(output_dim, latent_dim)
        self.norm = nn.LayerNorm(latent_dim)
    def forward(self, z):
        tmp = self.lin(self.ls4(z))
        ztilde = self.norm(tmp) + z[:,-1,:]
        ztilde = ztilde.unsqueeze(dim=1)
        return z + ztilde 


def append_ascent(nprior, nlatent, A, input_dim, output_dim, hidden_dim, latent_dim, step):
    res = []
    for n in range(1, nlatent+1):
        for i in range(nprior):
            res.append(LS4PriorBlock(A, input_dim, output_dim, hidden_dim, (2**(nlatent-n))*hidden_dim, step))
        res.append(nn.Linear(2**(nlatent-n)*hidden_dim, 2**(nlatent-n-1)*hidden_dim))
    res.append(nn.Linear(hidden_dim, latent_dim))
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
        layers_descent = [nn.Linear((2**n) * hidden_dim, (2**(n+1)) * hidden_dim) for n in range(self.nlatent)] # -1 ?
        layers_ascent = append_ascent(self.nprior, self.nlatent, A, input_dim, output_dim, hidden_dim, latent_dim, step)
        self.descent = nn.Sequential(*layers_descent)
        self.ascent = nn.Sequential(*layers_ascent)
        self.reparam = LS4PriorBlock(A, input_dim, output_dim, hidden_dim, latent_dim, step)
        # self.mu = nn.Parameter(torch.randn(latent_dim))
        # self.sigma = nn.(torch.randn(latent_dim, latent_dim))
        # self.length = length
    
    def forward(self, z):
        zh = self.lintransfo(z)
        zNlatent = self.descent(zh)
        ztilde = self.ascent(zNlatent)
        zgen = self.reparam(ztilde)
        znew = zgen.unsqueeze(dim=1)
        return torch.cat((z,znew), dim=1)



# """ LS4 Prior block for multidimensional variables
# """
# class LS4PriorMulti(nn.Module):
#     def __init__(self, C, psi):
#         super(LS4PriorMulti, self).__init__()
#         self.C = C
#         self.LS4_params = psi.LS4_params
#         self.linear = nn.Linear(C, C)
#         # conv_layer = torch.nn.Conv2d(3, 3, kernel_size=1) # to replace the linear 
#         self.LS4_prior_layer = LS4PriorNet(*self.LS4_params)
        
#     def forward(self, z):
#         # z: (B, L, C)
#         for c in range(self.C):
#             z[:,:,c] = self.LS4_prior_layer(z[:,:,c])
#         z = self.linear(z) # (B, L, C) channel-wise mixing
#         return z




##################################### GENERATIVE NETWORK #######################################################



""" LS4 Generative Layer
"""
class LS4GenerativeLayer(nn.Module):

    def __init__(self, A, input_dim, output_dim, hidden_dim, latent_dim, step) -> None:
        super().__init__()
        self.Abar = Abar(A, hidden_dim, step)
        B, C_x, E1, F_x = init_params(A, input_dim, hidden_dim, latent_dim, output_dim, step)
        _, C_z, E2, F_z = init_params(A, input_dim, hidden_dim, latent_dim, output_dim, step)
        D_x, D_z = torch.rand(output_dim, input_dim), torch.rand(output_dim, input_dim)
        self.B = nn.Parameter(B)
        self.C_x, self.C_z = nn.Parameter(C_x), nn.Parameter(C_z)
        self.D_x, self.D_z = nn.Parameter(D_x), nn.Parameter(D_z)
        self.E1, self.E2 = nn.Parameter(E1), nn.Parameter(E2)
        self.F_x, self.F_z = nn.Parameter(F_x), nn.Parameter(F_z)
        self.gelu = nn.GELU()

    def forward(self, x, z):
        batch_size = z.shape[0]
        length = z.shape[1]
        g_x = torch.randn(batch_size, self.p_)
        g_z = torch.randn(batch_size, self.p_)
        for i in range(batch_size):
            h = materialize_kernel_generative(self.A, self.B, self.E1, self.E2, x[i], z[i], length)
            g_x[i,:] = self.gelu(self.C_x @ h + self.D_x @ x[i][-1] + self.F_x @ z[i][-1])
            g_z[i,:] = self.gelu(self.C_z @ h + self.D_z @ x[i][-1] + self.F_z @ z[i][-1])
        return g_x, g_z



""" Generative Block
"""
class LS4GenerativeBlock(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim, latent_dim, time_step) -> None:
        super().__init__()
        self.ls4 = LS4GenerativeLayer(input_dim, output_dim, hidden_dim, latent_dim, time_step)
        self.lin_z = nn.Linear(output_dim, latent_dim)
        self.lin_x = nn.Linear(output_dim, latent_dim)
        self.norm_z = nn.LayerNorm([latent_dim])
        self.norm_x = nn.LayerNorm([input_dim])

    def forward(self, x, z):
        tmp_x, tmp_z = self.ls4(x, z)
        tmp_z = self.norm_z(self.lin_z(tmp_z)) + z[:,-1,:]
        tmp_x = self.norm_x(self.lin_x(tmp_x)) + x[:,-1,:]
        return tmp_x, tmp_z
""" Residual Generative Block
"""
class LS4GenerativeResBlock(nn.Module):
    def __init__(self, A, input_dim, output_dim, hidden_dim, latent_dim, time_step) -> None:
        super().__init__()
        self.ls4 = LS4GenerativeLayer(A, input_dim, output_dim, hidden_dim, latent_dim, time_step)
        self.lin_z = nn.Linear(output_dim,latent_dim)
        self.lin_x = nn.Linear(output_dim,input_dim)
        self.norm_z = nn.LayerNorm([latent_dim])
        self.norm_x = nn.LayerNorm([input_dim])

    def forward(self, x, z):
        tmp_x, tmp_z = self.ls4(x, z)
        tmp_z = self.norm_z(self.lin_z(tmp_z)) + z[:,-1,:]
        tmp_x = self.norm_x(self.lin_x(tmp_x)) + x[:,-1,:]
        tmp_x, tmp_z = tmp_x.unsqueeze(dim=1), tmp_z.unsqueeze(dim=1)
        return x + tmp_x, z + tmp_z


# Function stacking all the different layers allowing to get back to the initial latent space from the deep latent space
def append_ascent_generative(nprior, nlatent, A, input_dim, output_dim, hidden_dim, latent_dim, step):
    nnseq = []
    for n in range(1, nlatent+1):
        for i in range(nprior):
            nnseq.append(LS4GenerativeResBlock(A, 2**(nlatent-n)*hidden_dim, output_dim, hidden_dim, (2**(nlatent-n))*hidden_dim, step))
        nnseq.append(twoLinear((2**(nlatent-n))*hidden_dim, (2**(nlatent-n-1))*hidden_dim))
    return nnseq


""" Generative Net 
"""
class LS4GenerativeNet(nn.Module):

    def __init__(self, A, nprior, nlatent, input_dim, hidden_dim, latent_dim, output_dim, step) -> None:
        super().__init__()
        self.lin1_x = nn.Linear(input_dim, hidden_dim)
        self.lin1_z = nn.Linear(latent_dim, hidden_dim)
        ascent = append_ascent_generative(nprior, nlatent, A, input_dim, output_dim, hidden_dim, latent_dim, step)
        # ascent_z = append_ascent(nprior, nlatent, A, input_dim, output_dim, hidden_dim, latent_dim, step)
        self.ascent_x = mySequential(*ascent)
        # self.ascent_z = nn.Sequential(*ascent_z)
        descent_x = [nn.Linear((2**n) * hidden_dim, (2**(n+1)) * hidden_dim) for n in range(nlatent)]
        descent_z = [nn.Linear((2**n) * hidden_dim, (2**(n+1)) * hidden_dim) for n in range(nlatent)]
        self.descent_x = nn.Sequential(*descent_x)
        self.descent_z = nn.Sequential(*descent_z)
        self.norm_x = nn.LayerNorm([input_dim])
        self.norm_z = nn.LayerNorm([hidden_dim])

    def forward(self, x, z):
        xh, zh = self.lin1_z(x), self.lin1_x(z)
        xNlatent, zNlatent = self.descent_x(xh), self.descent_z(zh)
        xtilde, ztilde = self.ascent(xNlatent, zNlatent)
        xtilde, ztilde = self.norm_x(xtilde), self.norm_z(ztilde)
        xnew = torch.cat((xtilde,ztilde), dim=-1)  
        xnew = self.lintransfoinv(xnew)
        return xnew




##################################### INFERENCE NETWORK #######################################################


class LS4InferenceNet(nn.Module):
    
    pass



##################################### LS4 MODEL NETWORK #######################################################



class ModelLS4(nn.Module):

    def __init__(self, NumLatent, NumPrior, input_dim, hidden_dim, latent_dim, output_dim, step, length) -> None:
        super().__init__()
        # invariant matrices
        self.input_dim = input_dim
        self.z_dim = latent_dim
        self.A = HiPPO(hidden_dim)
        self.PriorNet = LS4PriorNet(NumLatent, NumPrior, self.A, input_dim, output_dim, hidden_dim, latent_dim, step)
        self.GenerativeNet = LS4GenerativeNet()
        self.InferenceNet = LS4InferenceNet()
        self.mu_z0 = nn.Parameter(torch.randn(latent_dim))
        self.sigma_z0 = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.sigma_x0 = nn.Parameter(torch.randn(input_dim, input_dim))
        self.lin_mu_x = nn.Linear(latent_dim, input_dim)


    def forward(self, length):
        eps_z, eps_x = torch.randn(self.z_dim), torch.randn(self.input_dim)
        z0 = self.mu_z0 + self.sigma_z0 @ eps_z
        zseq = [z0]
        x0 = self.lin_mu_x(z0) + self.sigma_x0 @ eps_x
        xseq = [x0]
        for t in range(length):
            znext = self.PriorNet(torch.stack(zseq))
            zseq.append(znext)
            xnext = self.GenerativeNet(torch.stack(xseq), torch.stack(zseq))
            xseq.append(xnext)
        return xseq
