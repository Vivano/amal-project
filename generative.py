from utils import *



""" Generative Layer """

class LS4GenerativeLayer(nn.Module):
    def __init__(self, A, input_dim, output_dim, hidden_dim, latent_dim, step) -> None:
        super().__init__()
        self.A = A
        self.Abar = Abar(self.A, step)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.step = step
        B_x, C_x, E_x, F_x = init_params(self.A, self.input_dim, self.hidden_dim, self.latent_dim, self.output_dim, self.step)
        B_z, C_z, E_z, F_z = init_params(self.A, self.input_dim, self.hidden_dim, self.latent_dim, self.output_dim, self.step)
        D_x, D_z = torch.rand(output_dim, input_dim), torch.rand(output_dim, input_dim)
        self.B_x, self.B_z = nn.Parameter(B_x), nn.Parameter(B_z)
        self.C_x, self.C_z = nn.Parameter(C_x), nn.Parameter(C_z)
        self.D_x, self.D_z = nn.Parameter(D_x), nn.Parameter(D_z)
        self.E_x, self.E_z = nn.Parameter(E_x), nn.Parameter(E_z)
        self.F_x, self.F_z = nn.Parameter(F_x), nn.Parameter(F_z)
        self.gelu = nn.GELU()


    def forward(self, x, z):
        batch_size = z.shape[0]
        g_x = torch.randn(batch_size, self.output_dim)
        g_z = torch.randn(batch_size, self.output_dim)
        for i in range(batch_size):
            h = materialize_kernel_generative(self.A, self.B_x, self.E_x, self.E_z, x[i], z[i], z.shape[1])
            g_x[i,:] = self.gelu(self.C_x @ h + self.D_x @ x[i][-1] + self.F_x @ z[i][-1]).squeeze(0)
            g_z[i,:] = self.gelu(self.C_z @ h + self.D_z @ x[i][-1] + self.F_z @ z[i][-1]).squeeze(0)
        return g_x, g_z



""" Generative Block """

# single element output
class LS4GenerativeBlock(nn.Module):
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
        return tmp_x, tmp_z

# sequence output
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


    
# Function stacking all the different layers allowing to get back to the initial latent space from the deep latent space in the generative network
class append_ascent_generative(nn.Module):
    def __init__(self, nprior, nlatent, A, input_dim, output_dim, hidden_dim, latent_dim, step) -> None:
        super().__init__()
        res = []
        for n in range(nlatent):
            for i in range(nprior):
                res.append(LS4GenerativeResBlock(A, (2**(nlatent-n))*hidden_dim, output_dim, hidden_dim, (2**(nlatent-n))*hidden_dim, step))
            res.append(twoLinear(n, nlatent, hidden_dim, latent_dim))
        self.res = mySequential(*res)

    def forward(self, x, z):
        return self.res(x,z)




""" Generative Net """

class LS4GenerativeNet(nn.Module):

    def __init__(self, Nlatent, Nprior, A, input_dim, output_dim, hidden_dim, latent_dim, step):
        super().__init__()
        self.nlatent = Nlatent
        self.nprior = Nprior
        self.latent_dim = latent_dim
        self.lintransfo_z = nn.Linear(latent_dim, hidden_dim)
        self.lintransfo_x = nn.Linear(input_dim, hidden_dim)
        self.lintransfoinv = nn.Linear(2*hidden_dim,input_dim)
        layers_descent = [nn.Linear((2**n) * hidden_dim, (2**(n+1)) * hidden_dim) for n in range(self.nlatent)] 
        layers_descent2 = [nn.Linear((2**n) * hidden_dim, (2**(n+1)) * hidden_dim) for n in range(self.nlatent)] 
        layers_ascent = append_ascent_generative(self.nprior, self.nlatent, A, input_dim, output_dim, hidden_dim, latent_dim, step)
        self.descent = nn.Sequential(*layers_descent)
        self.ascent = mySequential(layers_ascent)
        self.descent2 = nn.Sequential(*layers_descent2)
        self.norm = nn.LayerNorm([hidden_dim])
        self.norm2 = nn.LayerNorm([hidden_dim])
    
    def forward(self, x, z):
        zh = self.lintransfo_z(z)
        xh = self.lintransfo_x(x)
        zNlatent = self.descent(zh)
        xNlatent = self.descent2(xh)
        xtilde, ztilde = self.ascent(xNlatent, zNlatent)
        ztilde = self.norm(ztilde)
        xtilde = self.norm2(xtilde)
        xnew = torch.cat((xtilde[:,-1,:],ztilde[:,-1,:]), dim=-1)  
        xnew = self.lintransfoinv(xnew)
        return xnew
