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

    def __init__(self, nlatent, nprior, A, input_dim, hidden_dim, latent_dim, output_dim, step) -> None:
        super().__init__()
        self.lin1_x = nn.Linear(input_dim, hidden_dim)
        self.lin1_z = nn.Linear(latent_dim, hidden_dim)
        ascent = append_ascent_generative(nprior, nlatent, A, input_dim, output_dim, hidden_dim, latent_dim, step)
        # ascent_z = append_ascent(nprior, nlatent, A, input_dim, output_dim, hidden_dim, latent_dim, step)
        self.ascent = mySequential(*ascent)
        # self.ascent_z = nn.Sequential(*ascent_z)
        descent_x = [nn.Linear((2**n) * hidden_dim, (2**(n+1)) * hidden_dim) for n in range(nlatent-1)]
        descent_z = [nn.Linear((2**n) * hidden_dim, (2**(n+1)) * hidden_dim) for n in range(nlatent-1)]
        self.descent_x = nn.Sequential(*descent_x)
        self.descent_z = nn.Sequential(*descent_z)
        self.norm_x = nn.LayerNorm([input_dim])
        self.norm_z = nn.LayerNorm([hidden_dim])

    def forward(self, x, z):
        xh, zh = self.lin1_z(x), self.lin1_x(z)
        xNlatent, zNlatent = self.descent_x(xh), self.descent_z(zh)
        xtilde, ztilde = self.ascent(xNlatent, zNlatent)
        xtilde, ztilde = self.norm_x(xtilde), self.norm_z(ztilde)
        xnew = torch.cat((xtilde[:,-1,:],ztilde[:,-1,:]), dim=-1)  
        xnew = self.lintransfoinv(xnew)
        return xnew
