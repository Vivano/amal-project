class LS4InferenceLayer(nn.Module):
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
        D = torch.rand(output_dim, input_dim)
        self.D = nn.Parameter(D)
        self.gelu = nn.GELU()

    def forward(self, x):
        batch_size = x.shape[0]
        y = torch.randn(batch_size, self.output_dim)
        for i in range(batch_size):
            K = self.C @ materialize_kernel(self.Abar, self.B, x[i], x.shape[1])
            y[i,:] = self.gelu(K + self.D @ x[i][-1])
        return y




""" hidden_dim = 5
latent_dim = 2
input_dim = 1
output_dim = 1
time_step = 0.1
length = 5
batch_size = 20
A = HiPPO(hidden_dim)

model = LS4InferenceLayer(A, input_dim, output_dim, hidden_dim, latent_dim, time_step)
x = torch.rand(batch_size,length,input_dim)
print(model(x))  """


class LS4InferenceBlock(nn.Module):
    def __init__(self, A, input_dim, output_dim, hidden_dim, latent_dim, step) -> None:
        super().__init__()
        self.ls4 = LS4InferenceLayer(A, input_dim, output_dim, hidden_dim, latent_dim, step)
        self.lin = nn.Linear(output_dim,input_dim)
        self.norm = nn.LayerNorm([input_dim])

    def forward(self, x):
        tmp = self.lin(self.ls4(x))
        xtilde = self.norm(tmp) + x[:,-1,:]
        return xtilde 


""" hidden_dim = 5
latent_dim = 2
input_dim = 1
output_dim = 1
time_step = 0.1
length = 5
batch_size = 20
A = HiPPO(hidden_dim)
model = LS4InferenceBlock(A, input_dim, output_dim, hidden_dim, latent_dim, time_step)
x = torch.rand(batch_size,length,input_dim)
print(model(x))  """

class LS4InferenceResBlock(nn.Module):
    def __init__(self, A, input_dim, output_dim, hidden_dim, latent_dim, step) -> None:
        super().__init__()
        self.ls4 = LS4InferenceLayer(A, input_dim, output_dim, hidden_dim, latent_dim, step)
        self.lin = nn.Linear(output_dim,input_dim)
        self.norm = nn.LayerNorm([input_dim])

    def forward(self, x):
        tmp = self.lin(self.ls4(x))
        xtilde = self.norm(tmp) + x[:,-1,:]
        xnew = xtilde.unsqueeze(dim=1) 
        return x + xnew

def append_ascent_inference(nprior, nlatent, A, input_dim, output_dim, hidden_dim, latent_dim, step):
    res = []
    for n in range(nlatent):
        for i in range(nprior):
            res.append(LS4InferenceResBlock(A, (2**(nlatent-n))*hidden_dim, output_dim, hidden_dim, latent_dim, step))
        res.append(nn.Linear((2**(nlatent-n))*hidden_dim, (2**(nlatent-n-1))*hidden_dim))
    return res

class LS4InferenceNet(nn.Module):

    def __init__(self, Nlatent, Nprior, A, input_dim, output_dim, hidden_dim, latent_dim, step):
        super().__init__()
        self.nlatent = Nlatent
        self.nprior = Nprior
        self.latent_dim = latent_dim
        self.lintransfo = nn.Linear(input_dim, hidden_dim)
        self.lintransfoinv = nn.Linear(hidden_dim,input_dim)
        layers_descent = [nn.Linear((2**n) * hidden_dim, (2**(n+1)) * hidden_dim) for n in range(self.nlatent)] 
        layers_ascent = append_ascent_inference(self.nprior, self.nlatent, A, input_dim, output_dim, hidden_dim, latent_dim, step)
        self.descent = nn.Sequential(*layers_descent)
        self.ascent = nn.Sequential(*layers_ascent)
        self.reparam_mu = LS4InferenceBlock(A, input_dim, output_dim, hidden_dim, latent_dim, step)
        self.reparam_sigma = LS4InferenceBlock(A, input_dim, output_dim, hidden_dim, latent_dim, step)
        self.fc1 = nn.Linear(input_dim,latent_dim)
        self.fc2 = nn.Linear(input_dim,latent_dim)

    
    def forward(self, x):
        xh = self.lintransfo(x)
        xNlatent = self.descent(xh)
        xtilde = self.ascent(xNlatent)
        ztilde = self.lintransfoinv(xtilde)
        #print(ztilde.shape)
        z_mu = self.reparam_mu(ztilde)
        z_sigma = self.reparam_sigma(ztilde)
        z_mu = self.fc1(z_mu)
        z_sigma = self.fc2(z_sigma)
        #print(z_mu.shape, z_sigma.shape)
        return z_mu, z_sigma
