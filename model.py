from utils import *



class ModelLS4(nn.Module):

    def __init__(self, NumLatent, NumPrior, input_dim, hidden_dim, latent_dim, output_dim, step, length) -> None:
        super().__init__()
        # invariant matrices
        self.input_dim = input_dim
        self.z_dim = latent_dim
        self.A = HiPPO(hidden_dim)
        self.length = length
        self.PriorNet = LS4PriorNet(NumLatent, NumPrior, self.A, input_dim, output_dim, hidden_dim, latent_dim, step)
        self.GenerativeNet = LS4GenerativeNet()
        self.InferenceNet = LS4InferenceNet()
        self.mu_z0 = nn.Parameter(torch.randn(latent_dim))
        self.sigma_z0 = nn.Parameter(latent_dim)
        self.sigma_x0 = nn.Parameter(input_dim)
        self.lin_mu_x = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        eps_z, eps_x = torch.randn(self.z_dim), torch.randn(self.input_dim)
        z0 = self.mu_z0 + self.sigma_z0 * eps_z
        zseq = [z0]
        x0 = self.lin_mu_x(z0) + self.sigma_x0 * eps_x
        xseq = [x0]
        for t in range(self.length):
            znext = self.PriorNet(torch.stack(zseq))
            zseq.append(znext)
            xnext = self.GenerativeNet(torch.stack(xseq), torch.stack(zseq))
            xseq.append(xnext)
        return xseq
