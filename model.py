from utils import *



class ModelLS4(nn.Module):

    def __init__(self, NumLatent, NumPrior, x_dim, h_dim, z_dim, y_dim, step, length) -> None:
        super().__init__()
        # invariant matrices
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.A = HiPPO(hidden_dim)
        self.length = length
        self.PriorNet = LS4PriorNet(NumLatent, NumPrior, self.A, input_dim, output_dim, hidden_dim, latent_dim, step)
        self.GenerativeNet = LS4GenerativeNet()
        self.InferenceNet = LS4InferenceNet()
        # prior distribution parameters
        self.mu_prior, self.sigma_prior = nn.Parameter(torch.randn(z_dim)), nn.Parameter(torch.randn(z_dim))
        # decoder parameters
        self.lin_mu_decod, self.sigma_decod = nn.Linear(torch.randn(x_dim)), nn.Parameter(torch.randn(x_dim))



    def forward(self, x):
        
        length = x.shape[1]
        eps_z, eps_x = torch.randn(self.z_dim), torch.randn(self.input_dim)
        z0 = self.mu_z0 + self.sigma_z0 @ eps_z
        zseq = z0.unsqueeze(1)
        x0 = self.lin_mu_x(z0) + self.sigma_x0 @ eps_x
        xseq = x0.unsqueeze(1)

        prior_means_all = [self.mu_z0]
        prior_var_all = [self.sigma_z0]
        generative_means_all = [x0]
        inference_var_all = []
        inference_means_all = []

        for t in range(length-1):
            # Prior
            z_mu, z_sigma = self.PriorNet(zseq)
            eps_z = torch.rand_like(z_mu)
            z = z_mu + z_sigma @ eps_z
            z = z.unsqueeze(1)
            zseq = torch.cat((zseq,z),1)

            # Generative
            xnext = self.GenerativeNet(xeq, zseq)
            xnextt = xnext.unsqueeze(1)
            xseq = torch.cat((xseq,xnextt),1)

            # Inference
            z_muinf, z_sigmainf = self.InferenceNet(x)

            prior_means_all.append(z_mu)
            prior_var_all.append(z_sigma)
            generative_means_all.append(xnext)
            inference_var_all.append(z_muinf)
            inference_means_all.append(z_sigmainf)

        return prior_means_all, prior_var_all, generative_means_all, inference_means_all, inference_var_all, 



    def sampling(self, seq_len):

        zseq, xseq = [], []
        z = self.reparametrizer(self.mu_prior, self.sigma_prior)
        zseq.append(z)
        x = self.reparametrizer(self.lin_mu_decod(self.mu_prior), self.sigma_decod)
        xseq.append(x)

        for t in range(seq_length):
            

