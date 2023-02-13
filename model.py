from prior import *
from generative import *
from inference import *



class LS4Model(nn.Module):

    def __init__(self, NumLatent, NumPrior, input_dim, hidden_dim, latent_dim, output_dim, step) -> None:
        super().__init__()
        # invariant matrices
        self.input_dim = input_dim
        self.z_dim = latent_dim
        self.A = HiPPO(hidden_dim)
        self.PriorNet = LS4PriorNet(NumLatent, NumPrior, self.A, input_dim, output_dim, hidden_dim, latent_dim, step)
        self.GenerativeNet = LS4GenerativeNet(NumLatent, NumPrior, self.A, input_dim, output_dim, hidden_dim, latent_dim, step)
        self.InferenceNet = LS4InferenceNet(NumLatent, NumPrior, self.A, input_dim, output_dim, hidden_dim, latent_dim, step)
        self.mu_z0 = nn.Parameter(torch.randn(latent_dim))
        #self.sigma_z0 = nn.Parameter(torch.randn(latent_dim, latent_dim))
        self.sigma_z0 = nn.Parameter(torch.randn(latent_dim))
        #self.sigma_x0 = nn.Parameter(torch.randn(input_dim, input_dim))
        self.sigma_x0 = nn.Parameter(torch.randn(input_dim))
        self.lin_mu_x = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        length = x.shape[1]
        batch_size = x.shape[0]
        # z0
        z0 = torch.zeros(batch_size,self.z_dim)
        for i in range(batch_size):
            eps_z = torch.randn(self.z_dim)
            z0[i,:] = self.mu_z0 + self.sigma_z0 @ eps_z
        zseq = z0.unsqueeze(1)
        # x0
        x0 = torch.zeros(batch_size,self.input_dim)
        for i in range(batch_size):
            eps_x = torch.randn(self.input_dim)
            x0[i,:] = self.lin_mu_x(z0[i,:]) + self.sigma_x0 @ eps_x
        xseq = x0.unsqueeze(1)

        prior_means_all = [self.mu_z0.repeat(batch_size, 1)]
        prior_var_all = [self.sigma_z0.repeat(batch_size, 1)]
        #generative_means_all = [x0]
        inference_var_all = []
        inference_means_all = []

        for t in range(length-1):
            # Prior
            z_mu, z_sigma = self.PriorNet(zseq)
            eps_z = torch.rand_like(z_mu)
            z = z_mu + z_sigma * eps_z
            z = z.unsqueeze(1)
            zseq = torch.cat((zseq,z),1)

            # Generative
            xnext = self.GenerativeNet(xseq, zseq)
            xnextt = xnext.unsqueeze(1)
            xseq = torch.cat((xseq,xnextt),1)

            # Inference
            z_muinf, z_sigmainf = self.InferenceNet(x)

            prior_means_all.append(z_mu)
            prior_var_all.append(z_sigma)
            #generative_means_all.append(xnext)
            inference_var_all.append(z_muinf)
            inference_means_all.append(z_sigmainf)


        return prior_means_all, prior_var_all, inference_means_all, inference_var_all, xseq



    # def sampling(self, seq_len):

    #     zseq, xseq = [], []
    #     z = self.reparametrizer(self.mu_prior, self.sigma_prior)
    #     zseq.append(z)
    #     x = self.reparametrizer(self.lin_mu_decod(self.mu_prior), self.sigma_decod)
    #     xseq.append(x)

    #     for t in range(seq_length):
            

