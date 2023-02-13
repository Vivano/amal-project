from utils import *
import torch.distributions.normal as Norm
import torch.distributions.kl as KL



""" Loss Function """

def Loss(package, x):

    prior_means, prior_var, decoder_means, decoder_var, x_decoded = package
    loss = 0.
    loss_fn = torch.nn.CrossEntropyLoss()
    for i in range(x.shape[1]-1):
        # Kl loss
        for j in range(x.shape[0]):
            norm_dis1 = Norm.Normal(prior_means[i][j], torch.abs(prior_var[i][j]))
            norm_dis2 = Norm.Normal(decoder_means[i][j], torch.abs(decoder_var[i][j]))
            kl_loss = torch.mean(KL.kl_divergence(norm_dis1, norm_dis2))

            # reconstruction loss
            loss += kl_loss
        xent_loss = loss_fn(x_decoded[:, i, :], x[:, i, :]).mean()
        loss += xent_loss

    return loss