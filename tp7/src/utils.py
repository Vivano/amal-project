import torch
import torch.nn as nn
from datamaestro import prepare_dataset
from torch.utils.data import Dataset


class MNIST(Dataset):
    def __init__(self, ratio = 0.5):
        ds = prepare_dataset("com.lecun.mnist");
        train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
        test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()
        
        self.n = len(train_images)
        self.r = ratio
        
        self.X = torch.tensor(train_images[:int(self.n * self.r)]).view(-1,28*28)/255
        self.y = torch.tensor(train_labels[:int(self.n * self.r)])

    def __getitem__(self,index):
        return self.X[index], self.y[index]
        
    def __len__(self):
        return self.X.shape[0]



class NET(nn.Module):

	def __init__(self, input_size, hidden_sizes, output_size, ratio_dropout=0., batch_norm=False, layer_norm=False):
		super.__init__(self)
		self.input_size = input_size
		self.hidden_sizes = hidden_sizes
		self.output_size = output_size
		self.ratio = ratio_dropout
		self.batch_norm = batch_norm
		self.layer_norm = layer_norm
		if batch_norm and layer_norm:
			self.batch_norm = False
			self.layer_norm = False
		self.n_hidden = len(hidden_sizes)
		self.layers = nn.ModuleList([nn.Linear(self.input_size, self.hidden_sizes[0]), nn.ReLu()])
		self.tracked_layers = nn.ModuleList([nn.Linear(self.input_size, self.hidden_sizes[0])])
		for i in range(slef.n_hidden):
			self.layers.extend(
				[
					nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]), 
					nn.Dropout(self.ratio),
					nn.BatchNorm1d(hidden_sizes[i+1]) if self.batch_norm,
					nn.layer_norm
					nn.ReLu()
				]
			)
			self.tracked_layers.append(layers[-1])
		self.layers.append(nn.Linear(self.hidden_sizes[-1], self.output_size))
		self.tracked_layers(self.hidden_sizes[-1], self.output_size)


	def forward(self, x):
		grad = []
		for l in self.layers:
			x = l(x)
			#if l in self.tracked_layers:
			grad.append(store_grad(x))
		return x, grad





def store_grad(var):
    """Stores the gradient during backward

    For a tensor x, call `store_grad(x)`
    before `loss.backward`. The gradient will be available
    as `x.grad`

    """
    def hook(grad):
        var.grad = grad
    var.register_hook(hook)
    return var



def l1_reg(params):
	l1_penalty = torch.nn.L1Loss(size_average=False, reduction='sum')
	reg_loss = l1_penalty(param)
	return reg_loss