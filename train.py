from model import *
from loss import *
import pandas as pd

hidden_dim = 5
latent_dim = 2
input_dim = 1
output_dim = 1
step = 0.1
length = 10
N = 100
batch_size = 20
NumLatent = 2
NumPrior = 2

n_epochs = 10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LS4Model(NumLatent, NumPrior, input_dim, hidden_dim, latent_dim, output_dim, step)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

df = pd.read_csv("~/Documents/GitHub/amal-project/data/nn5.tsf", sep=',', header=None)
data = df.iloc[:,2:] # deleting the first column where time and data are concatenated
X = torch.tensor(data.values, dtype=torch.float).unsqueeze(-1)
y = torch.randn(N,output_dim)
train_dataset = TimeseriesDataset(X, y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
print(X.shape)



print('start')

for t in range(n_epochs):

    for data, _ in train_loader:

        print(data.shape)
        package = model(data)
        loss = Loss(package, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t%10 == 0 :
            print(f"Epoch {t+1} : loss = {loss.item()}")

print('end')