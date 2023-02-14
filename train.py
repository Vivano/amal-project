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

X = torch.randn(N,length,input_dim)
y = torch.randn(N,output_dim)
toy_dataset = TimeseriesDataset(X, y)
toy_loader = torch.utils.data.DataLoader(toy_dataset, batch_size = batch_size, shuffle = False)

n_epochs = 50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = LS4Model(NumLatent, NumPrior, input_dim, hidden_dim, latent_dim, output_dim, step)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

df_nn5 = pd.read_csv("~/Documents/GitHub/amal-project/data/nn5.tsf", sep=',', header=None)
data_nn5 = df_nn5.iloc[:,2:]
df_temp_rain = pd.read_csv("~/Documents/GitHub/amal-project/data/temp_rain.tsf", sep=',', header=None)
data_temp_rain = df_temp_rain.iloc[:,4:]
df_hospital = pd.read_csv("~/Documents/GitHub/amal-project/data/hospital_dataset.tsf", sep=',', header=None)
data_hospital = df_hospital.iloc[:,2:]

DAT = "hospital"
if DAT == "nn5":
    data = data_nn5.copy()
elif DAT == "temp_rain":
    data = data_temp_rain.copy()
elif DAT == "hospital":
    data = data_hospital.copy()

X = torch.tensor(data.values, dtype=torch.float).unsqueeze(-1)
y = torch.randn(N,output_dim)
train_dataset = TimeseriesDataset(X, y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = False, drop_last=True)
# print(X.shape)



print('start')

for t in range(n_epochs):

    for data, _ in toy_loader:

        # print(data.shape)
        package = model(data)
        loss = Loss(package, data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if t==0 or (t+1)%10 == 0 :
        print(f"Epoch {t+1} : loss = {loss.item()}")

# print(f"Distribution :  mu = {model.lin_mu_x(model.mu_z0)}, sigma = {model.sigma_x0}")

print('end')
