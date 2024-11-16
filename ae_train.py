import numpy as np
import torch
from autoencoder import Autoencoder
from tqdm import tqdm

lr = 1e-4
epochs = 50
T = 1000
training_loader = None
device = 'cuda'

ae_model = Autoencoder()
normal = None

def KL(p1, p2):
    pass

def L2(x1,x2):
    pass

def loss_func(x, x_rec, z):
    l = L2(x,x_rec) + KL(z, normal)  # we want both distances to be the smallest
    return l

def train(ae_model, training_loader, optimizer, epochs, T, device):
    running_loss = 0.
    clean_t_running_loss = 0 # running loss for the 5% smallest values of t: [0,0.05*T]
    last_loss = np.inf
    clean_t_last_loss = np.inf
    clean_t_samples = 0

    beta = np.linspace(1e-4, 0.02, num=T)

    ae_model.to(device)
    ae_model.train()

    dataset_name = training_loader.dataset_name

    for epoch in range(epochs):
        ts = []
        for x in tqdm(training_loader):
            optimizer.zero_grad()

            x = x.to(device)

            t = int(torch.randint(1, T, (1,), device=device).to(x.dtype))

            ts.append(t)

            alpha_1_to_t_array = []
            for i in range(1,t+1):
                alpha_t = 1-beta[i]
                alpha_1_to_t_array.append(alpha_t)
            alpha_t_bar = torch.prod(torch.Tensor(alpha_1_to_t_array))

            z = ae_model.encode(x)

            epsilon = torch.randn(z.shape, device=device).to(z.dtype)

            noisy_z = torch.sqrt(alpha_t_bar) * z + torch.sqrt(1 - alpha_t_bar) * epsilon

            x_rec = ae_model.decode(noisy_z)

            loss = loss_func(x_rec, x, z)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if t <= int(0.05 * T):
                clean_t_running_loss += loss.item()
                clean_t_samples += 1

        last_loss = running_loss / len(training_loader)
        print("loss:", last_loss)
        print("clean_t loss:", clean_t_last_loss)
        running_loss = 0

        if epoch % 10 == 0 and epoch > 0:
            model_name = './models/ae_model_' + str(round(last_loss, 4)) + "_cleanloss_" + str(round(clean_t_last_loss, 4)) + '.pth'
            torch.save(ae_model, model_name)
            print('saved model to:', model_name)

        if clean_t_samples > 20:
            clean_t_last_loss = clean_t_running_loss / clean_t_samples
            clean_t_running_loss = 0
            clean_t_samples = 0

optimizer = torch.optim.AdamW(ae_model.parameters(), lr=lr, weight_decay=1e-2)
train(ae_model, training_loader, optimizer, epochs, T, device)