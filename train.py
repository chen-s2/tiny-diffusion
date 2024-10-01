from unet import *

data_path = "../dit/data/train"
image_size = 48
batch_size = 32
epochs_num = 20
c_latent = 3
T = 200
load_model_path = './models/model_0.5655.pth'
time_emb_dim = image_size
device = 'cuda'


if load_model_path:
    model = torch.load(load_model_path)
    print("loading model:", load_model_path)
else:
    model = UNet(n_channels=c_latent, time_emb_dim_param=time_emb_dim, device=device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

training_loader = create_dataloader(data_path, image_size, batch_size)

train(model=model, optimizer=optimizer, loss_function=loss_function_mse, training_loader=training_loader, epochs_num=epochs_num, device=device, T=T)

print("done")