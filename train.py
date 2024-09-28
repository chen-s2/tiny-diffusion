from unet import *

data_path = "../dit/data/train"
image_size = 32
batch_size = 1
epochs_num = 10
c_latent = 3
T=1000
time_embed_dim = 256
device = 'cuda'

model = UNet(n_channels=c_latent, time_embed_dim=time_embed_dim)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

training_loader = create_dataloader(data_path, image_size, batch_size)

train(model=model, optimizer=optimizer, loss_function=loss_function_mse, training_loader=training_loader, epochs_num=epochs_num, device=device, T=T)

print("done")