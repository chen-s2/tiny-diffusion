from unet import *

data_path = "../dit/data/train"
image_size = 32
batch_size = 16
epochs_num = 10
c_latent = 3
device = 'cuda'

model = UNetDiffusion(n_channels=c_latent)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

training_loader = create_dataloader(data_path, image_size, batch_size)

train(model=model, optimizer=optimizer, loss_function=loss_function_mse, training_loader=training_loader, epochs_num=epochs_num, device=device)

print("done")