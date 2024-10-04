from unet import *
from dataset import *
from training import train, loss_function_mse

if __name__ == "__main__":
    image_size = 48
    batch_size = 64
    epochs_num = 10
    c_latent = 1
    T = 1000
    load_model_path = get_last_created_model()
    time_emb_dim = image_size
    device = 'cuda'

    if load_model_path:
        model = torch.load(load_model_path)
        print("loading model:", load_model_path)
    else:
        model = UNet(n_channels=c_latent, time_emb_dim_param=time_emb_dim, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0)

    training_loader = create_dataloader("./data/butterfly/train", image_size, batch_size, dataset_name="butterfly")
    # training_loader = create_dataloader_cifar(image_size, batch_size)

    train(model=model, optimizer=optimizer, loss_function=loss_function_mse, training_loader=training_loader, epochs_num=epochs_num, device=device, T=T)

    print("done")