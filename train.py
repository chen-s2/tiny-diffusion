from unet import *
from dataset import *
from training import train, loss_function_mse

if __name__ == "__main__":
    image_size = 64
    batch_size = 64
    epochs_num = 100
    c_latent = 3
    T = 1000
    gray = False
    load_model_path = None # get_last_created_model()
    apply_attn = [False, False, True, False]
    model_metadata = "rgb"
    time_emb_dim = image_size
    device = 'cuda'

    if load_model_path:
        model = torch.load(load_model_path)
        print("loading model:", load_model_path)
    else:
        model = UNet(n_channels=c_latent, time_emb_dim_param=time_emb_dim, device=device, apply_attn=apply_attn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

    training_loader = create_dataloader("./data/butterfly/train_and_test", image_size, batch_size, dataset_name="butterfly", gray=gray)
    # training_loader = create_dataloader_cifar(image_size, batch_size)

    train(model=model, optimizer=optimizer, loss_function=loss_function_mse, training_loader=training_loader, epochs_num=epochs_num, device=device, T=T, model_metadata=model_metadata)

    print("done")