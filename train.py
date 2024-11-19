from unet import *
from dataset import *
from training import train, loss_function_mse

if __name__ == "__main__":
    image_size = 64
    batch_size = 64

    epochs_num = 50
    channels = 1
    T = 1000

    load_model_path = None  # or resume using get_last_created_model()
    apply_attn = [False, False, True, False]
    model_metadata = "rgb"
    time_emb_dim = image_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if load_model_path:
        model = torch.load(load_model_path)
        print("loading model:", load_model_path)
    else:
        model = UNet(n_channels=channels, time_emb_dim_param=time_emb_dim, device=device, apply_attn=apply_attn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.5e-4, weight_decay=1e-2)

    training_loader = create_dataloader("./data/butterfly/train_and_test", image_size, batch_size, dataset_name="butterfly")

    train(model=model, optimizer=optimizer, loss_function=loss_function_mse, training_loader=training_loader, epochs_num=epochs_num, device=device, T=T, model_metadata=model_metadata)

    print("done")