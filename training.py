from unet import *
from tqdm import tqdm

def show_diffusion_chain(clean_image, epsilon, T):
    num_noise_levels = 20
    beta = np.linspace(1e-4, 0.02, num=T)
    timesteps = np.linspace(T-1, 1, num=num_noise_levels).astype('int')
    device = clean_image.device
    print("image shape:", clean_image.shape, ", epsilon shape:", epsilon.shape)

    fig, axes = plt.subplots(2, num_noise_levels//2, figsize=(16, 3))

    for noise_index, t in enumerate(timesteps):
        alpha_1_to_t_array = []
        for i in range(1, t + 1):
            alpha_t = 1 - beta[i]
            alpha_1_to_t_array.append(alpha_t)
        alpha_t_bar = torch.prod(torch.Tensor(alpha_1_to_t_array))

        noisy_image = torch.sqrt(alpha_t_bar) * clean_image + torch.sqrt(1 - alpha_t_bar) * epsilon

        noisy_image_transpose = np.transpose(dt(noisy_image), (1, 2, 0))

        min_img, max_img = np.min(noisy_image_transpose), np.max(noisy_image_transpose)
        noisy_image_transpose = 255.0 * ((noisy_image_transpose - min_img) / (max_img - min_img))
        noisy_image_transpose = noisy_image_transpose.astype('uint8')

        axes[math.floor(noise_index/10.0), noise_index%10].imshow(noisy_image_transpose) #, cmap='viridis')
        axes[math.floor(noise_index/10.0), noise_index%10].axis('off')

    plt.tight_layout()
    plt.show()

def train(model, optimizer, loss_function, training_loader, epochs_num, device, T, model_metadata):
    running_loss = 0.
    clean_t_running_loss = 0 # running loss for the 5% smallest values of t: [0,0.05*T]
    last_loss = np.inf
    clean_t_last_loss = np.inf
    clean_t_samples = 0

    beta = np.linspace(1e-4, 0.02, num=T)

    model.to(device)
    model.train()

    dataset_name = training_loader.dataset_name

    for epoch in range(epochs_num):
        print("epoch:", epoch)
        ts = []
        for i, data in tqdm(enumerate(training_loader)):
            optimizer.zero_grad()
            x0, _ = data

            x0 = x0.to(device)
            epsilon = torch.randn(x0.shape, dtype=x0.dtype, device=device)
            t = int(torch.randint(1, T, (1,), dtype=x0.dtype, device=device))
            ts.append(t)

            alpha_1_to_t_array = []
            for i in range(1,t+1):
                alpha_t = 1-beta[i]
                alpha_1_to_t_array.append(alpha_t)
            alpha_t_bar = torch.prod(torch.Tensor(alpha_1_to_t_array))

            noisy_image = torch.sqrt(alpha_t_bar) * x0 + torch.sqrt(1-alpha_t_bar) * epsilon

            # show_diffusion_chain(clean_image=x0[0], epsilon=epsilon[0], T=T)
            # quit()

            epsilon_pred = model(image=noisy_image, t=t)

            loss = loss_function(epsilon, epsilon_pred)  # todo: why don't we minimize the diff between epsilon_pred and sqrt(1-alpha_t)*epsilon?

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if t <= int(0.05*T):
                clean_t_running_loss += loss.item()
                clean_t_samples += 1

        last_loss = running_loss/len(training_loader)
        print("loss:", last_loss)
        print("clean_t loss:", clean_t_last_loss)
        running_loss = 0

        if epoch%10==0 and epoch>0:
            model_name = './models/model_' + dataset_name + "_" + model_metadata + "_" + str(round(last_loss,4)) + "_cleanloss_" + str(round(clean_t_last_loss,4)) + '.pth'
            torch.save(model, model_name)
            print('saved model to:', model_name)

        if clean_t_samples > 20:
            clean_t_last_loss = clean_t_running_loss / clean_t_samples
            clean_t_running_loss = 0
            clean_t_samples = 0

    model_name = './models/model_' + dataset_name + "_" + model_metadata + "_" + str(round(last_loss, 4)) + "_cleanloss_" + str(round(clean_t_last_loss, 4)) + '.pth'
    torch.save(model, model_name)
    print('saved model to:', model_name)

def loss_function_mse(epsilon, epsilon_pred):
    loss = F.mse_loss(epsilon, epsilon_pred, reduction='none')
    return loss.mean()

