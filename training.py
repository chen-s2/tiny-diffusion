from unet import *

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
        noisy_image_transpose = torch.Tensor(noisy_image_transpose).to(device)

        axes[math.floor(noise_index/10.0), noise_index%10].imshow(dt(noisy_image_transpose), cmap='gray')
        axes[math.floor(noise_index/10.0), noise_index%10].axis('off')

    plt.tight_layout()
    plt.show()

def train(model, optimizer, loss_function, training_loader, epochs_num, device, T):
    running_loss = 0.
    last_loss = 0.
    beta = np.linspace(1e-4, 0.02, num=T)

    model.to(device)
    model.train()

    dataset_name = training_loader.dataset_name

    for epoch in range(epochs_num):
        print("epoch:", epoch)
        for i, data in tqdm(enumerate(training_loader)):
            optimizer.zero_grad()
            x0, _ = data

            x0 = x0.to(device)
            epsilon = torch.randn(x0.shape, dtype=x0.dtype, device=device)
            t = int(torch.randint(1, T, (1,), dtype=x0.dtype, device=device))

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

        last_loss = running_loss/len(training_loader)
        print("loss:", last_loss)
        running_loss = 0

    model_name = './models/model_' + dataset_name + "_" + str(round(last_loss,4)) + '.pth'
    torch.save(model, model_name)
    print('saved model to:', model_name)

def loss_function_mse(epsilon, epsilon_pred):
    loss = F.mse_loss(epsilon, epsilon_pred, reduction='none')
    return loss.mean()

