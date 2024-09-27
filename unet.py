from torch import nn
import torch
from time_embedding import TimeEmbedding
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention
from dataset import create_dataloader

class UNetOutputLayer(nn.Module):

    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_chan)
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=3, padding=1)

    def forward(self, x):
        '''
        x: (B,320,H/8,W/8), or (B,in_chan,H/8,W/8)
        '''
        # (B,320,H/8,W/8) -> (B,320,H/8,W/8)
        x = self.groupnorm(x)

        # (B,320,H/8,W/8) -> (B,320,H/8,W/8)
        x = F.silu(x)

        # (B,320,H/8,W/8) -> (B,4,H/8,W/8)
        x = self.conv(x)

        # (B,4,H/8,W/8)
        return x

class UNetAttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

    def forward(self, x, text_context):
        '''
        :param x: B,features_len,h,w
        :param text_context: B,seq_len,embed_dim
        '''
        residue_long = x

        x = self.attention_1(x)
        x = self.attention_2(x, text_context)

        return self.conv_output(x) + residue_long

class UNetResidualBlock(nn.Module):
    def __init__(self, in_chan, out_chan, n_time=1280):
        super().__init__()
        self.linear_time = nn.Linear(n_time, out_chan)
        self.conv_merged = nn.Conv2d(out_chan, out_chan, kernel_size=3, padding=1)

        if in_chan == out_chan:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_chan, out_chan, kernel_size=1, padding=0)
    def forward(self, features, time_embed):
        '''
        :param features: B,in_chan,h,w
        :param time_embed: 1,1280
        '''

        residue = features

        time_embed = self.linear_time(time_embed)

        print("shapes:", features.shape, time_embed.unsqueeze(-1).unsqueeze(-1).shape)
        merged = features + time_embed.unsqueeze(-1).unsqueeze(-1)

        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)

class SwitchSequential(nn.Sequential):
    '''
    we pass this class 3 types of inputs in a single forward call,
    and this class contains multiple layers, each layer is using a different subset of these inputs.
    '''
    def forward(self, x, text_context, time_embed):
        for layer in self:
            if isinstance(layer, UNetAttentionBlock):
                x = layer(x, text_context)
            elif isinstance(layer, UNetResidualBlock):
                x = layer(x, time_embed)
            else:
                print("x.shape:", x.shape)
                x = layer(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
                                       SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
                                       SwitchSequential(UNetResidualBlock(320,320), UNetAttentionBlock(8,40)),
                                       SwitchSequential(nn.Conv2d(320,320,kernel_size=3, stride=2, padding=1)),
                                       SwitchSequential(UNetResidualBlock(320,1280))
                                       ])

        self.bottleneck = SwitchSequential(UNetResidualBlock(1280,1280),
                                           UNetAttentionBlock(8,160),
                                           UNetResidualBlock(1280,1280))

        self.decoders = nn.ModuleList([
            SwitchSequential(UNetResidualBlock(2560,1280)),
            SwitchSequential(UNetResidualBlock(1280,320), UNetAttentionBlock(8,40))
        ])

    def forward(self, image_latent, text_context, time_embed):
        '''
        :param image_latent: (B,4,h/8,w/8)
        :param text_context: (B,seq_len,embed_dim)
        :param time_embed: (1,1280)
        '''

        skip_connections = []
        x = image_latent
        for layers in self.encoders:
            x = layers(x, text_context, time_embed)
            skip_connections.append(x)

        x = self.bottleneck(x, text_context, time_embed)

        for layers in self.decoders:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layers(x, text_context, time_embed)

class UNetDiffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNet()
        self.unet_output = UNetOutputLayer(in_chan=320, out_chan=4)

    def forward(self, image_latent, text_context, time_embed):
        '''
        image_latent: (B, 4, H/8, W/8), the latent representation of the image, after the VAE-encoder, plus added noise
        text_context: (B, seq_len, embed_len), the prompt after passing CLIP encoding, seq_len is the max len of clip's output
        time_embed: (1, 320), the current timestep's in some (?) space
        '''

        # (1,320) -> (1,1280)
        time_embed = self.time_embedding(time_embed)

        # (B, 4, H/8, W/8) -> (B, 320, H/8, W/8)
        output = self.unet(image_latent, text_context, time_embed)

        # (B, 320, H/8, W/8) -> (B, 4, H/8, W/8)
        output = self.unet_output(output)

        return output

def train(model, optimizer, loss_function, training_loader, epochs_num, device):
    running_loss = 0.
    last_loss = 0.

    model.to(device)
    model.train()

    for epoch in range(epochs_num):
        for i, data in enumerate(training_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            '''
            image_latent: (B, 4, H/8, W/8), the latent representation of the image, after the VAE-encoder, plus added noise
            text_context: (B, seq_len, embed_len), the prompt after passing CLIP encoding, seq_len is the max len of clip's output
            time_embed: (1, 320), the current timestep's in some (?) space
            '''

            seq_len, embed_len = 80, 512
            text_context = torch.ones((batch_size, seq_len, embed_len)).to(device)
            time_embed = torch.ones((1,320)).to(device)

            outputs = model(image_latent=inputs, text_context=text_context, time_embed=time_embed)

            loss = loss_function(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('batch {} loss: {}'.format(i + 1, last_loss))
                running_loss = 0.

            return None

    return last_loss

def loss_function_mse(model_out, target):
    loss = F.mse_loss(model_out, target, reduction='none')
    return loss.mean()

data_path = "../data/train"
image_size = 256
batch_size = 16
epochs_num = 1
device = 'cuda'

model = UNetDiffusion()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

training_loader = create_dataloader(data_path, image_size, batch_size)

train(model=model, optimizer=optimizer, loss_function=loss_function_mse, training_loader=training_loader, epochs_num=epochs_num, device=device)

print("done")