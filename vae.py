import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from typing import List
import wandb

def sdf(y_lab, y_pred):
    '''signed distance function. idk how to implement this, so I'm just going to
    subtract for now, which I think is actually accurate for one-dimensional y'''
    return y_lab - y_pred

def normalize(data):
    # (x-u)/s
    return torch.nan_to_num(
        (data - torch.mean(data, dim=2).unsqueeze(2))/torch.std(data, dim=2, correction=1).unsqueeze(2),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

def loss_amp_shift(y_lab, y_pred, rand_weight=1):
    T_prime = y_lab.shape[-1]
    per_el_loss = T_prime * torch.sum(
        torch.abs(1 / T_prime - F.softmax(sdf(y_lab, y_pred), dim=-1)), dim=(-2, -1))
    return torch.mean(per_el_loss * rand_weight)

def loss_phase(y_lab, y_pred, rand_weight=1):
    fft_lab = torch.fft.rfft(y_lab).abs()
    fft_pred = torch.fft.rfft(y_pred).abs()
    
    # select only for dominant frequencies in the true FFT
#     width = 39
#     win_max = torch.nn.functional.max_pool1d(fft_lab, kernel_size=width, stride=9, padding=width//2, return_indices=True, ceil_mode=True)[1]
#     mask = win_max != win_max.roll(1, dims=-1)
#     temp_eeg = torch.zeros_like(fft_lab)
#     for i in range(len(mask)):
#         for j in range(len(mask[i])):
#             temp_eeg[i, j, win_max[i][j][mask[i][j]]] = 1
#     new_eeg = torch.where(temp_eeg == 1, fft_lab, 0)
    
    eps = 1e-7
    norm_lab  = fft_lab  / (torch.topk(fft_lab,  k=5, dim=2).values[:,:,4].unsqueeze(2) + eps)
    norm_pred = fft_pred / (torch.topk(fft_pred, k=5, dim=2).values[:,:,4].unsqueeze(2) + eps)
    weighted_diff = norm_lab - norm_pred
    
    per_el_loss = torch.linalg.matrix_norm(weighted_diff)
    return torch.mean(per_el_loss * rand_weight)

def loss_amp(y_lab, y_pred, rand_weight=1):
    if len(y_lab.shape) != 3:
        raise NotImplemented()
#     auto_corr = torch.nan_to_num(F.conv1d(y_lab, y_lab).div(y_lab.std(dim=-1)), nan=0.0, posinf=0.0, neginf=0.0)
#     cross_corr = torch.nan_to_num(F.conv1d(y_lab, y_pred).div(y_pred.std(dim=-1)), nan=0.0, posinf=0.0, neginf=0.0)
    
    norm_lab = normalize(y_lab)
    # the zero bias is intended to eliminate a bug where the loss is occasionally zero
    auto_corr = F.conv1d(norm_lab, norm_lab, bias=torch.zeros(y_lab.shape[0]))
    cross_corr = F.conv1d(norm_lab, normalize(y_pred), bias=torch.zeros(y_lab.shape[0]))

    if torch.norm(auto_corr - cross_corr).isnan().any():
        print(f'Caught a NaN in amp loss')
        print(y_lab)
        print(norm_lab)
        print(y_pred)
        print(normalize(y_pred))
        print(auto_corr.isnan().sum())
        print(cross_corr.isnan().sum())
        raise ValueError()
    
    # formula taken from https://xcdskd.readthedocs.io/en/latest/cross_correlation/cross_correlation_coefficient.html
#     auto_corr = torch.sum(normalize(y_lab)*normalize(y_lab)) / (y_lab.reshape(-1).size()[0]-1)
#     cross_corr = torch.sum(normalize(y_lab)*normalize(y_pred)) / (y_lab.reshape(-1).size()[0]-1)
    
    # for generic TILDE-Q, we would just return the mean of per-element loss
    per_el_loss = torch.linalg.matrix_norm(auto_corr - cross_corr)
    return torch.mean(per_el_loss * rand_weight)

def loss_tilde(y_lab, y_pred, alpha=0.04, gamma=0.03):
    # WARNING: does not have rand_weight, not meant to be used here. For reference only
    assert alpha > 0 and alpha < 1
    return alpha*loss_amp_shift(y_lab, y_pred) \
    + (1-alpha)*loss_phase(y_lab, y_pred) \
    + gamma * loss_amp(y_lab, y_pred)


class VanillaVAE(torch.nn.Module):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 input_len: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.batch_num = 0
        
        in_channels_init = in_channels

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        self.last_hidden_dim = hidden_dims[-1]
        self.end_conv_len = input_len // (2 ** len(hidden_dims))

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*self.end_conv_len, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*self.end_conv_len, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.end_conv_len)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose1d(2*hidden_dims[i],
                                       2*hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm1d(2*hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose1d(2*hidden_dims[-1],
                                               2*hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm1d(2*hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv1d(2*hidden_dims[-1], out_channels= in_channels_init,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
        
    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor, shuffled_inp: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        # 4 comes from taking 128 initial samples and halving once for each conv layer (stride=2, so half)
        # -Adel
        result = result.view(-1, self.last_hidden_dim, self.end_conv_len)
        # add shuffled input as another set of channels
        inp_enc = self.encoder(shuffled_inp)
        result = torch.cat((result, inp_enc), dim=1)
        
        result = self.decoder(result)
        # multiplying by 2 here because with 10-90% RobustScaler,
        # some values will be as high as 2 or sometimes even 3 after
        # standardization
        result = 2*self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, rand_prob: float, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        
        shuffled_inp = input.clone()
        self.batch_num += 1
        # shuffle some of the values along the time axis
        for i in range(64):
            perm_idx = torch.randperm(1024)
            mixed_idx = torch.where(torch.rand(1024) < rand_prob[i], perm_idx, torch.arange(1024))
            shuffled_inp[i] = shuffled_inp[i,:,mixed_idx]
        # add baseline noise
        rand_resized = rand_prob.repeat(input.shape[2], input.shape[1], 1)
        rand_resized = rand_resized.permute(2, 1, 0)
        shuffled_inp = torch.normal(shuffled_inp, rand_resized * 0.01)
        # randomly erase some of the amplitudes
        shuffled_inp = torch.where(torch.rand(shuffled_inp.size()) < rand_resized, torch.tensor(0).float(), shuffled_inp)
        return  [self.decode(z, shuffled_inp), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        
        y_lab = input
        y_pred = recons
        amp_shift_loss = wandb.config.alpha*loss_amp_shift(y_lab, y_pred, rand_weight=2*kwargs['rand_prob'])
        phase_loss = (1-wandb.config.alpha)*loss_phase(y_lab, y_pred, rand_weight=2*kwargs['rand_prob'])
        amp_loss = wandb.config.gamma * loss_amp(y_lab, y_pred, rand_weight=2*kwargs['rand_prob'])
        
        # B*M/N, where 0<=B<=1, M = latent size, N = batch size
        kld_weight =  wandb.config.kld_weight_beta*self.latent_dim/kwargs['batch_size'] # kwargs['M_N']
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        
        per_el_loss = torch.mean(F.mse_loss(y_pred, y_lab, reduction='none'), dim=(1,2)) * kwargs['rand_prob']
        loss = torch.mean(per_el_loss) # amp_shift_loss + phase_loss + amp_loss
        
        return {
            'loss': loss + kld_weight * kld_loss,
            'loss_amp_shift': amp_shift_loss,
            'loss_phase': phase_loss,
            'loss_amp': amp_loss,
            'loss_tilde': loss,
            'KLD_Loss': kld_weight * kld_loss,
        }

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x, 1)[0]