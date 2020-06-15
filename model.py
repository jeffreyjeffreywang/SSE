import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import wandb
from stft import *

EPS = 1e-8 # Avoid numeric errors
        
class Conv1dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, normalize=True):
        super(Conv1dBlock, self).__init__()
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, stride, padding)
        self.normalize = normalize
        if self.normalize:
            self.norm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.normalize:
            x = self.norm(x)
        x = F.softplus(x)
        return x

class ConvTranspose1dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, normalize=True):
        super(ConvTranspose1dBlock, self).__init__()
        self.conv = nn.ConvTranspose1d(input_dim, output_dim, kernel_size, stride, padding)
        self.normalize = normalize
        if self.normalize:
            self.norm = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.normalize:
            x = self.norm(x)
        x = F.softplus(x)
        return x

class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, normalize=True):
        super(Conv2dBlock, self).__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding)
        self.normalize = normalize
        if self.normalize:
            self.norm = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.normalize:
            x = self.norm(x)
        x = F.softplus(x)
        return x

class ConvTranspose2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding, normalize=True):
        super(ConvTranspose2dBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding)
        self.normalize = normalize
        if self.normalize:
            self.norm = nn.BatchNorm2d(output_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.normalize:
            x = self.norm(x)
        x = F.softplus(x)
        return x

class Encoder(nn.Module):
    def __init__(self,config,customize,inter_dims):
        super(Encoder, self).__init__()
        self.dim = config['dim']
        self.mag_power = config['mag_power']
        # STFT parameters and STFT layer
        self.filter_length = config['filter_length']
        self.stft = STFT(filter_length=config['filter_length'],hop_length=config['hop_length'],\
                         win_length=config['win_length'],window=config['window'])
        # Convolution layers
        self.layers = []
        if config['conv_dim'] == 1:
            if customize:
                for i in range(len(inter_dims)):
                    if i == 0:
                        self.layers.append(Conv1dBlock(int(self.filter_length/2)+1,inter_dims[0],\
                                       config['time']['kernel_size'],config['time']['stride'],config['time']['padding']))
                    else:
                        self.layers.append(Conv1dBlock(inter_dims[i-1],inter_dims[i],\
                                       config['time']['kernel_size'],config['time']['stride'],config['time']['padding']))
            else:
                for i in range(config['num_layers']):
                    if i == 0:
                        self.layers.append(Conv1dBlock(int(self.filter_length/2)+1,self.dim,config['time']['kernel_size']\
                                         ,config['time']['stride'],config['time']['padding']))
                    else:
                        self.layers.append(Conv1dBlock(self.dim//(2**(i-1)), self.dim//(2**i),config['time']['kernel_size']\
                                         ,config['time']['stride'],config['time']['padding']))

        self.conv = nn.Sequential(*self.layers)
        self.eq = config['EQ']
        self.all_eq = config['all_EQ']
                                              
    def forward(self,x): 
        a,p = self.stft(x)
        x = a
        # Reduce dynamic range
        x = (x+EPS)**self.mag_power
        # Pass through conv layers
        if self.all_eq:
            for i in range(len(self.layers)):
                x = self.layers[i](x)
                mu = torch.mean(x,2).unsqueeze(2)
                x = x/(mu+EPS)
        else:
            x = self.conv(x)    
        # EQ
        if self.eq:
            mu = torch.mean(x,2).unsqueeze(2)
            x = x/(mu+EPS)
        return x,a,p # latent magnitude, magnitude, phase

class Decoder(nn.Module):
    def __init__(self,config,customize,inter_dims):
        super(Decoder, self).__init__()
        self.dim = config['latent_dim']
        self.mag_power = config['mag_power']
        # STFT parameters
        self.filter_length = config['filter_length']
        self.hop_length = config['hop_length']
        self.istft = ISTFT(num_samples=config['clip_size'],filter_length=config['filter_length'],\
                           hop_length=config['hop_length'],win_length=config['win_length'],window=config['window'])
        
        # Transpose convolution layers
        self.layers = []
        if config['conv_dim'] == 1:
            if customize:
                for i in range(len(inter_dims)-1,-1,-1):
                    if i == 0:
                        self.layers.append(ConvTranspose1dBlock(inter_dims[i],int(self.filter_length/2)+1\
                                         ,config['time']['kernel_size'],config['time']['stride'],config['time']['padding']))
                    else:
                        self.layers.append(ConvTranspose1dBlock(inter_dims[i],inter_dims[i-1]\
                                         ,config['time']['kernel_size'],config['time']['stride'],config['time']['padding']))
            else:
                for i in range(config['num_layers']):
                    if i == config['num_layers']-1:
                        self.layers.append(ConvTranspose1dBlock(self.dim*(2**i),int(self.filter_length/2)+1\
                                         ,config['time']['kernel_size'],config['time']['stride'],config['time']['padding']))
                    else:
                        self.layers.append(ConvTranspose1dBlock(self.dim*(2**i), self.dim*(2**(i+1))\
                                         ,config['time']['kernel_size'],config['time']['stride'],config['time']['padding']))
        self.tconv = nn.Sequential(*self.layers)

    def forward(self,x,p):
        # Pass through transpose conv layers                                            
        a = self.tconv(x)
        # Restore original dynamic range
        a = (a+EPS)**(1.0/self.mag_power)
        x_recon = self.istft(a,p)
        return x_recon, a # time_series, magnitude

class VAE(nn.Module):
    def __init__(self,config,customize=False,inter_dims=None):
        super(VAE, self).__init__()
        self.enc = Encoder(config,customize,inter_dims)
        self.dec = Decoder(config,customize,inter_dims)

    def encode(self,x):
        z,a,p = self.enc(x)
        noise = Variable(torch.randn(z.size())).cuda(z.data.get_device())
        return z, noise, a, p

    def decode(self,z,p):
        x, a = self.dec(z,p)
        return x,a

    def forward(self,x):
        z,a,p = self.enc(x)
        noise = Variable(torch.randn(z.size())).cuda(z.data.get_device())
        x_recon, a = self.dec(z+noise,p)
        return x_recon, z

# Self-supervised speech enhancement network
class SSE(nn.Module):
    def __init__(self,config,gen_a=None):
        super(SSE,self).__init__()
        self.config = config
        if gen_a is not None:
            # Use pretrained CAE
            self.gen_a = gen_a
        else:
            self.gen_a = VAE(config=self.config,customize=config['customize_dim_a'],inter_dims=config['inter_dims_a'])
        self.gen_b = VAE(config=self.config,customize=config['customize_dim_b'],inter_dims=config['inter_dims_b'])

        self.opt_a = torch.optim.Adam([p for p in self.gen_a.parameters() if p.requires_grad], lr=config['lr'], betas=(config['beta1'],config['beta2']), weight_decay=config['weight_decay'])
        self.opt_b = torch.optim.Adam([p for p in self.gen_b.parameters() if p.requires_grad], lr=config['lr'], betas=(config['beta1'],config['beta2']), weight_decay=config['weight_decay'])

    def __compute_kl(self,mu,notnoise=None):
        if notnoise is not None:
            # Don't compute KL losses for pure noises
            return torch.sum(notnoise.unsqueeze(1).unsqueeze(2)*torch.pow(mu,2)) / \
                                    (torch.sum(notnoise)*mu.shape[1]*mu.shape[2]+EPS)
        return torch.mean(torch.pow(mu,2))

    def recon_criterion(self,x_recon,x=None,notnoise=None,cross=False):
        if cross:
            # Pure noise losses: enforce the output of pure noises to be zeros
            return torch.sum(notnoise.unsqueeze(1).unsqueeze(2)*(x_recon**2))\
                    /(torch.sum(notnoise)*x_recon.shape[1]*x_recon.shape[2]+EPS)
    
        if self.config['recon_loss_type'] == 'L2':
            if notnoise is not None:
                # Don't compute reonstruction losses for pure noises
                return torch.sum(notnoise.unsqueeze(1).unsqueeze(2)*\
                       ((x_recon**self.config['mag_power']-x**self.config['mag_power'])**2))\
                        /(torch.sum(notnoise)*x_recon.shape[1]*x_recon.shape[2]+EPS)
            return torch.mean((x_recon**self.config['mag_power']-x**self.config['mag_power'])**2)
    
    # Train the CAE
    def gen_a_update(self,x_a):
        self.opt_a.zero_grad()
        # Encode
        h_a,n_a,mag_a,p_a = self.gen_a.encode(x_a)
        # Decode
        x_a_recon,mag_a_recon = self.gen_a.decode(h_a+n_a,p_a)
        # Loss
        self.recon_loss = self.config['recon_a_w']*self.recon_criterion(mag_a_recon,mag_a)
        self.kl_loss = self.config['kl_a_w']*self.__compute_kl(h_a)

        self.total_loss = self.recon_loss + self.kl_loss
        self.total_loss.backward()
        self.opt_a.step()
        return self.total_loss, x_a_recon
    
    # Train the MAE
    def gen_b_update(self,x_b,notnoise):
        self.opt_b.zero_grad()
        # Encode
        h_b,n_b,mag_b,p_b = self.gen_b.encode(x_b)
        # Decode (within domain)
        x_b_recon, mag_b_recon = self.gen_b.decode(h_b+n_b,p_b)
        # Decode (cross domain)
        x_ba, mag_ba = self.gen_a.decode(h_b+n_b,p_b)
        # Encode again
        h_b_recon,n_b_recon,mag_b_recon,p_b_recon = self.gen_a.encode(x_ba)
        # Decode again
        x_bab, mag_bab = self.gen_b.decode(h_b_recon+n_b_recon,p_b_recon)
    
        # Cross domain reconstruction loss for pure noises
        if torch.sum(1-notnoise) != 0:
            self.loss_cross = self.config['cross_w']*self.recon_criterion(mag_ba,notnoise=1-notnoise,cross=True)
        else:
            self.loss_cross = 0
        # Losses
        self.loss_recon_b = self.config['recon_b_w']*self.recon_criterion(mag_b_recon,mag_b)
        self.loss_kl_b = self.config['kl_b_w']*self.__compute_kl(h_b)
        self.loss_recon_cyc_b = self.config['cyc_recon_b_w']*self.recon_criterion(mag_bab,mag_b,notnoise=notnoise)
        self.loss_kl_cyc_bab = self.config['cyc_kl_b_w']*self.__compute_kl(h_b_recon,notnoise=notnoise)
        self.loss_recon_latent_b = self.config['recon_latent_b_w']*self.recon_criterion(h_b_recon,h_b,notnoise=notnoise)
        # Log Losses
        wandb.log({'recon_loss': self.loss_recon_b})
        wandb.log({'recon_cyc_loss': self.loss_recon_cyc_b})
        wandb.log({'latent_loss': self.loss_recon_latent_b})
        wandb.log({'kl_loss': self.loss_kl_b+self.loss_kl_cyc_bab})
        wandb.log({'cross_loss': self.loss_cross})
        # Total loss
        self.loss = self.loss_recon_b + self.loss_kl_b + self.loss_recon_cyc_b + self.loss_kl_cyc_bab \
                                      + self.loss_recon_latent_b + self.loss_cross
        self.loss.backward()
        self.opt_b.step()

        return self.loss,mag_b,mag_b_recon,mag_ba,x_b_recon,x_ba

    def forward(self,x,option,notnoise=None):
        if option == 'a':
            return self.gen_a_update(x)
        elif option == 'b':
            return self.gen_b_update(x,notnoise)
        elif option == 'eval': 
            return self.evaluate(x)
    
    def evaluate(self,x_b):
        self.eval()
        # Encode
        h_b,n_b,mag_b,p_b = self.gen_b.encode(x_b)
        # Decode (within domain)
        x_b_recon, mag_b_recon = self.gen_b.decode(h_b+n_b,p_b)
        # Decode (cross domain)
        x_ba, mag_ba = self.gen_a.decode(h_b+n_b,p_b)
        # Encode again
        h_b_recon,n_b_recon,mag_b_recon,p_b_recon = self.gen_a.encode(x_ba)
        # Decode again
        x_bab, mag_bab = self.gen_b.decode(h_b_recon+n_b_recon,p_b_recon)
        self.train()
        return x_b_recon,mag_ba,x_ba,mag_b