import torch
import torch.nn as nn
import torchvision.models as models
from functionalities import loss

class MMD_autoencoder_loss(nn.Module):
    def __init__(self, a_distr, a_rec, a_spar, a_disen=0, a_disc=0, latent_dim=8, loss_type='l1', device='cpu',
                 disc_lst=None, feat_idx=None):
        super(MMD_autoencoder_loss, self).__init__()
        self.a_distr = a_distr
        self.a_rec = a_rec
        self.a_spar = a_spar
        self.a_disen = a_disen
        self.a_disc = a_disc
        self.disc_lst = disc_lst
        self.latent_dim = latent_dim
        self.loss_type = loss_type
        self.device = device

        if self.loss_type == 'vgg':
            if feat_idx is None:
                feat_idx = 5
            vgg16 = models.vgg16(pretrained=True)
            self.feat_model = nn.Sequential(*list(vgg16.features[:feat_idx]))
            self.feat_model.eval()
            self.feat_model.to(self.device)
        elif self.loss_type == 'resnet':
            if feat_idx is None:
                feat_idx = 4
            resnet18 = models.resnet18(pretrained=True)
            self.feat_model = nn.Sequential(*list(resnet18.children())[:feat_idx])
            self.feat_model.eval()
            self.feat_model.to(self.device)

    def forward(self, z, v, z_, target=None):
        if self.loss_type == 'l1':
            l_rec = self.a_rec * loss.l1_loss(z_, z)
        elif self.loss_type == 'l2':
            l_rec = self.a_rec * loss.l2_loss(z_, z)
        elif self.loss_type == 'vgg' or self.loss_type == 'resnet':
            l_rec = self.a_rec * loss.feat_loss(z_, z, self.feat_model)
        else:
            print('loss not found')

        l_sparse = self.a_spar * torch.mean(v[:, self.latent_dim:] ** 2)

        y = v.new_empty((v.size(0), self.latent_dim)).normal_()

        l_distr = self.a_distr * loss.MMD_gram(v[:, :self.latent_dim], y)

        l_disen = self.a_disen * loss.MMD_gram(v[:, :self.latent_dim], loss.shuffle(v[:, :self.latent_dim]))


        if target is not None and self.disc_lst is not None:
            l_disc = self.a_disc * loss.l2_loss(v[:, :1], self.disc_lst[target])
            l = l_rec + l_distr + l_sparse + l_disen + l_disc
            return [l, l_rec, l_distr, l_sparse, l_disen, l_disc]
        elif self.disc_lst is not None:
            l_disc = self.a_disc * loss.l2_loss(torch.min(torch.abs(v[:, :1] - self.disc_lst), 1)[0])
            l = l_rec + l_distr + l_sparse + l_disen + l_disc
            return [l, l_rec, l_distr, l_sparse, l_disen, l_disc]
        else:
            l = l_rec + l_distr + l_sparse + l_disen
            return [l, l_rec, l_distr, l_sparse, l_disen]