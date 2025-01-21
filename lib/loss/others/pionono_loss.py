import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.utils.tools.logger import Logger as Log
import copy
import logging
import math
from os.path import join as pjoin
import numpy as np
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
import torch.distributions as td
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import argparse
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from abc import ABCMeta, abstractmethod
from mmcv.cnn import ConvModule
import pdb

from lib.models.backbones.backbone_selector import BackboneSelector
from lib.models.modules.decoder_block import DeepLabHead
from lib.models.modules.projection import ProjectionHead
from lib.models.nets.unet import UNet
from lib.models.modules.prototype.proto_module_max import PrototypeModule_Max_V1
from lib.models.modules.prototype.proto_module_se import PrototypeModule_SE_V1
from lib.models.tools.module_helper import ModuleHelper
from lib.models.modules.decoder_block import ASPPModule
from lib.loss.loss_utils import DiceLoss, DiceCELoss


class PiononoLoss(nn.Module):
    def __init__(self, configer, input_channels=3, num_classes=2, annotators=2, gold_annotators=[0], latent_dim=8,
                 z_prior_mu=0.0, z_prior_sigma=2.0, z_posterior_init_sigma=8.0, no_head_layers=3, head_kernelsize=1,
                 head_dilation=1, kl_factor=0.0005, reg_factor=0.00001, mc_samples=5):
        super(PiononoLoss, self).__init__()
        self.configer = configer
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.annotators = annotators
        self.gold_annotators = gold_annotators
        self.no_head_layers = no_head_layers
        self.head_kernelsize = head_kernelsize
        self.head_dilation = head_dilation
        self.kl_factor = kl_factor
        self.reg_factor = reg_factor
        self.train_mc_samples = mc_samples
        self.test_mc_samples = 20
        self.z = LatentVariable(2, latent_dim, prior_mu_value=z_prior_mu, prior_sigma_value=z_prior_sigma,
                                z_posterior_init_sigma=z_posterior_init_sigma)
        self.head = PiononoHead(16, self.latent_dim, self.input_channels, self.num_classes,
                                self.no_head_layers, self.head_kernelsize, self.head_dilation, use_tile=True)
        self.phase = 'segmentation'
        self.name = 'PiononoModel'

        self.ann_ids = torch.zeros(2, dtype=torch.float32)
        self.loss_fct = DiceCELoss(self.configer)

    def elbo(self, inputs, labels, loss_fct, annotator: torch.tensor):
        """
        Calculate the evidence lower bound of the log-likelihood of P(Y|X)
        """
        # self.preds = self.sample(use_z_mean=False, annotator=annotator)
        self.log_likelihood_loss = loss_fct(inputs, labels)
        self.kl_loss = self.z.get_kl_loss(annotator) * self.kl_factor

        return -(self.log_likelihood_loss + self.kl_loss)

    def combined_loss(self, inputs, labels, loss_fct, annotator):
        """
        Combine ELBO with regularization of deep network weights.
        """
        elbo = self.elbo(inputs, labels, loss_fct=loss_fct, annotator=annotator)
        self.reg_loss = l2_regularisation(self.head.layers) * self.reg_factor
        loss = -elbo + self.reg_loss
        return loss

    def forward(self, inputs, target):
        loss = self.combined_loss(inputs, target, self.loss_fct, self.ann_ids)
        # loss = self.loss_fct(inputs, target)
        return loss


class LatentVariable(nn.Module):
    """
    This module defines the random latent variable z with distribution q(z|r) with r being the rater.
    """

    def __init__(self, num_annotators, latent_dims=2, prior_mu_value=0.0, prior_sigma_value=1.0, z_posterior_init_sigma=0.0):
        super(LatentVariable, self).__init__()
        self.latent_dims = latent_dims
        self.no_annotators = num_annotators
        prior_mu, prior_cov = self._init_distributions(prior_mu=prior_mu_value, prior_sigma=prior_sigma_value)
        self.prior_mu = torch.nn.Parameter(prior_mu)
        self.prior_covtril = torch.nn.Parameter(prior_cov)
        self.prior_mu.requires_grad = False
        self.prior_covtril.requires_grad = False
        post_mu_value = np.random.standard_normal(size=[num_annotators, latent_dims])*z_posterior_init_sigma + prior_mu_value
        post_sigma_value = prior_sigma_value
        posterior_mu, posterior_cov = self._init_distributions(prior_mu=post_mu_value, prior_sigma=post_sigma_value)
        self.posterior_mu = torch.nn.Parameter(posterior_mu)
        self.posterior_covtril = torch.nn.Parameter(posterior_cov)
        self.name = 'LatentVariable'

    def _init_distributions(self, prior_mu=0.0, prior_sigma=1.0):
        mu_list = []
        cov_list = []
        prior_mu = np.array(prior_mu)
        prior_sigma = np.array(prior_sigma)
        for a in range(self.no_annotators):
            if prior_mu.size > 1:
                mu = prior_mu[a]
            else:
                mu = prior_mu
            if prior_sigma.size > 1:
                sigma = prior_sigma[a]
            else:
                sigma = prior_sigma
            mu_a = np.ones(self.latent_dims)*mu
            # we use sigma values (instead of sigma+sigma) because we pass this matrix as tril matrix L
            # this makes the sigma value of the cov matrix squared
            cov_a = np.eye(self.latent_dims) * (sigma)
            mu_list.append(mu_a)
            cov_list.append(cov_a)
        mu_list = torch.tensor(np.array(mu_list))
        covtril_list = torch.tensor(np.array(cov_list))
        return mu_list, covtril_list

    def forward(self, annotator, sample=True):
        z = torch.zeros([len(annotator), self.latent_dims])
        annotator = annotator.long()
        for i in range(len(annotator)):
            a = annotator[i]
            dist_a = torch.distributions.multivariate_normal.MultivariateNormal(self.posterior_mu[a],
                                                                                scale_tril=torch.tril(self.posterior_covtril[a]))

            if sample:
                z_i = dist_a.rsample()
            else:
                z_i = dist_a.loc
            z[i] = z_i
        return z

    def get_kl_loss(self, annotator):
        kl_loss = torch.zeros([len(annotator)])
        annotator = annotator.long()
        for i in range(len(annotator)):
            a = annotator[i]
            dist_a_posterior = torch.distributions.multivariate_normal.MultivariateNormal(self.posterior_mu[a],
                                                                                          scale_tril=torch.tril(self.posterior_covtril[a]))
            dist_a_prior = torch.distributions.multivariate_normal.MultivariateNormal(self.prior_mu[a],
                                                                                      scale_tril=torch.tril(self.prior_covtril[a]))

            kl_loss[i] = torch.distributions.kl_divergence(dist_a_posterior, dist_a_prior)
        kl_mean = torch.mean(kl_loss)
        return kl_mean


class PiononoHead(nn.Module):
    """
    The Segmentation head combines the sample taken from the latent space,
    and feature map by concatenating them along their channel axis.
    """

    def __init__(self, num_filters_last_layer, latent_dim, num_output_channels, num_classes, no_convs_fcomb,
                 head_kernelsize, head_dilation, use_tile=True):
        super(PiononoHead, self).__init__()
        self.num_channels = num_output_channels  # output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2, 3]
        self.num_filters_last_layer = num_filters_last_layer
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb
        self.head_kernelsize = head_kernelsize
        self.name = 'PiononoHead'

        if self.use_tile:
            layers = []

            # Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(self.num_filters_last_layer+self.latent_dim, self.num_filters_last_layer,
                                    kernel_size=self.head_kernelsize, dilation=head_dilation, padding='same'))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb-2):
                layers.append(nn.Conv2d(self.num_filters_last_layer, self.num_filters_last_layer,
                                        kernel_size=self.head_kernelsize, dilation=head_dilation, padding='same'))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)
            self.last_layer = nn.Conv2d(self.num_filters_last_layer, self.num_classes, kernel_size=self.head_kernelsize,
                                        dilation=head_dilation, padding='same')
            self.activation = torch.nn.Softmax(dim=1)

            self.layers.apply(self.initialize_weights)
            self.last_layer.apply(self.initialize_weights)

    def initialize_weights(self, module):
        for m in module.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(
            np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z, use_softmax=True):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z, 2)
            z = self.tile(z, 2, feature_map.shape[self.spatial_axes[0]])
            z = torch.unsqueeze(z, 3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

            # Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            x = self.layers(feature_map)
            y = self.last_layer(x)
            if use_softmax:
                y = self.activation(y)
            return y


def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        # nn.init.normal_(m.weight, std=0.001)
        # nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)


def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        # nn.init.normal_(m.bias, std=0.001)


def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg
