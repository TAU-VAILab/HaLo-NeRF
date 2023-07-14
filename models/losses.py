import kornia.losses
import torch
from torch import nn
import math
import numpy as np



class ColorLoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return self.coef * loss

class CosineAnnealingWeight():
    def __init__(self, max, min, Tmax):
        super().__init__()
        # 5e-2
        self.max = max
        self.min = min
        self.Tmax = Tmax

    def getWeight(self, Tcur):
        return self.min + (self.max - self.min) * (1 + math.cos(math.pi * Tcur / self.Tmax)) / 2

class ExponentialAnnealingWeight():
    def __init__(self, max, min, k):
        super().__init__()
        # 5e-2
        self.max = max
        self.min = min
        self.k = k

    def getWeight(self, Tcur):
        return max(self.min, self.max * math.exp(-Tcur*self.k))

class HaNeRFLoss(nn.Module):
    def __init__(self, hparams, coef=1, lambda_u=0.01):
        super().__init__()
        self.coef = coef
        self.lambda_u = lambda_u
        # self.Annealing = CosineAnnealingWeight(max = hparams.maskrs_max, min = hparams.maskrs_min, Tmax = hparams.num_epochs-1)
        self.Annealing = ExponentialAnnealingWeight(max = hparams.maskrs_max, min = hparams.maskrs_min, k = hparams.maskrs_k)
        self.BCE_loss = nn.BCELoss()

    def forward(self, inputs, targets, semantics_gt, hparams, global_step):
        ret = {}

        if 'a_embedded' in inputs:
            ret['kl_a'] = self._l2_regularize(inputs['a_embedded']) * hparams.weightKL
            if 'a_embedded_random_rec' in inputs:
                ret['rec_a_random'] = torch.mean(torch.abs(inputs['a_embedded_random'].detach() - inputs['a_embedded_random_rec'])) * hparams.weightRecA
                ret['mode_seeking'] = hparams.weightMS * 1 / \
                  ((torch.mean(torch.abs(inputs['rgb_fine'].detach() - inputs['rgb_fine_random'])) / \
                  torch.mean(torch.abs(inputs['a_embedded'].detach() - inputs['a_embedded_random'].detach() + 1e-6))) + 1 * 1e-5)

        if 'out_mask' in inputs:
            mask = inputs['out_mask']
            ret['c_l'] = 0.5 * ((1 - mask.detach()) * (inputs['rgb_coarse'] - targets)**2).mean()
        else:
            ret['c_l'] = 0.5 * ((inputs['rgb_coarse'] - targets)**2).mean()

        if 'rgb_fine' in inputs:
            if 'out_mask' in inputs:
                ret['r_ms'], ret['r_md'] = self.mask_regularize(inputs['out_mask'],  self.Annealing.getWeight(global_step), hparams.maskrd)
                ret['f_l'] = 0.5 * ((1 - mask) * (inputs['rgb_fine'] - targets)**2).mean()
            else:
                ret['f_l'] = 0.5 * ((inputs['rgb_fine']-targets)**2).mean()

        if 'semantics_fine' in inputs:
            p = inputs['semantics_fine'][:, 1]
            gt = semantics_gt.cuda()
            # ret['semantics_fine'] = 0.5*(self.BCE_loss(p, semantics_gt.cuda()) + torch.mean(p*(1-p)))
            # ret['semantics_fine'] = -torch.mean(gt * (gt*torch.log(p)+(1-gt)*torch.log(1-p)))
            # ret['semantics_fine'] = self.BCE_loss(p, semantics_gt.cuda()) #+ 0.05*(1/(p+0.1))

            # bce = -(0.67*gt*torch.log(p)+0.33*(1-gt)*torch.log(1-p))
            bce = -(gt * torch.log(p) + (1 - gt) * torch.log(1 - p))
            # bce = self.BCE_loss(p, semantics_gt.cuda())

            max_entropy_loss = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
            max_entropy_loss = torch.mean(max_entropy_loss)
            pt = torch.exp(-bce)
            self.gamma = 0
            #
            # tv_loss = kornia.losses.TotalVariation()
            # l = len(p)
            # p_reshaped = p.reshape(1, 1, hparams['H'], hparams['W'])
            # res_tv_loss = tv_loss(p_reshaped) / l

            # ret['semantics_fine'] = 0.5 * torch.mean(((1-pt + 1e-5)**self.gamma) * bce) + 0.5 * max_entropy_loss #1* res_tv_loss # +
            ret['semantics_fine'] = torch.mean(((1 - pt + 1e-5) ** self.gamma) * bce)

        if 'semantics_coarse' in inputs:
            p = inputs['semantics_coarse'][:, 1]
            gt = semantics_gt.cuda()
            # ret['semantics_coarse'] = self.BCE_loss(p, semantics_gt.cuda())

            # ret['semantics_coarse'] = 0.5*(self.BCE_loss(p, semantics_gt.cuda()) + torch.mean(p*(1-p)))
            # ret['semantics_coarse'] = -torch.mean(gt * (gt*torch.log(p)+(1-gt)*torch.log(1-p)))

            # bce = -(0.67*gt * torch.log(p) +0.33*(1 - gt) * torch.log(1 - p))

            bce = -(gt * torch.log(p) + (1 - gt) * torch.log(1 - p))
            # bce = self.BCE_loss(p, semantics_gt.cuda())

            max_entropy_loss = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
            max_entropy_loss = torch.mean(max_entropy_loss)

            pt = torch.exp(-bce)
            self.gamma = 0

            # tv_loss = kornia.losses.TotalVariation()
            # l = len(p)
            # p_reshaped = p.reshape(1, 1, hparams['H'], hparams['W'])
            # res_tv_loss = tv_loss(p_reshaped) / l

            # ret['semantics_coarse'] = 0.5*torch.mean(((1 - pt + 1e-5) ** self.gamma) * bce) +0.5*max_entropy_loss #+ 1* res_tv_loss  #
            ret['semantics_coarse'] = torch.mean(((1 - pt + 1e-5) ** self.gamma) * bce)
            


        for k, v in ret.items():
            if k=='semantics_coarse' or  k=='semantics_fine':
                ret[k] = 0.1 * v  #self.coef * v #0.02 * v
            else:
                ret[k] = self.coef * v
        return ret, self.Annealing.getWeight(global_step)

    def mask_regularize(self, mask, size_delta, digit_delta):
        focus_epsilon = 0.02

        # # l2 regularize
        loss_focus_size = torch.pow(mask, 2)
        loss_focus_size = torch.mean(loss_focus_size) * size_delta

        loss_focus_digit = 1 / ((mask - 0.5)**2 + focus_epsilon)
        loss_focus_digit = torch.mean(loss_focus_digit) * digit_delta

        return loss_focus_size, loss_focus_digit
      
    def _l2_regularize(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss

loss_dict = {'color': ColorLoss,
             'hanerf': HaNeRFLoss}