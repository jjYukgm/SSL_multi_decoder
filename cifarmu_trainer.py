import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as vutils

import data
import config
import model

import random
import time
import os, sys
import math
import argparse
from collections import OrderedDict

import numpy as np
from utils import *
import losses, ramps
import copy
# from metrics import ArcMarginProduct    # cosine distance
from losses import FocalLoss
# import random
# import pdb
# pdb.set_trace()
# pdb.set_trace = lambda: None

class Trainer(object):
    def __init__(self, config, args):
        self.config = config
        for k, v in args.__dict__.items():
            setattr(self.config, k, v)
        setattr(self.config, 'save_dir', '{}_log'.format(self.config.dataset))
        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)

        disp_str = ''
        for attr in sorted(dir(self.config), key=lambda x: len(x)):
            if not attr.startswith('__'):
                disp_str += '{} : {}\n'.format(attr, getattr(self.config, attr))
        sys.stdout.write(disp_str)
        sys.stdout.flush()

        self.labeled_loader, self.unlabeled_loader, self.dev_loader, self.special_set \
            = data.get_data_loaders(config)

        self.gen = nn.ModuleList()
        self.gen.append(model.UNetWithResnet50Encoder(n_classes=3, res=config.gen_mode).cuda())
        for i in range(config.num_label-1):
            self.gen.append(model.Resnet50Decoder_skip(n_classes=3, res=config.gen_mode).cuda())
        if config.train_step != 1:
            batch_per_epoch = int((len(self.unlabeled_loader) + config.train_batch_size - 1) / config.train_batch_size)
            if config.step1_epo_lbl is not 0:
                epo_label = config.step1_epo_lbl
            else:
                epo_label = self.config.step1_epo * batch_per_epoch

            self.load_model(self.gen, 'G', epo_label, suffix=config.suffix+"_s1")
            for i in range(1, config.num_label):    # clone decoders
                self.gen[i].up_blocks = copy.deepcopy(self.gen[0].up_blocks)
            # create dis
            in_channels = [int(i) for i in config.dis_channels.split(",")]
            self.dis = model.Unet_Discriminator(config, in_channels=in_channels, ucnet=config.dis_uc).cuda()
            self.ema_dis = model.Unet_Discriminator(config, in_channels=in_channels, ucnet=config.dis_uc).cuda() # , ema=True).cuda()

        if hasattr(self, 'dis'):
            dis_para = [{'params': self.dis.parameters()},]

            if config.dis_double:
                self.dis_dou = model.Discriminative_out(config).cuda()
                dis_para.append({'params': self.dis_dou.parameters()})

            self.dis_optimizer = optim.Adam(dis_para, lr=config.dis_lr, betas=(0.5, 0.999))
        # self.dis_optimizer = optim.SGD(self.dis.parameters(), lr=config.dis_lr,
        #                                momentum=config.momentum,
        #                                weight_decay=config.weight_decay,
        #                                nesterov=config.nesterov)
        if hasattr(self, 'gen'):
            if config.gop == 'SGD':
                self.gen_optimizer = optim.SGD(self.gen.parameters(), lr=config.gen_lr,
                                               momentum=config.momentum,
                                               weight_decay=config.weight_decay,
                                               nesterov=config.nesterov)
            else:
                self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=config.gen_lr, betas=(0.0, 0.999))

        self.d_criterion = nn.CrossEntropyLoss()
        if config.consistency_type == 'mse':
            self.consistency_criterion = losses.softmax_mse_loss  # F.MSELoss()    # (size_average=False)
        elif config.consistency_type == 'kl':
            self.consistency_criterion = losses.softmax_kl_loss  # nn.KLDivLoss()  # (size_average=False)
        else:
            pass
        self.consistency_weight = 0

        # add step into data suffix
        self.config.suffix+="_s{}".format(config.train_step)
        log_path = os.path.join(self.config.save_dir, '{}.FM+VI.{}.txt'.format(
            self.config.dataset, self.config.suffix))
        if config.resume:
            self.logger = open(log_path, 'ab')
        else:
            self.logger = open(log_path, 'wb')
            self.logger.write(disp_str)

        # for arcface
        self.s = 30.0
        m = 0.50
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        # for dg start epoch
        if config.dg_start > 0:
            self.dg_flag = False
        else:
            self.dg_flag = True
        # for enc lab update:
        self.lab_feat_cen = None

        if hasattr(self, 'dis'):
            print self.dis

    def _get_vis_images(self, labels):
        labels = labels.data.cpu()
        vis_images = self.special_set.index_select(0, labels)
        return vis_images

    def gram_matrix(self, y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram

    def arcface_loss(self, x, linear, label):
        w = linear.weight
        cosine = F.linear(F.normalize(x), F.normalize(w))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        # if self.easy_margin:
        #     # phi = torch.where(cosine > 0, phi, cosine)
        #     phi = phi * (cosine > 0).float() + cosine *(cosine <= 0).float()
        # else:
            # phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        phi = phi * (cosine > self.th).float() + (cosine - self.mm)*(cosine <= self.th).float()
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot = Variable(torch.zeros(cosine.size()).cuda())
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        return output

    def step1_train(self, iter=None):
        # use unlabeled data train
        self.gen.train()
        # lab_images, lab_labels = self.labeled_loader.next()
        # lab_images, lab_labels = Variable(lab_images.cuda()), Variable(lab_labels.cuda())
        unl_images,_ = self.unlabeled_loader.next()
        unl_images = Variable(unl_images.cuda())
        # lab_cons_loss = 0
        unl_cons_loss = 0

        # 1 class/ gen[decoder]
        gen_feat2 = self.gen[0](unl_images, skip_encode=True)
        # for i in range(self.config.num_label):
        gen_images2 = self.gen[0].decode(gen_feat2)
        unl_cons_loss += nn.MSELoss()(gen_images2, unl_images)
        # mask = (lab_labels == i).nonzero()
        # if mask.nelement() <= 1:
        #     continue
        # input_images = lab_images[mask[:,0]]
        # gen_feat = self.gen[0](input_images, skip_encode=True)
        # gen_images = self.gen[i].decode(gen_feat)
        # lab_cons_loss += nn.MSELoss()(gen_images, input_images)

        # unl_cons_loss /= self.config.num_label
        g_loss = unl_cons_loss
        # g_loss = lab_cons_loss + unl_cons_loss

        self.gen_optimizer.zero_grad()
        g_loss.backward()
        self.gen_optimizer.step()
        monitor_dict = OrderedDict()
        # monitor_dict['lab con loss'] = lab_cons_loss.data[0]
        monitor_dict['unl con loss'] = unl_cons_loss.data[0]
        return monitor_dict

    def step2_train(self, iter=0, labeled=None, vis=False):
        config = self.config
        self.dis.train()
        # self.ema_dis.train()
        if config.dis_double: self.dis_dou.train()
        if hasattr(self, 'gen'): self.gen.train()

        ##### train Dis
        lab_images, lab_labels = self.labeled_loader.next()
        lab_images, lab_labels = Variable(lab_images.cuda()), Variable(lab_labels.cuda())

        unl_images, _ = self.unlabeled_loader.next()
        unl_images = Variable(unl_images.cuda())

        lab_loss = 0
        lab_loss2 = 0
        cons_loss = 0
        nei_loss = 0
        tri_loss = 0
        ult_loss = 0
        dgl_loss = 0
        lab_feat = self.gen[0](lab_images, encode=True)
        unl_feat = self.gen[0](unl_images, encode=True)
        gen_feat = None
        # if iter % config.dg_ratio == 0:
        gen_feat = self.gen[0](self.get_gens_img(unl_images).detach(), encode=True)
        if config.dis_uc:
            unl_logits, unl_uc = self.dis(unl_feat, uc=True)
            lab_logits, lab_uc = self.dis(lab_feat, uc=True)
        else:
            unl_logits = self.dis(unl_feat)
            lab_logits = self.dis(lab_feat)
        if gen_feat is not None:
            gen_logits = self.dis(gen_feat)

        # Standard classification loss
        if config.dis_uc:
            lab_loss,_ = losses.uncertainty_loss(self.d_criterion, lab_logits, lab_uc, lab_labels)
        else:
            lab_loss = self.d_criterion(lab_logits, lab_labels)
        if config.dis_double:
            lab_logits2 = self.dis_dou(lab_feat)
            lab_loss2 += self.d_criterion(lab_logits2, lab_labels)
        if config.dis_triple:
            lab_logits2 = self.dis_dou.out_net3(lab_feat)
            lab_loss2 += self.d_criterion(lab_logits2, lab_labels)
        lab_loss *= config.dl_weight
        lab_loss2 *= config.dl_weight



        # GAN true-fake loss: sumexp(logits) is seen as the input to the sigmoid
        unl_logsumexp = log_sum_exp(unl_logits)
        if gen_feat is not None:
            gen_logsumexp = log_sum_exp(gen_logits)

        true_loss = - 0.5 * torch.mean(unl_logsumexp) + 0.5 * torch.mean(F.softplus(unl_logsumexp))
        fake_loss = 0
        if gen_feat is not None:
            fake_loss = 0.5 * torch.mean(F.softplus(gen_logsumexp))
        unl_loss = config.du_weight * true_loss
        if self.dg_flag:
            unl_loss += config.dg_weight * fake_loss

        # ema consistency loss
        if config.nei_coef > 0 or config.con_coef > 0:
            ema_unl_logits = self.ema_dis(unl_feat)
            ema_unl_logits = Variable(ema_unl_logits.detach().data, requires_grad=False)
        if config.con_coef > 0:
            if config.dis_uc:
                cons_loss,_ = losses.uncertainty_loss(self.consistency_criterion,
                                                      unl_logits, unl_uc, ema_unl_logits)
            else:
                cons_loss = self.consistency_criterion(unl_logits, ema_unl_logits)
            cons_loss *= self.consistency_weight * config.con_coef
            cons_loss /= (config.train_batch_size + config.train_batch_size_2)

        if config.dis_double and config.dt_weight > 0:  # todo: add double, triple
            unl_logits2 = self.dis_dou(unl_feat)
            _, unl_lab1 = torch.max(unl_logits, 1)
            _, unl_lab2 = torch.max(unl_logits2, 1)
            tri_loss += self.d_criterion(unl_logits, unl_lab2)
            tri_loss += self.d_criterion(unl_logits2, unl_lab1)
            # GAN true-fake loss
            unl_logsumexp = log_sum_exp(unl_logits2)
            gen_logsumexp = log_sum_exp(self.dis_dou(gen_feat))
            true_loss = - 0.5 * torch.mean(unl_logsumexp) + 0.5 * torch.mean(F.softplus(unl_logsumexp))
            fake_loss = 0.5 * torch.mean(F.softplus(gen_logsumexp))
            ult_loss += true_loss + fake_loss

            if config.dis_triple:
                unl_logits3 = self.dis_dou.out_net3(unl_feat)
                _, unl_lab3 = torch.max(unl_logits3, 1)
                tri_loss += self.d_criterion(unl_logits, unl_lab3)
                tri_loss += self.d_criterion(unl_logits2, unl_lab3)
                tri_loss += self.d_criterion(unl_logits3, unl_lab1)
                tri_loss += self.d_criterion(unl_logits3, unl_lab2)
                unl_logsumexp = log_sum_exp(unl_logits3)
                gen_logsumexp = log_sum_exp(self.dis_dou.out_net3(gen_feat))
                true_loss = - 0.5 * torch.mean(unl_logsumexp) + 0.5 * torch.mean(F.softplus(unl_logsumexp))
                fake_loss = 0.5 * torch.mean(F.softplus(gen_logsumexp))
                ult_loss += true_loss + fake_loss
            tri_loss *= config.dt_weight
            ult_loss *= config.ut_weight

        if config.dgl_weight > 0:
            gen_lab_img, new_lbls = self.get_gens_img(lab_images, lbls=lab_labels)
            dgl_feat = self.gen[0](gen_lab_img, encode=True)
            if config.dis_uc:
                dgl_logits, dgl_uc = self.dis(dgl_feat, uc=True)
                dgl_loss,_ = losses.uncertainty_loss(self.d_criterion, dgl_logits, dgl_uc, new_lbls)
            else:
                dgl_logits = self.dis(dgl_feat)
                dgl_loss = self.d_criterion(dgl_logits, new_lbls)
            dgl_loss *= config.dgl_weight



        d_loss = lab_loss + unl_loss + cons_loss + lab_loss2 + tri_loss + ult_loss + dgl_loss

        ##### Monitoring (train mode)
        # true-fake accuracy
        unl_acc = torch.mean(nn.functional.sigmoid(unl_logsumexp.detach()).gt(0.5).float())
        # top-1 logit compared to 0: to verify Assumption (2) and (3)
        max_unl_acc = torch.mean(unl_logits.max(1)[0].detach().gt(0.0).float())
        gen_acc = None
        if gen_feat is not None:
            gen_acc = torch.mean(nn.functional.sigmoid(gen_logsumexp.detach()).gt(0.5).float())
            max_gen_acc = torch.mean(gen_logits.max(1)[0].detach().gt(0.0).float())

        self.dis_optimizer.zero_grad()
        if iter % (config.dg_ratio*config.eg_ratio) == 0:
            self.gen_optimizer.zero_grad()
        d_loss.backward()
        self.dis_optimizer.step()
        if iter % (config.dg_ratio*config.eg_ratio) == 0:
            self.gen_optimizer.step()

        # del no need
        # del gen_images
        del lab_feat
        del unl_feat
        del unl_logits
        if gen_feat is not None:
            del gen_logits
        del gen_feat

        ##### train Gen and Enc
        tv_loss = 0
        st_loss = 0
        fm_loss = 0
        gl_loss = 0
        gn_loss = 0
        gr_loss = 0
        gc_loss = 0
        ef_loss = 0
        el_loss = 0

        # enc lab center
        if iter % (self.batch_per_epoch*config.eg_ratio) == 0 and config.el_weight > 0:
            self.lab_feat_cen = [None] * config.num_label
            lab_num = [0] * config.num_label
            # all lab feat sum
            local_loader = self.labeled_loader.get_iter(shuffle=False)
            for img, lbl in local_loader:
                img, lbl = Variable(img.cuda()), Variable(lbl.cuda())
                loc_feat = self.gen[0](img, encode=True).detach()
                for i in range(config.num_label):
                    mask = (lbl == i).nonzero()
                    mask_num = mask.nelement()
                    if mask_num < 1:
                        continue
                    loc_feat2 = loc_feat[mask[:,0]]
                    if mask_num != 1:
                        loc_feat2 = torch.sum(loc_feat2, 0).unsqueeze(0)
                    if self.lab_feat_cen[i] is None:
                        self.lab_feat_cen[i] = loc_feat2
                    else:
                        self.lab_feat_cen[i] += loc_feat2
                    lab_num[i] += mask_num
            # feat sum -> feat mean
            for i in range(config.num_label):
                self.lab_feat_cen[i] = self.lab_feat_cen[i] / lab_num[i]


                # update # d / 1 g
        if iter % config.dg_ratio == 0:
            lab_feat = self.gen[0](lab_images, encode=True).detach()
            unl_feat = self.gen[0](unl_images, encode=True).detach()
            gen_images = self.get_gens_img(unl_images, spbatch=True, partcode=config.halfgnoise)
            unl_images = unl_images[range(gen_images.size(0))]
            img_per_gen = gen_images.size(0) // config.num_label



            if config.el_weight > 0:    # lbl mean cluster
                for i in range(config.num_label):
                    mask = (lab_labels == i).nonzero()
                    mask_num = mask.nelement()
                    if mask_num < 1:
                        continue
                    part_lab_feat = lab_feat[mask[:,0]]
                    el_loss += nn.KLDivLoss()(part_lab_feat, self.lab_feat_cen[i].repeat(mask_num, 1))
                el_loss *= config.el_weight

            if config.ef_weight > 0:    # lbl std < ts, total std > ts better
                ts = config.ef_ts
                for i in range(config.num_label):
                    mask = (lab_labels == i).nonzero()
                    mask_num = mask.nelement()
                    if mask_num <= 1:
                        continue
                    part_lab_feat = lab_feat[mask[:,0]]
                    plf_std = torch.std(part_lab_feat, 0)
                    ef_loss += torch.mean(torch.max(plf_std - ts,
                                                    Variable(torch.zeros(plf_std.size())).cuda()))
                ef_loss += ts
                # total std
                ef_std = torch.std(unl_feat, 0)
                ef_loss += torch.mean(torch.max(ts - ef_std, Variable(torch.zeros(ef_std.size())).cuda()))
                ef_loss *= config.ef_weight

            if config.gf_weight > 0 or config.gn_weight > 0: # or config.gl_weight > 0:
                gen_feat = self.gen[0](gen_images, encode=True)
            # gen lab feat loss: mean(En(xl)) - mean(En(De(En(xl))))
            if config.gl_weight > 0:
                diff_ul = torch.abs(torch.mean(lab_feat, 0) - torch.mean(unl_feat, 0))
                gl_ts = torch.mean(diff_ul) * 2
                for i in range(config.num_label):
                    mask = (lab_labels == i).nonzero()
                    mask_num = mask.nelement()
                    if mask_num < 1:
                        continue
                    # part_lab_images = lab_images[mask[:,0]]
                    # gen_lab_feat = self.gen[0](self.gen[i].decode(
                    #     self.gen[0](part_lab_images, skip_encode=True)), encode=True)
                    mean_mask_feat = lab_feat[mask[:,0]]
                    if mask_num != 1:
                        mean_mask_feat = torch.mean(mean_mask_feat, 0)
                        # gen_lab_feat = torch.mean(gen_lab_feat, 0)
                    # gen_unl_feat = self.gen[i].decode(self.gen[0](unl_images, skip_encode=True))
                    gen_unl_feat = torch.mean(gen_feat[range(i*img_per_gen, (i+1)*img_per_gen)], 0)
                    diff = torch.abs(mean_mask_feat - gen_unl_feat)
                    gl_loss += mask_num * \
                               torch.mean(torch.max(diff - gl_ts,
                                                    Variable(torch.zeros(diff.size())).cuda()))
                gl_loss /= lab_feat.size(0)
                gl_loss *= config.gl_weight

            # Feature matching loss:  En(xu) - En(De(En(xu)))
            if config.gf_weight > 0:
                fm_loss += nn.KLDivLoss()(torch.mean(gen_feat, 0), torch.mean(unl_feat, 0)) + \
                          torch.mean(torch.abs(torch.std(gen_feat, 0) - torch.std(unl_feat, 0)))
                # fm_loss = torch.mean(torch.abs(gen_feat - unl_feat[:gen_feat.size(0)]))
                fm_loss *= config.gf_weight

            if config.gc_weight > 0:
                key_ = "layer_{}".format(model.UNetWithResnet50Encoder.DEPTH - 1)
                feat_size = self.gen[0](unl_images, skip_encode=True)[key_][:img_per_gen*config.num_label].size()
                rand_codes = Variable(torch.rand(feat_size).cuda())  # .unsqueeze(-1).unsqueeze(-1)
                gen_rand_feat = self.gen[0](
                    self.get_gens_img(unl_images, codes=rand_codes), encode=True)
                rand_codes = rand_codes.mean(3, True).mean(2, True)   # .repeat(config.num_label, 1)
                gc_loss = nn.MSELoss()(gen_rand_feat, rand_codes)
                gc_loss *= config.gc_weight

            # reconstruction loss
            if config.gr_weight > 0:
                unl_tmp = unl_images[:img_per_gen].repeat(config.num_label, 1, 1, 1)
                # blur
                # get nn.L1Loss;F.MSELoss;nn.KLDivLoss
                gr_loss = nn.MSELoss()(gen_images, unl_tmp)
                gr_loss *= config.gr_weight

            # could impact the gr
            # gen neighbor loss: same => closer; diff => farther
            if config.gn_weight > 0:
                pos, neg = 0, 0
                diff = None
                for j in range(config.num_label-1):
                    gen_feat_j = gen_feat[range(j*img_per_gen, (j+1)*img_per_gen)]
                    for i in range(j+1, config.num_label):
                        # if i <= j:
                        #     continue
                        diff_ = gen_feat_j - \
                                gen_feat[range(i*img_per_gen, (i+1)*img_per_gen)]
                        diff_ = torch.mean(torch.abs(diff_), 0, True)
                        if diff is None:
                            diff = diff_
                        else:
                            diff = torch.cat((diff, diff_), dim=0)

                    mean_gen_feat_j = torch.mean(gen_feat_j, 0, True).repeat(img_per_gen, 1).detach()
                    pos += nn.KLDivLoss()(gen_feat_j, mean_gen_feat_j)
                gen_feat_j = gen_feat[range((config.num_label-1)*img_per_gen, (config.num_label)*img_per_gen)]
                mean_gen_feat_j = torch.mean(gen_feat_j, 0, True).repeat(img_per_gen, 1).detach()
                pos += nn.KLDivLoss()(gen_feat_j, mean_gen_feat_j)
                # pos /= config.num_label

                # diff = torch.mean(diff, 0, True)
                neg = torch.mean(torch.max(config.nei_margin - diff, Variable(torch.zeros(diff.size()).cuda())))

                # neg /= (config.num_label - 1) * gen_feat.size(1)    # * config.num_label
                gn_loss = pos + neg     # (torch.mean(torch.cat((pos, neg), 0)))
                gn_loss *= self.consistency_weight * config.gn_weight



            # neighbor loss
            if config.nei_coef > 0:
                tot_feat = torch.cat((lab_feat, unl_feat), dim=0)
                inds = torch.randperm(tot_feat.size(0)).cuda()
                # pdb.set_trace()
                # topk do
                if config.nei_top>1:
                    _, ema_lbl = torch.topk(ema_unl_logits,config.nei_top,dim=1)
                    ema_lbl = torch.zeros(ema_unl_logits.size()).cuda().scatter_(1,ema_lbl.data.long(),1)
                    lab_labels_tmp = torch.zeros(lab_logits.size()).cuda().scatter_(1,lab_labels.data.long().unsqueeze(1),1)
                    ema_lbl = Variable(torch.cat((lab_labels_tmp, ema_lbl), dim=0))
                    ema_lbl = ema_lbl[inds]
                    nei_mask = ema_lbl[:config.train_batch_size] * ema_lbl[config.train_batch_size:]
                    nei_mask = torch.sum(nei_mask, 1).float() / config.nei_top
                else:   # top1 do
                    _, ema_lbl = torch.max(ema_unl_logits, 1)
                    ema_lbl = torch.cat((lab_labels, ema_lbl), dim=0)
                    ema_lbl = ema_lbl[inds]
                    nei_mask = torch.eq(ema_lbl[:config.train_batch_size], ema_lbl[config.train_batch_size:]).float()  # nei or not
                tot_feat = tot_feat[inds]
                diff = tot_feat[:config.train_batch_size] - tot_feat[config.train_batch_size:]
                diff = torch.sqrt(torch.mean(diff ** 2, 1))
                pos = nei_mask * diff
                neg = (1 - nei_mask) * (torch.max(config.nei_margin - diff, Variable(torch.zeros(diff.size())).cuda()) ** 2)
                nei_loss = self.consistency_weight * config.nei_coef * \
                           (torch.mean(pos + neg))

            # tv losss
            if config.tv_weight > 0:
                (_, c_x, h_x, w_x) = gen_images.size()
                count_h = c_x * (h_x - 1) * w_x
                count_w = c_x * h_x * (w_x - 1)
                h_tv = torch.pow((gen_images[:, :, 1:, :] - gen_images[:, :, :-1, :]), 2).sum()
                w_tv = torch.pow((gen_images[:, :, :, 1:] - gen_images[:, :, :, :-1]), 2).sum()
                tv_loss = config.tv_weight * (h_tv / count_h + w_tv / count_w) / config.train_batch_size

            if config.st_weight > 0:
                # key = "layer_{}".format(model.UNetWithResnet50Encoder.DEPTH - 1)
                # gen_gram = self.gen[0](gen_images, skip_encode=True)
                # gen_gram = gen_gram[key]
                # gen_gram = self.gram_matrix(gen_gram)
                # unl_gram = self.gen[0](unl_images, skip_encode=True)
                # unl_gram = unl_gram[key].detach()
                # unl_gram = self.gram_matrix(unl_gram)
                gen_gram = self.gram_matrix(gen_images)
                unl_gram = self.gram_matrix(unl_images)
                st_loss += config.st_weight * nn.KLDivLoss()(gen_gram, unl_gram)

            # Generator loss
            g_loss = fm_loss + nei_loss + \
                     ef_loss + el_loss + \
                     tv_loss + st_loss + \
                     gl_loss + gn_loss + gr_loss + gc_loss

            self.gen_optimizer.zero_grad()
            g_loss.backward()
            self.gen_optimizer.step()

        monitor_dict = OrderedDict()
        monitor_dict['unl acc'] = unl_acc.data[0]
        if gen_acc is not None: monitor_dict['gen acc'] = gen_acc.data[0] * config.dg_ratio
        else: monitor_dict['gen acc'] = 0
        monitor_dict['max unl acc'] = max_unl_acc.data[0]
        if gen_acc is not None: monitor_dict['max gen acc'] = max_gen_acc.data[0] * config.dg_ratio
        else: monitor_dict['max gen acc'] = 0
        monitor_dict['lab loss'] = lab_loss.data[0]
        monitor_dict['unl loss'] = unl_loss.data[0]
        if config.dgl_weight > 0: monitor_dict['dgl loss'] = dgl_loss.data[0]
        if config.con_coef > 0: monitor_dict['con loss'] = cons_loss.data[0]
        if config.dis_double:
            monitor_dict['la2 loss'] = lab_loss2.data[0]
            if config.dt_weight > 0: monitor_dict['tri loss'] = tri_loss.data[0]
            if config.ut_weight > 0: monitor_dict['ult loss'] = ult_loss.data[0]
        if hasattr(self, 'gen') and iter % config.dg_ratio == 0:
            if config.gf_weight > 0: monitor_dict['fm loss'] = fm_loss.data[0] * config.dg_ratio
            if config.ef_weight > 0: monitor_dict['ef loss'] = ef_loss.data[0] * config.dg_ratio
            if config.el_weight > 0: monitor_dict['el loss'] = el_loss.data[0] * config.dg_ratio
            if config.tv_weight > 0: monitor_dict['tv loss'] = tv_loss.data[0] * config.dg_ratio
            if config.st_weight > 0: monitor_dict['st loss'] = st_loss.data[0] * config.dg_ratio
            if config.nei_coef > 0: monitor_dict['nei loss'] = nei_loss.data[0] * config.dg_ratio
            if config.gl_weight > 0: monitor_dict['gl loss'] = gl_loss.data[0] * config.dg_ratio
            if config.gn_weight > 0: monitor_dict['gn loss'] = gn_loss.data[0] * config.dg_ratio
            if config.gr_weight > 0: monitor_dict['gr loss'] = gr_loss.data[0] * config.dg_ratio
            if config.gc_weight > 0: monitor_dict['gc loss'] = gc_loss.data[0] * config.dg_ratio
            if config.gl_weight > 0: monitor_dict['gl ts'] = gl_ts.data[0] * config.dg_ratio
        elif iter % config.dg_ratio != 0:
            if config.gf_weight > 0: monitor_dict['fm loss'] = 0
            if config.ef_weight > 0: monitor_dict['ef loss'] = 0
            if config.el_weight > 0: monitor_dict['el loss'] = 0
            if config.tv_weight > 0: monitor_dict['tv loss'] = 0
            if config.st_weight > 0: monitor_dict['st loss'] = 0
            if config.nei_coef > 0: monitor_dict['nei loss'] = 0
            if config.gl_weight > 0: monitor_dict['gl loss'] = 0
            if config.gn_weight > 0: monitor_dict['gn loss'] = 0
            if config.gr_weight > 0: monitor_dict['gr loss'] = 0
            if config.gc_weight > 0: monitor_dict['gc loss'] = 0
            if config.gl_weight > 0: monitor_dict['gl ts'] = 0

        return monitor_dict

    def eval_true_fake(self, data_loader, max_batch=None):
        self.gen.eval()
        self.dis.eval()

        cnt = 0
        unl_acc, gen_acc, max_unl_acc, max_gen_acc = 0., 0., 0., 0.
        for i, (images, _) in enumerate(data_loader.get_iter()):
            images = Variable(images.cuda(), volatile=True)
            unl_feat = self.gen[0](images, encode=True)
            gen_feat = self.gen[0](self.get_gens_img(images), encode=True)

            unl_logits = self.dis(unl_feat)
            gen_logits = self.dis(gen_feat)

            unl_logsumexp = log_sum_exp(unl_logits)
            gen_logsumexp = log_sum_exp(gen_logits)

            ##### Monitoring (eval mode)
            # true-fake accuracy
            unl_acc += torch.mean(nn.functional.sigmoid(unl_logsumexp).gt(0.5).float()).data[0]
            gen_acc += torch.mean(nn.functional.sigmoid(gen_logsumexp).gt(0.5).float()).data[0]
            # top-1 logit compared to 0: to verify Assumption (2) and (3)
            max_unl_acc += torch.mean(unl_logits.max(1)[0].gt(0.0).float()).data[0]
            max_gen_acc += torch.mean(gen_logits.max(1)[0].gt(0.0).float()).data[0]

            cnt += 1
            if max_batch is not None and i >= max_batch - 1: break

        return unl_acc / cnt, gen_acc / cnt, max_unl_acc / cnt, max_gen_acc / cnt

    def eval(self, data_loader, max_batch=None, ema=False, tri=0):
        if ema:
            # if self.consistency_weight == 0.:
            #     return 0.
            dis = self.ema_dis
        else:
            dis = self.dis

        if tri == 0:
            dis_out = dis
        elif tri == 2:
            dis_out = self.dis_dou.out_net3
        else:   # 1
            dis_out = self.dis_dou.out_net2
        # self.gen.eval()
        dis.eval()

        loss, incorrect, cnt = 0, 0, 0
        for i, (images, labels) in enumerate(data_loader.get_iter()):
            images = Variable(images.cuda(), volatile=True)
            labels = Variable(labels.cuda(), volatile=True)
            feat = self.gen[0](images, encode=True)
            pred_prob = dis_out(feat)
            loss += self.d_criterion(pred_prob, labels).data[0]
            cnt += 1
            incorrect += torch.ne(torch.max(pred_prob, 1)[1], labels).data.sum()
            if max_batch is not None and i >= max_batch - 1: break

        return loss / cnt, incorrect

    def get_gens_img(self, images, spbatch=False, partcode=False, lbls=None, codes=None):
        # images: Variable(Tensor)
        gen_images = []
        if lbls is not None:
            new_lbls = []
        img_per_gen = images.size(0) // self.config.num_label
        num_part = []
        for j in range(self.config.num_label):
            if spbatch:
                num_part.append(range(img_per_gen))
            elif lbls is not None:
                mask = (lbls == j).nonzero().squeeze()
                num_mask = len(mask)
                num_part.append(mask)
                if num_mask < 1:
                    continue
                new_lbls += [j]*num_mask
            else:
                num_part.append(range(j*img_per_gen, (j+1)*img_per_gen))
        gen_feat = self.gen[0](images, skip_encode=True)
        if partcode:
            lay_key = "layer_{}".format(model.UNetWithResnet50Encoder.DEPTH - 1)
            keep_len = gen_feat[lay_key].size(1) // 2
            gn_size = gen_feat[lay_key][:,keep_len:].size()
            gen_feat[lay_key] = gen_feat[lay_key][:,:keep_len]
            gn = Variable(torch.rand(gn_size).cuda()) * 2
            gen_feat[lay_key] = torch.cat((gen_feat[lay_key], gn), 1)
        elif codes is not None:
            lay_key = "layer_{}".format(model.UNetWithResnet50Encoder.DEPTH - 1)
            # codes = codes[:gen_feat[lay_key].size(0)]
            gen_feat[lay_key] = codes
        for j in range(self.config.num_label):
            if len(num_part[j]) < 1:
                continue
            j_feat = dict()
            for i in gen_feat.keys():
                j_feat[i] = gen_feat[i][num_part[j]]
            gen_image = self.gen[j].decode(j_feat)
            gen_images.append(gen_image)
        gen_images = torch.cat(gen_images, 0)
        if lbls is not None:
            new_lbls = Variable(torch.from_numpy(np.array(new_lbls)).cuda())
            return gen_images, new_lbls
        else:
            return gen_images

    def visualize(self, data_loader):
        self.gen.eval()
        # self.dis.eval()

        vis_size = 100
        num_label = self.config.num_label
        img_per_batch = self.config.dev_batch_size // num_label
        img_per_batch *= num_label
        # nrow = int((10 // num_label)*num_label)
        inp_images = []
        gen_images = []
        for i, (images, _) in enumerate(data_loader.get_iter()):
            if i * self.config.dev_batch_size >= vis_size:
                break
            inp_images.append(images[:img_per_batch])
            images = Variable(images.cuda(), volatile=True)
            gen_images.append(self.get_gens_img(images))

        inp_images = torch.cat(inp_images, 0)
        gen_images = torch.cat(gen_images, 0)

        save_path = os.path.join(self.config.save_dir,
                                 '{}.FM+VI.{}_.png'.format(self.config.dataset, self.config.suffix))
        vutils.save_image(gen_images.data.cpu(), save_path, normalize=True, range=(-1, 1), nrow=10)
        save_path = os.path.join(self.config.save_dir,
                                 '{}.FM+VI.{}_d.png'.format(self.config.dataset, self.config.suffix))
        vutils.save_image(inp_images, save_path, normalize=True, range=(-1, 1), nrow=10)

    def param_init(self):
        def func_gen(flag):
            def func(m):
                if hasattr(m, 'init_mode'):
                    setattr(m, 'init_mode', flag)

            return func

        images = []
        num_img = 500

        for i in range(num_img // self.config.train_batch_size):
            lab_images, _ = self.labeled_loader.next()
            images.append(lab_images)
        images = torch.cat(images, 0)

        if hasattr(self, 'dis'):
            self.dis.apply(func_gen(True))
            if self.config.dis_double: self.dis_dou.apply(func_gen(True))
            feat = self.gen[0](Variable(images.cuda()), encode=True)
            if self.config.dis_uc:
                logits,_ = self.dis(feat, uc=True)
            else:
                logits = self.dis(feat)
            if self.config.dis_double: logits = self.dis_dou(feat)
            if self.config.dis_triple: logits = self.dis_dou.out_net3(feat)
            self.dis.apply(func_gen(False))
            if self.config.dis_double: self.dis_dou.apply(func_gen(False))

            self.ema_dis = copy.deepcopy(self.dis) # clone weight_scale and weight

    def calculate_remaining(self, t1, t2, epoch):  # ta
        tot_progress = (epoch + 0.) / self.config.max_epochs
        if self.config.resume:
            progress = (epoch - self.config.last_epochs + 0.) / (self.config.max_epochs- self.config.last_epochs)
        else:
            progress = tot_progress
        elapsed_time = t2 - t1
        if (progress > 0):
            remaining_time = elapsed_time * (1 / progress) - elapsed_time
        else:
            remaining_time = 0

        # elapsed time
        esec = int(elapsed_time % 60)
        emin = int((elapsed_time // 60) % 60)
        ehr = int(elapsed_time / 3600)
        # remaining_time
        rsec = int(remaining_time % 60)
        rmin = int((remaining_time // 60) % 60)
        rhr = int(remaining_time / 3600)
        time_str = '[{:8.2%}], {:3d}:{:2d}:{:2d}<{:3d}:{:2d}:{:2d} '.format(tot_progress, ehr, emin, esec, rhr, rmin, rsec)

        time_str = '| ' + time_str + '\n'
        return time_str

    def save_model(self, net, net_label, epo_label):  # ta
        save_filename = 'VI.{}_{}_net_{}.pth'.format(self.config.suffix, epo_label, net_label)
        save_path = os.path.join(self.config.save_dir, save_filename)
        torch.save(net.cpu().state_dict(), save_path)
        if torch.cuda.is_available():
            net.cuda()

    def del_model(self, net_label, epo_label):  # ta
        del_filename = 'VI.{}_{}_net_{}.pth'.format(self.config.suffix, epo_label, net_label)
        del_path = os.path.join(self.config.save_dir, del_filename)
        if os.path.exists(del_path):
            os.remove(del_path)
        else:
            print("The file does not exist, {}".format(del_path))

    def load_model(self, net, net_label, epo_label, suffix=None):  # ta
        if suffix is None:
            suffix = self.config.suffix
        load_filename = 'VI.{}_{}_net_{}.pth'.format(suffix, epo_label, net_label)
        load_path = os.path.join(self.config.save_dir, load_filename)
        load_net = torch.load(load_path)
        net.cpu()
        model.load_my_state_dict(net, load_net)

        if torch.cuda.is_available():
            net.cuda()

    def save(self, epo_label):  # ta
        # save new
        if hasattr(self, 'dis'):
            self.save_model(self.dis, 'D', epo_label)
            self.save_model(self.ema_dis, 'M', epo_label)
        if hasattr(self, 'gen'):
            self.save_model(self.gen, 'G', epo_label)
        if hasattr(self, 'dis_dou'):
            self.save_model(self.dis_dou, 'D2', epo_label)
        # del old
        if epo_label >= self.config.vis_period:
            epo_label -= self.config.vis_period
            if hasattr(self, 'dis'):
                self.del_model('D', epo_label)
                self.del_model('M', epo_label)
            if hasattr(self, 'gen'):
                self.del_model('G', epo_label)
            if hasattr(self, 'dis_dou'):
                self.save_model('D2', epo_label)

    def resume(self, epo_label):  # ta
        # load old
        if hasattr(self, 'dis'):
            self.load_model(self.dis, 'D', epo_label)
            self.load_model(self.ema_dis, 'M', epo_label)
        if hasattr(self, 'gen'):
            self.load_model(self.gen, 'G', epo_label)
        if hasattr(self, 'dis_dou'):
            self.load_model(self.dis_dou, 'D2', epo_label)

    def adjust_learning_rate(self, optimizer, lr, ini_lr, epoch):

        # LR warm-up to handle large minibatch sizes from https://arxiv.org/abs/1706.02677
        lr = ramps.linear_rampup(epoch, self.config.lr_rampup) * (lr - ini_lr) + ini_lr

        # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
        if self.config.lr_rampdn:
            assert self.config.lr_rampdn >= self.config.max_epochs
            lr *= ramps.cosine_rampdown(epoch, self.config.lr_rampdn )

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def update_ema_variables(self, alpha, global_step, batch_per_epoch):  # ta2
        # alpha: min of weight reservation, hp
        # global_step: history update step counts
        # Use the true average until the exponential average is more correct
        epoch = global_step / batch_per_epoch
        if epoch < self.config.t_start:
            return
        alpha = min(1 - 1 / (global_step + 1), alpha)
        if epoch == self.config.t_start \
                or self.config.t_forget_coef == 0 \
                or (self.config.t_forget_coef > 0.
                    and epoch % (self.config.c_rampup * self.config.t_forget_coef) == 0):
            alpha = 0.
        for ema_param, param in zip(self.ema_dis.parameters(), self.dis.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def get_current_consistency_weight(self, epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        if epoch < self.config.t_start:
            self.consistency_weight = 0.
        else:
            self.consistency_weight = ramps.sigmoid_rampup(epoch, self.config.c_rampup)

    def train(self):
        config = self.config
        batch_per_epoch = int((len(self.unlabeled_loader) +
                               config.train_batch_size - 1) / config.train_batch_size)
        self.batch_per_epoch = batch_per_epoch
        if not config.resume:
            self.param_init()
            # self.iter_cnt = 0
            iter = 0
        else:
            # self.iter_cnt = 0 + config.last_epochs
            if config.last_epo_lbl is not 0:
                iter = config.last_epo_lbl
            else:
                iter = batch_per_epoch*(config.last_epochs)
            self.resume(iter)
            iter += 1

        min_dev_incorrect = e_mdi = 1e6
        monitor = OrderedDict()

        if config.eval_period == -1:
            config.eval_period = batch_per_epoch
            self.config.eval_period = batch_per_epoch
        if config.vis_period == -1:
            config.vis_period = batch_per_epoch
        if config.t_start == -1:
            config.t_start = 1. / batch_per_epoch

        min_lr = config.min_lr if hasattr(config, 'min_lr') else 0.0
        last_tr_inco = [1.0]
        start_time = time.time()
        while True:
            if iter % batch_per_epoch == 0:
                epoch = iter / batch_per_epoch
                if not self.dg_flag and config.dg_start <= epoch:
                    self.dg_flag = True
                epoch_ratio = float(epoch) / float(config.max_epochs)
                # use another outer max to prevent any float computation precision problem
                if hasattr(self, 'dis'):
                    self.dis_optimizer.param_groups[0]['lr'] = max(min_lr, config.dis_lr *
                                                                   min(3. * (1. - epoch_ratio), 1.))
                if hasattr(self, 'gen'):
                    self.gen_optimizer.param_groups[0]['lr'] = max(min_lr, config.gen_lr *
                                                                   min(3. * (1. - epoch_ratio), 1.))

            self.get_current_consistency_weight(iter / batch_per_epoch)
            if hasattr(self, 'dis'):
                self.adjust_learning_rate(self.dis_optimizer, config.dis_lr,
                                          config.ini_lr, iter / batch_per_epoch)
            iter_vals = getattr(self, "step{}_train".format(config.train_step))(iter=iter)
            if hasattr(self, 'dis'):
                self.update_ema_variables(self.config.ema_decay, iter, batch_per_epoch)

            if len(monitor.keys()) == 0:
                for k in iter_vals.keys():
                    monitor[k] = 0.
                    # if not monitor.has_key(k):
                    #     monitor[k] = 0.
            for k, v in iter_vals.items():
                monitor[k] += v

            if iter % config.eval_period == 0:
                if hasattr(self, 'dis'):
                    train_loss, train_incorrect = self.eval(self.labeled_loader)
                    dev_loss, dev_incorrect = self.eval(self.dev_loader)
                    ema_result = self.eval(self.dev_loader, ema=True)
                    if isinstance(ema_result, tuple):
                        ema_train_result = self.eval(self.labeled_loader, ema=True)
                        ema_train_result_ = ema_train_result[1] / (1.0 * len(self.labeled_loader))
                        ema_result_ = ema_result[1] / (1.0 * len(self.dev_loader))
                    if config.dis_double:
                        _, tri_result1 = self.eval(self.dev_loader, tri=1)
                        tri_result1 = tri_result1 / (1.0 * len(self.dev_loader))
                        if self.config.dis_triple:
                            _, tri_result2 = self.eval(self.dev_loader, tri=2)
                            tri_result2 = tri_result2 / (1.0 * len(self.dev_loader))
                        else:
                            tri_result2 = 0.
                    if hasattr(self, 'gen'):
                        unl_acc, gen_acc, max_unl_acc, max_gen_acc = self.eval_true_fake(self.dev_loader, 10)

                disp_str = "#{}".format(iter)
                if hasattr(self, 'dis'):
                    train_incorrect /= 1.0 * len(self.labeled_loader)
                    dev_incorrect /= 1.0 * len(self.dev_loader)
                    min_dev_incorrect = min(min_dev_incorrect, dev_incorrect)
                    e_mdi = min(e_mdi, ema_result_)
                    disp_str += '\ttrain: {:.4f}, {:.4f} | dev: {:.4f}, {:.4f} | best: {:.4f}'.format(
                        train_loss, train_incorrect, dev_loss, dev_incorrect, min_dev_incorrect)
                    if isinstance(ema_result, tuple):
                        disp_str += ' | ema: {:.4f}, {:.4f}, {:.4f}'.format(ema_train_result_, ema_result_, e_mdi)
                    else:
                        disp_str += ' | ema:   None ,   None'
                    if config.dis_double:
                        disp_str += ' | tri: {:.4f}, {:.4f}'.format(tri_result1, tri_result2)

                for k, v in monitor.items():
                    disp_str += ' | {}: {:.4f}'.format(k, v / config.eval_period)

                if hasattr(self, 'dis') and hasattr(self, 'gen'):
                    disp_str += ' | [Eval] unl acc: {:.4f}, gen acc: {:.4f}, max unl acc: {:.4f}, max gen acc: {:.4f}'.format(
                        unl_acc, gen_acc, max_unl_acc, max_gen_acc)
                if hasattr(self, 'dis'):
                    disp_str += ' | dlr: {:.5f}'.format(self.dis_optimizer.param_groups[0]['lr'])
                elif hasattr(self, 'gen'):
                    disp_str += ' | glr: {:.5f}'.format(self.gen_optimizer.param_groups[0]['lr'])
                disp_str += '\n'

                monitor = OrderedDict()

                # timer   # ta
                time_str = self.calculate_remaining(start_time, time.time(), iter / batch_per_epoch)

                self.logger.write(disp_str)
                sys.stdout.write(disp_str)
                sys.stdout.write(time_str)  # ta
                sys.stdout.flush()

                # stop check
                thres = 1 #0.4; 0.3
                if hasattr(self, 'dis') and train_incorrect > sum(last_tr_inco)/len(last_tr_inco) + thres:
                    print("tr_inco encrease > {}!".format(thres))
                    break
                elif hasattr(self, 'dis'):
                    last_tr_inco.append(train_incorrect)
                    if len(last_tr_inco) > 3:
                        last_tr_inco.pop(0)
                epoch = iter / batch_per_epoch
                if epoch >= config.max_epochs:
                    self.save(iter)
                    self.visualize(self.dev_loader)
                    break

            if iter % config.vis_period == 0:
                # save model   # ta
                self.save(iter)
                self.visualize(self.dev_loader)




            iter += 1
            # self.iter_cnt += 1


if __name__ == '__main__':
    cc = config.cifarmu_config()
    parser = argparse.ArgumentParser(description='cifarmu_trainer.py')
    parser.add_argument('-suffix', default='mu0', type=str, help="Suffix added to the save images.")
    parser.add_argument('-r', dest='resume', action='store_true')
    parser.add_argument('-num_label', default=cc.num_label, type=int,
                        help="label num")
    parser.add_argument('-allowed_label', default=cc.allowed_label, type=str,
                        help="allowed label in dataset")
    parser.add_argument('-dataset', default=cc.dataset, type=str,
                        help="dataset: cifar, stl10, coil20")
    parser.add_argument('-image_side', default="32", type=int,
                        help="cifar: 32, stl10: 96")
    parser.add_argument('-train_step', default=cc.train_step, type=int,
                        help="train step: 1, 2")
    parser.add_argument('-step1_epo', default=0, type=int,
                        help="load gen from train step 1 epo #")
    parser.add_argument('-step1_epo_lbl', default=0, type=int,
                        help="load gen from train step 1 epo label #")
    parser.add_argument('-dg_ratio', default=5, type=int,
                        help="update # d/g")
    parser.add_argument('-eg_ratio', default=1, type=int,
                        help="update # g/enc-d")
    parser.add_argument('-dis_channels', default=cc.dis_channels, type=str,
                        help="# of dis channels, r50: '1024,192'; r34: '512,192' ")
    parser.add_argument('-max_epochs', default=cc.max_epochs, type=int,
                        help="max epoches")
    parser.add_argument('-last_epochs', default=cc.last_epochs, type=int,
                        help="last epochs")
    parser.add_argument('-last_epo_lbl', default=0, type=int,
                        help="last epochs label")
    parser.add_argument('-dg_start', default=cc.dg_start, type=int,
                        help="start dis loss epoch")
    parser.add_argument('-eval_period', default=cc.eval_period, type=int,
                        help="evaluate period, -1: per-epoch")
    parser.add_argument('-vis_period', default=cc.vis_period, type=int,
                        help="visualize period, -1: per-epoch")
    parser.add_argument('-ld', '--size_labeled_data', default=cc.size_labeled_data, type=int,
                        help="labeled data num")
    parser.add_argument('-ud', '--size_unlabeled_data', default=cc.size_unlabeled_data, type=int,
                        help="unlabeled data num")
    parser.add_argument('-train_batch_size', default=cc.train_batch_size, type=int,
                        help="labeled batch size")
    parser.add_argument('-train_batch_size_2', default=cc.train_batch_size_2, type=int,
                        help="unlabeled batch size")
    parser.add_argument('-dis_lr', default=cc.dis_lr, type=float,
                        help="discriminator learn rate")
    parser.add_argument('-gen_lr', default=cc.gen_lr, type=float,
                        help="generator learn rate")
    # parser.add_argument('-weight_decay', default=cc.weight_decay, type=float,
    #                     help="generator weight decay")
    parser.add_argument('-gop', default=cc.gop, type=str,
                        help="gen optim: Adam, SGD")
    parser.add_argument('-con_coef', default=cc.con_coef, type=float,
                        help="Consistency loss content")
    parser.add_argument('-nei_coef', default=cc.nei_coef, type=float,
                        help="neighbor loss content")
    parser.add_argument('-nei_margin', default=cc.nei_margin, type=float,
                        help="neighbor margin content; less better")
    parser.add_argument('-nei_top', default=cc.nei_top, type=int,
                        help="neighbor top-k")
    parser.add_argument('-c_rampup', default=cc.c_rampup, type=int,
                        help="rampup period")
    parser.add_argument('-ini_lr', default=cc.ini_lr, type=float,
                        help="lr rampup ini")
    parser.add_argument('-lr_rampup', default=cc.lr_rampup, type=int,
                        help="lr rampup fin epoch")
    parser.add_argument('-lr_rampdn', default=cc.lr_rampdn, type=int,
                        help="lr rampdn fin epoch")
    parser.add_argument('-t_forget_coef', default=cc.t_forget_coef, type=float,
                        help="teacher corget content * c_r, 0: always forget, -1: no forget")
    parser.add_argument('-t_start', default=cc.t_start, type=float,
                        help="teacher start calculate loss, -1: 2nd batch start")
    parser.add_argument('-dl_weight', default=cc.dl_weight, type=float,
                        help="dis lab loss content")
    parser.add_argument('-du_weight', default=cc.du_weight, type=float,
                        help="dis unlabeled loss content")
    parser.add_argument('-dg_weight', default=cc.dg_weight, type=float,
                        help="dis gen loss content")
    parser.add_argument('-dgl_weight', default=0, type=float,
                        help="dis gen lab loss content")
    parser.add_argument('-dt_weight', default=cc.dt_weight, type=float,
                        help="dis triple loss content")
    parser.add_argument('-ut_weight', default=cc.ut_weight, type=float,
                        help="dis triple gan loss content")
    parser.add_argument('-ef_weight', default=0, type=float,
                        help="encode feat mean & std loss content")
    parser.add_argument('-ef_ts', default=0.3, type=float,
                        help="encode feat threshold, def: 0.3")
    parser.add_argument('-el_weight', default=0, type=float,
                        help="encode lab feat mean to clustering")
    parser.add_argument('-tv_weight', default=cc.tv_weight, type=float,
                        help="tv loss weight")
    parser.add_argument('-st_weight', default=cc.st_weight, type=float,
                        help="style loss weight")
    parser.add_argument('-gf_weight', default=1, type=float,
                        help="gen feat measure loss content")
    parser.add_argument('-gl_weight', default=cc.gl_weight, type=float,
                        help="gen lab feat loss content")
    parser.add_argument('-gn_weight', default=0, type=float,
                        help="gen feat nei loss content")
    parser.add_argument('-gr_weight', default=0, type=float,
                        help="gen reconstruct loss content")
    parser.add_argument('-gc_weight', default=0, type=float,
                        help="gen code loss content")
    parser.add_argument('-gen_mode', default=cc.gen_mode, type=str,
                        help="gen model mode: res '50', '34', non")
    parser.add_argument('-f', dest='flip', action='store_true')
    parser.add_argument('-dd', dest='dis_double', action='store_true',
                        help="double dis")
    parser.add_argument('-dt', dest='dis_triple', action='store_true',
                        help="trible dis")
    parser.add_argument('-hgn', dest='halfgnoise', action='store_true',
                        help="whether the wwhole E(img) is the input of Decoder")
    parser.add_argument('-uc', dest='dis_uc', action='store_true',
                        help="dis uncertainty or not")
    parser.set_defaults(resume=False)
    parser.set_defaults(dis_double=False)
    parser.set_defaults(dis_triple=False)
    parser.set_defaults(halfgnoise=False)
    parser.set_defaults(dis_uc=False)
    parser.set_defaults(flip=cc.flip)
    args = parser.parse_args()

    trainer = Trainer(cc, args)
    trainer.train()
