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

        disp_str = ''
        for attr in sorted(dir(self.config), key=lambda x: len(x)):
            if not attr.startswith('__'):
                disp_str += '{} : {}\n'.format(attr, getattr(self.config, attr))
        sys.stdout.write(disp_str)
        sys.stdout.flush()

        self.labeled_loader, self.unlabeled_loader, self.dev_loader, self.special_set = data.get_cifar_loaders(
            config)

        self.dis = model.Discriminative(config).cuda()
        self.ema_dis = model.Discriminative(config).cuda() # , ema=True).cuda()
        # for param in self.ema_dis.parameters():
        #     param.detach_()
        if config.gen_mode != "non":
            self.gen = model.generator(image_side=config.image_side, noise_size=config.noise_size, large=config.double_input_size, gen_mode=config.gen_mode).cuda()

        dis_para = [{'params': self.dis.parameters()},]
        if 'm' in config.dis_mode:  # svhn: 168; cifar:192
            self.m_criterion = FocalLoss(gamma=2)

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
        if config.gen_mode == "z2i":
            self.enc = model.Encoder(config.image_side, noise_size=config.noise_size, output_params=True).cuda()
            self.enc_optimizer = optim.Adam(self.enc.parameters(), lr=config.enc_lr, betas=(0.0, 0.999))

        self.d_criterion = nn.CrossEntropyLoss()
        if config.consistency_type == 'mse':
            self.consistency_criterion = losses.softmax_mse_loss  # F.MSELoss()    # (size_average=False)
        elif config.consistency_type == 'kl':
            self.consistency_criterion = losses.softmax_kl_loss  # nn.KLDivLoss()  # (size_average=False)
        else:
            pass
        self.consistency_weight = 0

        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)

        if "," in config.dis_mode or config.cd_mode_iter > 0:
            assert "," in config.dis_mode
            assert config.cd_mode_iter > 0
            self.dis_mode = config.dis_mode
            config.dis_mode = config.dis_mode.split(",")[0]

        log_path = os.path.join(self.config.save_dir, '{}.FM+VI.{}.txt'.format(self.config.dataset, self.config.suffix))
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

    def _train(self, labeled=None, vis=False):
        config = self.config
        self.dis.train()
        self.ema_dis.train()
        if config.dis_double: self.dis_dou.train()
        if hasattr(self, 'gen'): self.gen.train()
        if config.gen_mode == "z2i": self.enc.train()
        # if 'm' in config.dis_mode: self.metric_fc.train()

        ##### train Dis
        lab_images, lab_labels = self.labeled_loader.next()
        lab_images, lab_labels = Variable(lab_images.cuda()), Variable(lab_labels.cuda())

        unl_images, _ = self.unlabeled_loader.next()
        unl_images = Variable(unl_images.cuda())
        gen_images = None
        gla_images = None
        if config.gen_mode == "z2i":
            noise = Variable(torch.Tensor(unl_images.size(0), config.noise_size).uniform_().cuda())
            gen_images = self.gen(noise)
            if config.gl_weight > 0:    # gen z label
                gla_labels = torch.Tensor(range(config.num_label))
                gla_labels = gla_labels.unsqueeze(1).repeat(1, config.train_batch_size_2 / config.num_label).view(-1).long()
                labels_oh = Variable(torch.zeros(config.train_batch_size_2, config.num_label).scatter_(1, gla_labels.unsqueeze(1), 1).cuda())
                if config.dgl_weight > 0:
                    lab_noise = Variable(torch.Tensor(config.train_batch_size_2, config.noise_size - config.num_label).uniform_().cuda())
                    lab_noise = torch.cat((labels_oh, lab_noise), dim=1)
                    gla_images = self.gen(lab_noise)
                gla_labels = Variable(gla_labels.cuda())
        if config.gen_mode == "i2i":
            gen_images = self.gen(unl_images)


        # Standard classification loss
        lab_loss = 0
        arc_loss = 0
        lab_loss2 = 0
        cons_loss = 0
        nei_loss = 0
        tri_loss = 0
        ult_loss = 0
        dgl_loss = 0
        lab_feat = self.dis(lab_images, feat=True)
        unl_feat = self.dis(unl_images, feat=True)
        unl_logits = self.dis.out_net(unl_feat)
        if gen_images is not None:
            gen_feat = self.dis(gen_images.detach(), feat=True)
            gen_logits = self.dis.out_net(gen_feat)
        if 'd' in config.dis_mode:
            lab_logits = self.dis.out_net(lab_feat)
            lab_loss += self.d_criterion(lab_logits, lab_labels)
            if config.dis_double:
                lab_logits2 = self.dis_dou(lab_feat)
                lab_loss2 += self.d_criterion(lab_logits2, lab_labels)
            if config.dis_triple:
                lab_logits2 = self.dis_dou.out_net3(lab_feat)
                lab_loss2 += self.d_criterion(lab_logits2, lab_labels)
        lab_loss *= config.dl_weight
        lab_loss2 *= config.dl_weight

        if 'm' in config.dis_mode:  # arcface
            lab_logits = self.arcface_loss(lab_feat, self.dis.out_net.weight, lab_labels)
            arc_loss += self.m_criterion(lab_logits, lab_labels)
            arc_loss *= config.da_weight


        # GAN true-fake loss: sumexp(logits) is seen as the input to the sigmoid
        unl_logsumexp = log_sum_exp(unl_logits)
        if gen_images is not None:
            gen_logsumexp = log_sum_exp(gen_logits)

        true_loss = - 0.5 * torch.mean(unl_logsumexp) + 0.5 * torch.mean(F.softplus(unl_logsumexp))
        fake_loss = 0
        if gen_images is not None:
            fake_loss = 0.5 * torch.mean(F.softplus(gen_logsumexp))
        if gla_images is not None and config.dgl_weight > 0:
            gla_logits = self.dis(gla_images.detach())
            dgl_loss -= self.consistency_weight * config.dgl_weight * self.d_criterion(gla_logits, gla_labels)
        unl_loss = config.du_weight * true_loss
        if self.dg_flag:
            unl_loss += config.dg_weight * fake_loss

        # ema consistency loss
        if config.nei_coef > 0 or config.con_coef > 0:
            ema_lab_logits = self.ema_dis(lab_images) if 'd' in config.dis_mode \
                else self.emetric_fc.test(self.ema_dis(lab_images, feat=True))
            ema_unl_logits = self.ema_dis(unl_images) if 'd' in config.dis_mode \
                else self.emetric_fc.test(self.ema_dis(unl_images, feat=True))
            ema_lab_logits = Variable(ema_lab_logits.detach().data, requires_grad=False)
            ema_unl_logits = Variable(ema_unl_logits.detach().data, requires_grad=False)
        if config.con_coef > 0:
            cons_loss = self.consistency_weight * config.con_coef * \
                        (self.consistency_criterion(lab_logits, ema_lab_logits) +
                         self.consistency_criterion(unl_logits, ema_unl_logits)) \
                        / (config.train_batch_size + config.train_batch_size_2)

        # neighbor loss
        if config.nei_coef > 0:
            tot_feat = torch.cat((lab_feat, unl_feat), dim=0)
            inds = torch.randperm(tot_feat.size(0)).cuda()
            # pdb.set_trace()
            # topk do
            if config.nei_top>1:
                _, ema_lbl = torch.topk(ema_unl_logits,config.nei_top,dim=1)
                ema_lbl = torch.zeros(ema_unl_logits.size()).cuda().scatter_(1,ema_lbl.data.long(),1)
                lab_labels_tmp = torch.zeros(ema_lab_logits.size()).cuda().scatter_(1,lab_labels.data.long().unsqueeze(1),1)
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

        if config.dis_double and config.dt_weight > 0:
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



        d_loss = lab_loss + unl_loss + cons_loss + nei_loss + arc_loss + lab_loss2 + tri_loss + ult_loss + dgl_loss

        ##### Monitoring (train mode)
        # true-fake accuracy
        unl_acc = torch.mean(nn.functional.sigmoid(unl_logsumexp.detach()).gt(0.5).float())
        # top-1 logit compared to 0: to verify Assumption (2) and (3)
        max_unl_acc = torch.mean(unl_logits.max(1)[0].detach().gt(0.0).float())
        if gen_images is not None:
            gen_acc = torch.mean(nn.functional.sigmoid(gen_logsumexp.detach()).gt(0.5).float())
            max_gen_acc = torch.mean(gen_logits.max(1)[0].detach().gt(0.0).float())

        self.dis_optimizer.zero_grad()
        d_loss.backward()
        self.dis_optimizer.step()

        ##### train Gen and Enc
        vi_loss = 0
        gf_loss = 0
        tv_loss = 0
        st_loss = 0
        gl_loss = 0
        im_loss = 0
        cim_loss = 0
        il_loss = 0
        if hasattr(self, 'gen'):
            unl_feat = Variable(unl_feat.detach().data, requires_grad=False)
        if config.gen_mode == "z2i":
            # instead of using just trained g_img
            noise = Variable(torch.Tensor(unl_images.size(0), config.noise_size).uniform_().cuda())
            gen_images = self.gen(noise)

            # Entropy loss via variational inference
            mu, log_sigma = self.enc(gen_images)
            vi_loss = config.vi_weight * gaussian_nll(mu, log_sigma, noise)
            # e,g real image feat loss
            if config.gf_weight > 0:
                mu, sig = self.enc(lab_images)
                enc_noise2 = mu + sig * Variable(torch.normal(0, torch.ones(sig.size()))).cuda()
                gel_images = self.gen(enc_noise2)
                mu, sig = self.enc(unl_images)
                enc_noise3 = mu + sig * Variable(torch.normal(0, torch.ones(sig.size()))).cuda()
                geu_images = self.gen(enc_noise3)
                lab_feat = Variable(lab_feat.detach().data, requires_grad=False)
                gel_feat = self.dis(gel_images, feat=True)
                geu_feat = self.dis(geu_images, feat=True)
                lab_feat = torch.cat((lab_feat, unl_feat), dim=0)
                geu_feat = torch.cat((gel_feat, geu_feat), dim=0)
                mean_diff = torch.mean(geu_feat, 0) - torch.mean(lab_feat, 0)  # diff_mean
                mean_diff = torch.norm(mean_diff, 2) / mean_diff.nelement()  # (1/d)*norm(diff)
                gf_loss = self.consistency_weight * config.gf_weight * torch.mean(mean_diff)  # add con weight

            if config.gl_weight > 0:    # gen z lab loss
                assert config.train_batch_size_2 % config.num_label == 0, "mod(bsize, num_labels) is not 0"
                # label
                # labels = torch.Tensor(range(config.num_label))
                # labels = labels.unsqueeze(1).repeat(1, config.train_batch_size_2 / config.num_label).view(-1).long()
                # labels_oh = Variable(torch.zeros(config.train_batch_size_2, config.num_label).scatter_(1, labels.unsqueeze(1), 1).cuda())
                lab_noise = Variable(torch.Tensor(config.train_batch_size_2, config.noise_size - config.num_label).uniform_().cuda())
                lab_noise = torch.cat((labels_oh, lab_noise), dim=1)
                gla_images = self.gen(lab_noise)
                gl_feat = self.dis(gla_images, feat=True)
                gl_logits = self.dis.out_net(gl_feat)
                if config.dis_double:
                    gl_logits += self.dis_dou.out_net2(gl_feat)
                    if config.dis_triple:
                        gl_logits += self.dis_dou.out_net3(gl_feat)
                        gl_logits /= 3
                    else:
                        gl_logits /= 2
                gl_loss = config.gf_weight * self.d_criterion(gl_logits, gla_labels)

        if config.gen_mode == "i2i":
            # gel_images = self.gen(lab_images)
            # gen_images = self.gen(unl_images)
            # image matching loss
            if config.im_weight > 0:
                im_loss = torch.mean(0 - torch.abs(gen_images - unl_images), 0)
                im_loss = config.im_weight * \
                          torch.mean(im_loss *
                                     Variable((im_loss.data > -0.1).type(torch.FloatTensor)).cuda())    # loss if dif_value too close to 0
            # image matching loss, cosine loss
            if config.cim_weight > 0:
                cim_loss = torch.mean(F.cosine_similarity(gen_images, unl_images), 0)
                cim_loss = config.cim_weight * \
                           torch.mean(cim_loss *
                                      Variable((cim_loss.data > 0.9).type(torch.FloatTensor)).cuda())    # loss if the direction too close
            # image lab loss
            if config.il_weight > 0:
                gel_images = self.gen(lab_images)
                if 'd' in config.dis_mode:
                    lab_logits = self.dis(gel_images)
                    il_loss += self.d_criterion(lab_logits, lab_labels)

                if 'm' in config.dis_mode:  # arcface
                    lab_logits = self.metric_fc(self.dis(gel_images, feat=True), lab_labels)
                    il_loss += self.m_criterion(lab_logits, lab_labels)
                il_loss *= config.il_weight

        if hasattr(self, 'gen'):
            # Feature matching loss, dis
            gen_feat = self.dis(gen_images, feat=True)
            if "z2i" in config.gen_mode:
                fm_loss = torch.mean(torch.abs(torch.mean(gen_feat, 0) - torch.mean(unl_feat, 0)))
            else:
                fm_loss = torch.mean(torch.abs(gen_feat - unl_feat))

            # # g1, g2 neg fm loss
            # # gg_loss = -1 * config.gg_weight * torch.mean(torch.abs(torch.mean(gel_feat, 0) - torch.mean(gen_feat, 0)))
            # gg_loss = torch.norm(gel_feat - gen_feat, 2) / gel_feat.nelement()  # t.norm: abs + norm
            # gg_loss = config.gg_margin - torch.mean(gg_loss)
            # gg_loss = config.gg_weight * torch.max(gg_loss, Variable(torch.zeros(gg_loss.size())).cuda())[0]

            # tv losss
            if config.tv_weight > 0:
                (_, c_x, h_x, w_x) = gen_images.size()
                # c_x = gen_images.size()[1]
                # h_x = gen_images.size()[2]
                # w_x = gen_images.size()[3]
                count_h = c_x * (h_x - 1) * w_x
                count_w = c_x * h_x * (w_x - 1)
                h_tv = torch.pow((gen_images[:, :, 1:, :] - gen_images[:, :, :-1, :]), 2).sum()
                w_tv = torch.pow((gen_images[:, :, :, 1:] - gen_images[:, :, :, :-1]), 2).sum()
                tv_loss = config.tv_weight * (h_tv / count_h + w_tv / count_w) / config.train_batch_size

            if config.st_weight > 0:
                gen_gram = self.gram_matrix(gen_images)
                unl_gram = self.gram_matrix(unl_images)
                st_loss += config.st_weight * nn.MSELoss()(gen_gram, unl_gram)

            # Generator loss
            g_loss = fm_loss + tv_loss + st_loss + \
                     vi_loss + gf_loss + gl_loss + \
                     im_loss + cim_loss + il_loss   # + gg_loss

            self.gen_optimizer.zero_grad()
            if config.gen_mode == "z2i": self.enc_optimizer.zero_grad()
            g_loss.backward()
            self.gen_optimizer.step()
            if config.gen_mode == "z2i": self.enc_optimizer.step()

        monitor_dict = OrderedDict()
        monitor_dict['unl acc'] = unl_acc.data[0]
        if gen_images is not None: monitor_dict['gen acc'] = gen_acc.data[0]
        monitor_dict['max unl acc'] = max_unl_acc.data[0]
        if gen_images is not None: monitor_dict['max gen acc'] = max_gen_acc.data[0]
        monitor_dict['lab loss'] = lab_loss.data[0]
        monitor_dict['unl loss'] = unl_loss.data[0]
        if config.da_weight > 0: monitor_dict['arc loss'] = arc_loss.data[0]
        if config.dgl_weight > 0: monitor_dict['dgl loss'] = dgl_loss.data[0]
        if config.con_coef > 0: monitor_dict['con loss'] = cons_loss.data[0]
        if config.nei_coef > 0: monitor_dict['nei loss'] = nei_loss.data[0]
        if config.dis_double:
            monitor_dict['la2 loss'] = lab_loss2.data[0]
            if config.dt_weight > 0: monitor_dict['tri loss'] = tri_loss.data[0]
            if config.ut_weight > 0: monitor_dict['ult loss'] = ult_loss.data[0]
        if hasattr(self, 'gen'):
            monitor_dict['fm loss'] = fm_loss.data[0]
            if config.tv_weight > 0: monitor_dict['tv loss'] = tv_loss.data[0]
            if config.st_weight > 0: monitor_dict['st loss'] = st_loss.data[0]
        if config.gen_mode == "z2i":
            monitor_dict['vi loss'] = vi_loss.data[0]
            if config.gf_weight > 0: monitor_dict['gf loss'] = gf_loss.data[0]
            if config.gl_weight > 0: monitor_dict['gl loss'] = gl_loss.data[0]
        if config.gen_mode == "i2i":
            if config.im_weight > 0: monitor_dict['im loss'] = im_loss.data[0]
            if config.cim_weight > 0: monitor_dict['cim loss'] = cim_loss.data[0]
            if config.il_weight > 0: monitor_dict['il loss'] = il_loss.data[0]

        return monitor_dict

    def eval_true_fake(self, data_loader, max_batch=None):
        self.gen.eval()
        self.dis.eval()
        # if not 'd' in self.config.dis_mode:
        #     self.metric_fc.eval()
        # self.enc.eval()

        cnt = 0
        unl_acc, gen_acc, max_unl_acc, max_gen_acc = 0., 0., 0., 0.
        for i, (images, _) in enumerate(data_loader.get_iter()):
            images = Variable(images.cuda(), volatile=True)
            if self.config.gen_mode == "z2i":
                noise = Variable(torch.Tensor(images.size(0), self.config.noise_size).uniform_().cuda(), volatile=True)
                gen_feat = self.dis(self.gen(noise), feat=True)
            elif self.config.gen_mode == "i2i":
                gen_feat = self.dis(self.gen(images), feat=True)
            unl_feat = self.dis(images, feat=True)


            unl_logits = self.dis.out_net(unl_feat)
            gen_logits = self.dis.out_net(gen_feat)

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
            dis_out = dis.out_net
        elif tri == 2:
            dis_out = self.dis_dou.out_net3
        else:   # 1
            dis_out = self.dis_dou.out_net2
        # self.gen.eval()
        dis.eval()
        # self.enc.eval()

        loss, incorrect, cnt = 0, 0, 0
        for i, (images, labels) in enumerate(data_loader.get_iter()):
            images = Variable(images.cuda(), volatile=True)
            labels = Variable(labels.cuda(), volatile=True)
            feat = dis(images, feat=True)
            pred_prob = dis_out(feat)
            loss += self.d_criterion(pred_prob, labels).data[0]
            cnt += 1
            incorrect += torch.ne(torch.max(pred_prob, 1)[1], labels).data.sum()
            if max_batch is not None and i >= max_batch - 1: break

        return loss / cnt, incorrect

    def visualize(self, data_loader=None):
        self.gen.eval()
        # self.dis.eval()
        # self.enc.eval()

        vis_size = 100
        if self.config.gen_mode == "z2i":
            noise = Variable(torch.Tensor(vis_size, self.config.noise_size).uniform_().cuda())
            gen_images = self.gen(noise)
        elif self.config.gen_mode == "i2i":
            gen_images = None
            for i, (images, _) in enumerate(data_loader.get_iter()):
                if i * self.config.dev_batch_size >= vis_size:
                    break
                images = Variable(images.cuda(), volatile=True)
                gen_image = self.gen(images)
                if i == 0:
                    gen_images = gen_image
                else:
                    gen_images = torch.cat((gen_images, gen_image), 0)

        save_path = os.path.join(self.config.save_dir,
                                 '{}.FM+VI.{}.png'.format(self.config.dataset, self.config.suffix))
        vutils.save_image(gen_images.data.cpu(), save_path, normalize=True, range=(-1, 1), nrow=10)

    def param_init(self):
        def func_gen(flag):
            def func(m):
                if hasattr(m, 'init_mode'):
                    setattr(m, 'init_mode', flag)

            return func

        images = []
        if self.config.double_input_size:
            num_img = 125
        else:
            num_img = 500

        for i in range(num_img / self.config.train_batch_size):
            lab_images, _ = self.labeled_loader.next()
            images.append(lab_images)
        images = torch.cat(images, 0)

        if hasattr(self, 'gen'):
            if self.config.gen_mode == "z2i":
                noise = Variable(torch.Tensor(images.size(0), self.config.noise_size).uniform_().cuda())
            else:
                noise = Variable(torch.Tensor(images.size()).uniform_().cuda())
                self.gen._initialize_weights()
            self.gen.apply(func_gen(True))
            gen_images = self.gen(noise)
            self.gen.apply(func_gen(False))

        if hasattr(self, 'enc'):
            self.enc.apply(func_gen(True))
            self.enc(gen_images)
            self.enc.apply(func_gen(False))

        self.dis.apply(func_gen(True))
        if self.config.dis_double: self.dis_dou.apply(func_gen(True))
        feat = self.dis(Variable(images.cuda()), feat=True)
        logits = self.dis.out_net(feat)
        if self.config.dis_double: logits = self.dis_dou(feat)
        if self.config.dis_triple: logits = self.dis_dou.out_net3(feat)
        self.dis.apply(func_gen(False))
        if self.config.dis_double: self.dis_dou.apply(func_gen(False))

        self.ema_dis = copy.deepcopy(self.dis) # clone weight_scale and weight

    def calculate_remaining(self, t1, t2, epoch):  # ta
        progress = (epoch + 0.) / self.config.max_epochs
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
        time_str = '[{:8.2%}], {:3d}:{:2d}:{:2d}<{:3d}:{:2d}:{:2d} '.format(progress, ehr, emin, esec, rhr, rmin, rsec)

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

    def load_model(self, net, net_label, epo_label):  # ta
        load_filename = 'VI.{}_{}_net_{}.pth'.format(self.config.suffix, epo_label, net_label)
        load_path = os.path.join(self.config.save_dir, load_filename)
        load_net = torch.load(load_path)
        net.cpu()
        net.load_my_state_dict(load_net)

        if torch.cuda.is_available():
            net.cuda()

    def save(self, epo_label):  # ta
        # save new
        self.save_model(self.dis, 'D', epo_label)
        self.save_model(self.ema_dis, 'M', epo_label)
        if hasattr(self, 'gen'):
            self.save_model(self.gen, 'G', epo_label)
        if self.config.gen_mode == "z2i":
            self.save_model(self.enc, 'E', epo_label)
        if self.config.dis_double:
            self.save_model(self.dis_dou, 'D2', epo_label)
        # del old
        if epo_label >= self.config.eval_period:
            epo_label -= self.config.eval_period
            self.del_model('D', epo_label)
            self.del_model('M', epo_label)
            if hasattr(self, 'gen'):
                self.del_model('G', epo_label)
            if self.config.gen_mode == "z2i":
                self.del_model('E', epo_label)
            if self.config.dis_double:
                self.save_model('D2', epo_label)

    def resume(self, epo_label):  # ta
        # load old
        self.load_model(self.dis, 'D', epo_label)
        self.load_model(self.ema_dis, 'M', epo_label)
        if hasattr(self, 'gen'):
            self.load_model(self.gen, 'G', epo_label)
        if self.config.gen_mode == "z2i":
            self.load_model(self.enc, 'E', epo_label)
        if self.config.dis_double:
            self.load_model(self.enc, 'D2', epo_label)

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

    def change_dis_mode(self, now_epoch):
        if self.config.cd_mode_iter <= 0:
            return
        if now_epoch == self.config.cd_mode_iter:
            self.config.dis_mode = self.dis_mode.split(",")[1]
        pass

    def train(self):
        config = self.config
        batch_per_epoch = int((len(self.unlabeled_loader) + config.train_batch_size - 1) / config.train_batch_size)
        if not config.resume:
            self.param_init()
            self.iter_cnt = 0
            iter = 0
        else:
            self.iter_cnt = 0 + config.last_epochs
            iter = batch_per_epoch*(config.last_epochs-1)
            self.resume(iter)

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
        start_time = time.time()
        while True:
            if iter % batch_per_epoch == 0:
                epoch = iter / batch_per_epoch
                if not self.dg_flag and config.dg_start <= epoch:
                    self.dg_flag = True
                if epoch >= config.max_epochs:
                    # save model   # ta
                    self.save(iter)
                    break
                epoch_ratio = float(epoch) / float(config.max_epochs)
                # use another outer max to prevent any float computation precision problem
                self.dis_optimizer.param_groups[0]['lr'] = max(min_lr, config.dis_lr * min(3. * (1. - epoch_ratio), 1.))
                if hasattr(self, 'gen'):
                    self.gen_optimizer.param_groups[0]['lr'] = max(min_lr, config.gen_lr * min(3. * (1. - epoch_ratio), 1.))
                if config.gen_mode == "z2i":
                    self.enc_optimizer.param_groups[0]['lr'] = max(min_lr, config.enc_lr * min(3. * (1. - epoch_ratio), 1.))

            self.change_dis_mode(iter / batch_per_epoch)
            self.get_current_consistency_weight(iter / batch_per_epoch)
            self.adjust_learning_rate(self.dis_optimizer, config.dis_lr, config.ini_lr, iter / batch_per_epoch)
            iter_vals = self._train()
            self.update_ema_variables(self.config.ema_decay, iter, batch_per_epoch)

            if len(monitor.keys()) == 0:
                for k in iter_vals.keys():
                    monitor[k] = 0.
                    # if not monitor.has_key(k):
                    #     monitor[k] = 0.
            for k, v in iter_vals.items():
                monitor[k] += v

            if iter % config.vis_period == 0:
                if config.gen_mode == "z2i":
                    self.visualize()
                elif config.gen_mode == "i2i":
                    self.visualize(self.dev_loader)

            if iter % config.eval_period == 0:
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

                train_incorrect /= 1.0 * len(self.labeled_loader)
                dev_incorrect /= 1.0 * len(self.dev_loader)
                min_dev_incorrect = min(min_dev_incorrect, dev_incorrect)
                e_mdi = min(e_mdi, ema_result_)

                disp_str = '#{}\ttrain: {:.4f}, {:.4f} | dev: {:.4f}, {:.4f} | best: {:.4f}'.format(
                    iter, train_loss, train_incorrect, dev_loss, dev_incorrect, min_dev_incorrect)
                if isinstance(ema_result, tuple):
                    disp_str += ' | ema: {:.4f}, {:.4f}, {:.4f}'.format(ema_train_result_, ema_result_, e_mdi)
                else:
                    disp_str += ' | ema:   None ,   None'
                if config.dis_double:
                    disp_str += ' | tri: {:.4f}, {:.4f}'.format(tri_result1, tri_result2)

                for k, v in monitor.items():
                    disp_str += ' | {}: {:.4f}'.format(k, v / config.eval_period)

                if hasattr(self, 'gen'):
                    disp_str += ' | [Eval] unl acc: {:.4f}, gen acc: {:.4f}, max unl acc: {:.4f}, max gen acc: {:.4f}'.format(
                        unl_acc, gen_acc, max_unl_acc, max_gen_acc)
                disp_str += ' | lr: {:.5f}'.format(self.dis_optimizer.param_groups[0]['lr'])
                disp_str += '\n'

                monitor = OrderedDict()

                # timer   # ta
                time_str = self.calculate_remaining(start_time, time.time(), iter / batch_per_epoch)

                self.logger.write(disp_str)
                sys.stdout.write(disp_str)
                sys.stdout.write(time_str)  # ta
                sys.stdout.flush()

            iter += 1
            self.iter_cnt += 1


if __name__ == '__main__':
    cc = config.cifarmg_config()
    parser = argparse.ArgumentParser(description='cifarmg_trainer.py')
    parser.add_argument('-suffix', default='mg0', type=str, help="Suffix added to the save images.")
    parser.add_argument('-r', dest='resume', action='store_true')
    parser.add_argument('-max_epochs', default=cc.max_epochs, type=int,
                        help="max epoches")
    parser.add_argument('-last_epochs', default=cc.last_epochs, type=int,
                        help="last epochs")
    parser.add_argument('-noise_size', default=cc.noise_size, type=int,
                        help="gen noise size")
    parser.add_argument('-dg_start', default=cc.dg_start, type=int,
                        help="start dis loss epoch")
    parser.add_argument('-eval_period', default=cc.eval_period, type=int,
                        help="evaluate period, -1: per-epoch")
    parser.add_argument('-vis_period', default=cc.vis_period, type=int,
                        help="visualize period, -1: per-epoch")
    parser.add_argument('-ld', '--size_labeled_data', default=cc.size_labeled_data, type=int,
                        help="labeled data num")
    parser.add_argument('-train_batch_size', default=cc.train_batch_size, type=int,
                        help="labeled batch size")
    parser.add_argument('-train_batch_size_2', default=cc.train_batch_size_2, type=int,
                        help="unlabeled batch size")
    parser.add_argument('-dis_lr', default=cc.dis_lr, type=float,
                        help="discriminator learn rate")
    parser.add_argument('-gen_lr', default=cc.gen_lr, type=float,
                        help="generator learn rate")
    parser.add_argument('-con_coef', default=cc.con_coef, type=float,
                        help="Consistency loss content")
    parser.add_argument('-nei_coef', default=cc.nei_coef, type=float,
                        help="neighbor loss content")
    parser.add_argument('-nei_margin', default=cc.nei_margin, type=float,
                        help="neighbor margin content")
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
    parser.add_argument('-dgl_weight', default=cc.dgl_weight, type=float,
                        help="dis gen lab loss content")
    parser.add_argument('-da_weight', default=cc.da_weight, type=float,
                        help="dis arcface loss content")
    parser.add_argument('-dt_weight', default=cc.dt_weight, type=float,
                        help="dis triple loss content")
    parser.add_argument('-ut_weight', default=cc.ut_weight, type=float,
                        help="dis triple gan loss content")
    parser.add_argument('-tv_weight', default=cc.tv_weight, type=float,
                        help="tv loss weight")
    parser.add_argument('-st_weight', default=cc.st_weight, type=float,
                        help="style loss weight")
    parser.add_argument('-gf_weight', default=cc.gf_weight, type=float,
                        help="z2i, gl loss content")
    parser.add_argument('-gl_weight', default=cc.gl_weight, type=float,
                        help="z2i, gen z lab loss content")
    parser.add_argument('-im_weight', default=cc.im_weight, type=float,
                        help="i2i, image matching loss weight")
    parser.add_argument('-cim_weight', default=cc.cim_weight, type=float,
                        help="i2i, cosine image matching loss weight")
    parser.add_argument('-il_weight', default=cc.il_weight, type=float,
                        help="i2i, image lab loss weight")
    parser.add_argument('-gop', default=cc.gop, type=str,
                        help="gen optim: Adam, SGD")
    parser.add_argument('-gen_mode', default=cc.gen_mode, type=str,
                        help="gen model mode: z2i, i2i, non")
    parser.add_argument('-dis_mode', default=cc.dis_mode, type=str,
                        help="dis model mode: d, m, dm; 'd,dm'")
    parser.add_argument('-cd_mode_iter', default=cc.cd_mode_iter, type=int,
                        help="change dis mode")
    parser.add_argument('-d', dest='double_input_size', action='store_true')
    parser.add_argument('-f', dest='flip', action='store_true')
    parser.add_argument('-dd', dest='dis_double', action='store_true')
    parser.add_argument('-dt', dest='dis_triple', action='store_true')
    parser.set_defaults(resume=False)
    parser.set_defaults(dis_double=False)
    parser.set_defaults(dis_triple=False)
    parser.set_defaults(double_input_size=cc.double_input_size)
    parser.set_defaults(flip=cc.flip)
    args = parser.parse_args()

    trainer = Trainer(cc, args)
    trainer.train()
