import torch
import torch.nn as nn
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
# from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
# from PIL import Image

from utils import *
# from cifarmg_trainer import Trainer
import losses, ramps
import copy
# from metrics import ArcMarginProduct    # cosine distance
from losses import FocalLoss
# import random
import pdb
# pdb.set_trace()
# pdb.set_trace = lambda: None

class Tester(object):
    def __init__(self, config, args):
        self.config = config
        for k, v in args.__dict__.items():
            setattr(self.config, k, v)
        setattr(self.config, 'save_dir', '{}_log'.format(self.config.dataset))

        assert os.path.exists(self.config.save_dir), "There is no log folder"

        # try to load config
        log_path = os.path.join(self.config.save_dir, '{}.FM+VI.{}.txt'.format(self.config.dataset, self.config.suffix))
        if os.path.isfile(log_path):
            print("log config covering...")
        logger = open(log_path, 'r')

        keep_str = ['dis_channels']
        keep_val = ['last_epochs', 'last_epo_lbl']
        for li in logger:
            if "|" in li: break
            key, val = li.split(" : ")
            val = val[:-1]  # abort \n
            if key in keep_val:
                continue
            elif key not in keep_str:
                val = self.int_float_bool(val)
            setattr(self.config, key, val)
        logger.close()

        # self.labeled_loader, self.unlabeled_loader, self.dev_loader, self.special_set = data.get_cifar_loaders_test(
        #     config)
        self.loaders = data.get_data_loaders_test(config)
        if config.mu:
            self.label_list = range(config.num_label)
            in_channels = [int(i) for i in config.dis_channels.split(",")]
            self.dis = model.Unet_Discriminator(config, in_channels=in_channels).cuda()
            self.ema_dis = model.Unet_Discriminator(config, in_channels=in_channels).cuda()
        else:
            self.dis = model.Discriminative(config).cuda()
            self.ema_dis = model.Discriminative(config).cuda()  # , ema=True).cuda()

        if config.mu:
            self.gen = nn.ModuleList()
            self.gen.append(model.UNetWithResnetEncoder(n_classes=3, res=config.gen_mode).cuda())
            for i in range(config.num_label-1):
                self.gen.append(model.ResnetDecoder_skip(n_classes=3, res=config.gen_mode).cuda())
        elif hasattr(self.config, 'gen_mode') and self.config.gen_mode != "non":
            self.gen = model.generator(image_side=config.image_side,
                                       noise_size=config.noise_size,
                                       large=config.double_input_size,
                                       gen_mode=config.gen_mode).cuda()

        dis_para = [{'params': self.dis.parameters()}, ]
        if 'm' in config.dis_mode:  # svhn: 168; cifar:192
            self.m_criterion = FocalLoss(gamma=2)

        if config.dis_double:
            self.dis_dou = model.Discriminative_out(config).cuda()
            dis_para.append({'params': self.dis_dou.parameters()})

        if config.gen_mode == "z2i":
            self.enc = model.Encoder(config.image_side, noise_size=config.noise_size, output_params=True).cuda()

        self.d_criterion = nn.CrossEntropyLoss()
        if config.consistency_type == 'mse':
            self.consistency_criterion = losses.softmax_mse_loss  # F.MSELoss()    # (size_average=False)
        elif config.consistency_type == 'kl':
            self.consistency_criterion = losses.softmax_kl_loss  # nn.KLDivLoss()  # (size_average=False)
        else:
            pass
        self.consistency_weight = 0

        disp_str = ''
        for attr in sorted(dir(self.config), key=lambda x: len(x)):
            if not attr.startswith('__'):
                disp_str += '{} : {}\n'.format(attr, getattr(self.config, attr))
        sys.stdout.write(disp_str)
        sys.stdout.flush()

        # for arcface
        self.s = 30.0
        m = 0.50
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        # for tsne
        self.gen_feat = None
        self.p_fs = tuple([ int(i) for i in self.config.p_fs.split(",")])
        # for mu
        self.img_per_cls = None


    def int_float_bool(self, value):
        try:
            value = int(value)
        except:
            pass
        if isinstance(value, str):
            try:
                value = float(value)
            except:
                pass
        if isinstance(value, str):
            if "False" in value:
                value = False
            elif "true" in value:
                value = True
        return value

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

    def get_feat(self, images, mu_layers=5, ema=False):
        if self.config.mu:
            return self.gen[0](images, encode=True)
        else:
            if ema:
                return self.ema_dis(images, feat=True)
            return self.dis(images, feat=True)

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
                gen_feat = self.get_feat(self.gen(noise))
            elif self.config.gen_mode == "i2i":
                gen_feat = self.get_feat(self.gen(images))
            else:
                gen_feat = self.get_feat(self.gen[i%self.config.num_label](images))

            unl_feat = self.get_feat(images)

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
        else:  # 1
            dis_out = self.dis_dou.out_net2
        self.gen.eval()
        dis.eval()
        # self.enc.eval()

        loss, incorrect, cnt = 0, 0, 0
        for i, (images, labels) in enumerate(data_loader.get_iter()):
            images = Variable(images.cuda(), volatile=True)
            labels = Variable(labels.cuda(), volatile=True)
            feat = self.get_feat(images, ema=ema)
            pred_prob = dis_out(feat)
            loss += self.d_criterion(pred_prob, labels).data[0]
            cnt += 1
            incorrect += torch.ne(torch.max(pred_prob, 1)[1], labels).data.sum()
            if max_batch is not None and i >= max_batch - 1: break

        return loss / cnt, incorrect

    def visualize_iter(self, data_loader=None, bsize=400, iters=1, data_suffix="g", gzlab=-1):
        assert data_loader is not None or "g" in data_suffix, "g or loader"
        iter_num = lambda a, b: int((len(a) + b - 1) // b)
        if data_loader is not None and ("g" not in data_suffix or iters == -1):
            iters = iter_num(data_loader, bsize)
        elif iters != -1 and self.config.declbl and self.config.mu and not self.config.nstf:
            iters = iter_num(data_loader, bsize)

        nrow = int(bsize ** 0.5)
        start_time = time.time()
        first_str = "{} iters: {}".format(data_suffix, iters)
        for i in range(iters):
            data_suffix2 = data_suffix + "{:03d}".format(i)
            self.visualize(data_loader, vis_size=bsize, data_suffix=data_suffix2, nrow=nrow, gzlab=gzlab)
            time_str = self.calculate_remaining(start_time, time.time(), float(i) / iters)
            sys.stdout.write('\r{}{}'.format(first_str, time_str))  # ta
            sys.stdout.flush()
        sys.stdout.write('\n')
        sys.stdout.flush()

    def visualize(self, data_loader=None, vis_size=100, data_suffix="g", nrow=10, gzlab=-1):
        self.gen.eval()
        self.dis.eval()
        # vis_size = 100
        lab = None
        if self.config.gen_mode == "z2i" and data_loader is None:
            if gzlab > 0:
                lab = torch.Tensor(range(gzlab))
                lab = lab.unsqueeze(1).repeat(1, vis_size / gzlab).view(-1).long()
                labels_oh = Variable(torch.zeros(vis_size, gzlab).scatter_(1, lab.unsqueeze(1), 1).cuda())
                noise = Variable(torch.Tensor(vis_size, self.config.noise_size - gzlab).uniform_().cuda())
                noise = torch.cat((labels_oh, noise), dim=1)
            else:
                noise = Variable(torch.Tensor(vis_size, self.config.noise_size).uniform_().cuda())
            gen_images = self.gen(noise)
        elif self.config.gen_mode == "i2i":
            gen_images = []
            cnt = 0
            while (True):
                images, _ = data_loader.next()
                images = Variable(images.cuda(), volatile=True)
                gen_image = self.gen(images)
                gen_images.append(gen_image)
                cnt += data_loader.batch_size
                if cnt + data_loader.batch_size > vis_size:
                    break
            gen_images = torch.cat(gen_images, 0)
        elif "g" not in data_suffix and data_loader is not None:    # just image
            gen_images = None
            i, cnt = 0, 0
            while (True):
                images, labs = data_loader.next()
                images = Variable(images.cuda(), volatile=True)
                if i == 0:
                    gen_images = images
                    lab = labs
                else:
                    gen_images = torch.cat((gen_images, images), 0)
                    lab = torch.cat((lab, labs), 0)
                i += 1
                cnt += data_loader.batch_size
                if cnt + data_loader.batch_size > vis_size:
                    break
        elif self.config.declbl and self.config.mu and not self.config.tsne \
                and not self.config.nsg and self.config.nstf: # category-wised img in a row
            ori_images = []
            gen_images = []
            for i in range(self.config.num_label):
                ori_images.append([])
                gen_images.append([])
            while (True):
                cnt = 0
                images, labs = data_loader.next()
                for i in range(self.config.num_label):
                    i_count = sum(labs == i)
                    if i_count == 0 and len(ori_images[i]) == 0:
                        cnt += 1
                    if i_count == 0 or len(ori_images[i]) > 0:
                        continue
                    inds = (labs == i).nonzero().squeeze()
                    ori_images[i] = images[inds,:,:,:][0].unsqueeze(0)
                if cnt == 0:
                    break
            ori_images = torch.cat(ori_images, 0)
            ori_images = Variable(ori_images.cuda(), volatile=True)
            inp_feat = self.gen[0](ori_images, skip_encode=True)
            for i in range(self.config.num_label):
                gen_image = self.gen[i].decode(inp_feat)
                gen_images[i].append(gen_image)
            for i in range(self.config.num_label):
                gen_images[i] = torch.cat(gen_images[i], 0) # .squeeze(0).transpose(0, 1)
            o_size = ori_images.size()
            gen_images = torch.stack(gen_images, 0).transpose(0, 1).contiguous().view(-1, o_size[1], o_size[2], o_size[3])
            # gen_images = torch.cat(gen_images, 0)
        elif self.config.declbl and self.config.mu:
            ori_images = []
            gen_images = []
            for i in range(self.config.num_label):
                ori_images.append([])
                gen_images.append([])
            cnt = 0
            img_per_cls = np.zeros(self.config.num_label, dtype=int)
            while (True):
                images, labs = data_loader.next()
                images = Variable(images.cuda(), volatile=True)
                inp_feat = self.gen[0](images, skip_encode=True)
                for i in range(self.config.num_label):
                    i_count = sum(labs == i)
                    if i_count == 0:
                        continue
                    img_per_cls[i] += i_count
                    inds = (labs == i).nonzero().squeeze().cuda()
                    i_feat = dict()
                    for j in inp_feat.keys():
                        i_feat[j] = inp_feat[j][inds,:,:,:]
                    gen_image = self.gen[i].decode(i_feat)
                    ori_images[i].append(images[inds,:,:,:])
                    gen_images[i].append(gen_image)
                cnt += data_loader.batch_size
                if cnt + data_loader.batch_size > vis_size:
                    break
            for i in range(self.config.num_label):
                if len(gen_images[i]) != 0:
                    ori_images[i] = torch.cat(ori_images[i], 0)
                    gen_images[i] = torch.cat(gen_images[i], 0)
        elif self.config.mu:   # mu
            ori_images = []
            gen_images = []
            cnt = 0
            img_per_cls = data_loader.batch_size    # // self.config.num_label
            while (True):
                images, _ = data_loader.next()
                images = Variable(images.cuda(), volatile=True)
                ori_images.append(images)
                inp_feat = self.gen[0](images, skip_encode=True)    # , [range(i*img_per_cls, (i+1)*img_per_cls)], skip_encode=True)
                for i in range(self.config.num_label):
                    gen_image = self.gen[i].decode(inp_feat)
                    gen_images.append(gen_image)
                cnt += img_per_cls * self.config.num_label
                if cnt + img_per_cls * self.config.num_label > vis_size:
                    break
            ori_images = torch.cat(ori_images, 0)
            gen_images = torch.cat(gen_images, 0)

        # for tsne
        if "g" in data_suffix and self.config.tsne:
            if self.config.mu and self.config.declbl:   # may diff # every cls
                if self.gen_feat is None:
                    self.gen_feat = [None] * self.config.num_label
                for i in range(self.config.num_label):
                    if img_per_cls[i] != 0:
                        feat = self.get_feat(gen_images[i]).data
                        if self.gen_feat[i] is None:
                            self.gen_feat[i] = feat
                        else:
                            self.gen_feat[i] = torch.cat((self.gen_feat[i], feat), dim=0)
                if self.img_per_cls is None:
                    self.img_per_cls = img_per_cls
                else:
                    self.img_per_cls += img_per_cls
            else:
                feat = self.get_feat(gen_images).data
                if self.config.mu and self.img_per_cls is None:
                    self.img_per_cls = img_per_cls
                if self.gen_feat is None:
                    self.gen_feat = feat
                else:
                    self.gen_feat = torch.cat((self.gen_feat, feat), dim=0)
        if self.config.nsg:
            return
        if type(gen_images) == list:
            ori_images = torch.cat(ori_images, 0)
            gen_images = torch.cat(gen_images, 0)

        if self.config.declbl and self.config.mu and not self.config.tsne \
                and not self.config.nsg and self.config.nstf:
            nrow = gen_images.size(0) // self.config.num_label
        save_path = os.path.join(self.config.save_dir,
                                 'Te{}.FM+VI.{}.{}.png'.format(self.config.dataset, self.config.suffix, data_suffix))
        vutils.save_image(gen_images.data.cpu(), save_path, normalize=True, range=(-1, 1), nrow=nrow)
        if "g" in data_suffix and data_loader is not None:
            save_path = os.path.join(self.config.save_dir,
                                     'Te{}.FM+VI.{}.{}_ori.png'.format(self.config.dataset, self.config.suffix, data_suffix))
            vutils.save_image(ori_images.data.cpu(), save_path, normalize=True, range=(-1, 1), nrow=1)
        if self.config.nstf:
            return
        # dis true img
        gen_logits = self.dis.out_net(self.get_feat(gen_images))
        # pdb.set_trace()
        # self.visualize_accs(gen_images, gen_logits, data_suffix)
        # gen_images = Variable(torch.Tensor([batch, 3, 32, 32])).cuda
        # gen_logits = Variable(torch.Tensor([batch])).cuda
        gen_logsumexp = log_sum_exp(gen_logits)
        acc = nn.functional.sigmoid(gen_logsumexp).gt(0.5).data.long()
        max_acc = gen_logits.max(1)[0].detach().gt(0.0).data.long()
        # acc_images[1-acc,:,:,:] *= 0
        acc_ind = acc.unsqueeze(1).repeat(1, gen_images.nelement() /
                                          gen_images.size(0)).view(gen_images.size())
        acc_ind = acc_ind.float()
        err_ind = 1 - acc_ind
        if acc_ind.sum() > 0:
            acc_images = gen_images.clone().data  # acc img
            acc_images -= 2 * err_ind
            save_path = os.path.join(self.config.save_dir,
                                     'Te{}.FM+VI.{}.{}ac.png'.format(self.config.dataset, self.config.suffix,
                                                                     data_suffix))
            vutils.save_image(acc_images.cpu(), save_path, normalize=True, range=(-1, 1), nrow=nrow)
        # gen_err
        if err_ind.sum() > 0:
            err_images = gen_images.clone().data
            # err_images[acc,:,:,:] *= 0
            err_images -= 2 * acc_ind
            save_path = os.path.join(self.config.save_dir,
                                     'Te{}.FM+VI.{}.{}er.png'.format(self.config.dataset, self.config.suffix,
                                                                     data_suffix))
            vutils.save_image(err_images.cpu(), save_path, normalize=True, range=(-1, 1), nrow=nrow)
        # acc_images[1-max_acc,:,:,:] *= 0
        acc_ind = max_acc.unsqueeze(1).repeat(1, gen_images.nelement() /
                                              gen_images.size(0)).view(gen_images.size())
        acc_ind = acc_ind.float()
        err_ind = 1 - acc_ind
        if acc_ind.sum() > 0:
            acc_images = gen_images.clone().data  # max_acc img
            acc_images -= 2 * err_ind
            save_path = os.path.join(self.config.save_dir,
                                     'Te{}.FM+VI.{}.{}mac.png'.format(self.config.dataset, self.config.suffix,
                                                                      data_suffix))
            vutils.save_image(acc_images.cpu(), save_path, normalize=True, range=(-1, 1), nrow=nrow)
        if err_ind.sum() > 0:
            # max_gen_err
            err_images = gen_images.clone().data
            # err_images[max_acc,:,:,:] *= 0
            err_images -= 2 * acc_ind
            save_path = os.path.join(self.config.save_dir,
                                     'Te{}.FM+VI.{}.{}mer.png'.format(self.config.dataset, self.config.suffix,
                                                                      data_suffix))
            vutils.save_image(err_images.cpu(), save_path, normalize=True, range=(-1, 1), nrow=nrow)
        # record report
        save_path = os.path.join(self.config.save_dir,
                                 'Te{}.FM+VI.{}.{}.txt'.format(self.config.dataset, self.config.suffix, data_suffix))
        save_str = ""
        topk = self.config.num_label
        val, ind = torch.topk(gen_logits, topk)
        val, ind = val.data.cpu().numpy(), ind.data.cpu().numpy()
        acc, max_acc = acc.cpu().numpy(), max_acc.cpu().numpy()
        save_str += "sum{}_m{}/{} ".format(acc.sum(), max_acc.sum(), acc.shape[0])
        if lab is not None:  # real data should have label
            lab = lab.numpy()
            acc_str_row = lambda a: "{}_m{} {}: ".format(acc[a], max_acc[a], lab[a])
            pred_accumu = lab[ind[:, 0] == lab]
            lab_accumu = lab
        else:
            acc_str_row = lambda a: "{}_m{}: ".format(acc[a], max_acc[a])
            pred_accumu = ind[max_acc * acc > 0, 0]
            lab_accumu = ind[:, 0]
        # accumulate all labels
        for i in range(gen_logits.size(1)):
            save_str += "{}({}/{}) ".format(i, np.sum(pred_accumu == i), np.sum(lab_accumu == i))
        save_str += "\n"
        for i in range(vis_size):
            save_str += acc_str_row(i)
            for j in range(topk):
                save_str += "{}({}) ".format(ind[i, j], val[i, j])
            save_str += "\n"
        logger = open(save_path, 'wb')
        logger.write(save_str)

    def tsne(self, data_suffix="u"):
        # g check
        if not hasattr(self.config, 'gen_mode') or self.config.gen_mode == "non":
            data_suffix = data_suffix.replace("g", "")
        # value check
        if data_suffix == "":
            print("No suffix!")
            return
        # get real feature
        cifar_feats = lbls = None
        u_lbls = g_lab = None

        if "u" in data_suffix:
            cifar_feats, u_lbls = self.eval_feats(self.loaders[0], data="u")
            lbls = u_lbls.copy()
            # if self.config.te:
            #     cifar_feats = cifar_feats[:1000]
            #     lbls = lbls[:1000]
        if "l" in data_suffix :
            if not len(self.loaders) > 2:
                print("You cannot plot {}".format(data_suffix))
                return
            feats, l_lbls = self.eval_feats(self.loaders[2], data="l")
            if cifar_feats is None:
                cifar_feats = feats
                lbls = l_lbls
            else:
                pass
                # cifar_feats = np.concatenate((cifar_feats, feats), axis=0)
                # lbls = np.concatenate((lbls, l_lbls), axis=0)
        if "d" in data_suffix:
            feats, d_lbls = self.eval_feats(self.loaders[1], data="d")
            if cifar_feats is None:
                cifar_feats = feats
                lbls = d_lbls
            else:
                cifar_feats = np.concatenate((cifar_feats, feats), axis=0)
                lbls = np.concatenate((lbls, d_lbls), axis=0)

        num_label = self.config.num_label
        g_offset = 20 if self.config.dataset == "coil20" else 10
        # get fake feature
        if "g" in data_suffix:
            if self.config.mu and self.config.declbl:
                num_label += self.config.num_label
                g_lab = []
                for i in range(self.config.num_label):
                    g_lab.append(np.array([g_offset+i]*self.img_per_cls[i]))
                g_lab = np.concatenate(g_lab)

            elif self.config.mu:
                num_label += self.config.num_label
                iter_num = self.gen_feat.shape[0] / (self.img_per_cls * self.config.num_label)
                g_lab = np.tile(np.arange(g_offset, g_offset+self.config.num_label).repeat(self.img_per_cls), iter_num)
            else:
                num_label += 1
                g_lab = np.array([g_offset]).repeat(self.gen_feat.shape[0])

            if cifar_feats is None:
                cifar_feats = self.gen_feat
                lbls = g_lab
            else:
                cifar_feats = np.concatenate((cifar_feats, self.gen_feat), axis=0)
                lbls = np.concatenate((lbls, g_lab), axis=0)


        data_num = [u_lbls.shape[0],  g_lab.shape[0]] \
            if "g" in data_suffix and "u" in data_suffix else None
        self.plot_scatter(cifar_feats, lbls, data_suffix, data_num)

        if not self.config.banl == "":
            # pdb.set_trace()
            data_suffix2 = data_suffix+"_" + self.config.banl
            banl = str(self.config.banl)
            ban_e = [int(i) for i in banl.split(",")]
            ban_ind = np.array([], dtype=int)
            for i in ban_e:
                ban_ind = np.concatenate((ban_ind, np.where(lbls == i)[0]), axis=0)
                num_label -= 1
                if self.config.mu:
                    ban_ind = np.concatenate((ban_ind, np.where(lbls == g_offset+i)[0]), axis=0)
                    num_label -= 1
            # ban_ind.sort()
            mask = np.ones(len(lbls), dtype=bool)
            mask[ban_ind] = False
            lbls2 = lbls[mask]
            cifar_feats2 = cifar_feats[mask]
            if self.config.mu:
                num_ul = num_label // 2
            else:
                num_ul = num_label - 1
            data_num = [(u_lbls.shape[0]//self.config.num_label) * num_ul,  g_lab.shape[0]] \
                if "g" in data_suffix and "u" in data_suffix else None
            self.plot_scatter(cifar_feats2, lbls2, data_suffix2, data_num)

        if not self.config.showlab == "":
            data_num = [u_lbls.shape[0]//self.config.num_label,  g_lab.shape[0]] \
                if "g" in data_suffix and "u" in data_suffix else None
            for i in self.config.showlab.split(","):
                data_suffix2 = data_suffix+"_" + i
                show_e = int(i)
                mask = np.where(lbls == show_e)[0]
                mask = np.concatenate((mask, np.where(lbls == g_offset)[0]), axis=0)
                # ban_ind.sort()
                lbls2 = lbls[mask]
                cifar_feats2 = cifar_feats[mask]
                self.plot_scatter(cifar_feats2, lbls2, data_suffix2, data_num)

    def plot_scatter(self, feats, y_dist, data_suffix, data_num=None):
        # x_dist = ts_feat
        # y_dist = lbls
        print("Plot tsne {}".format(data_suffix))
        splitul = False
        if "u" in data_suffix and "l" in data_suffix:
            print("Also Plot tsne {}2".format(data_suffix))
            assert data_num is not None, "data_num is None"
            splitul = True
        if self.config.te:
            print("TSNE point num: {}".format(len(y_dist)))
        x_dist = TSNE(n_components=2).fit_transform(feats)
        x_dist *= self.config.p_scale
        if self.config.te:
            print("after TSNE transform")
        # plot
        plt.ioff()
        # fig = plt.figure(figsize=self.p_fs, dpi=self.config.p_d)
        # fig.add_subplot(111)
        fig, ax = plt.subplots(figsize=self.p_fs, dpi=self.config.p_d)
        if self.config.num_label <= 10:
            colors = {-1: 'lightblue',
                       0: '#FF7070',  1: '#FFAA70',  2: '#FFEB62',  3: '#C1FF62',  4: '#68FFAF',
                       5: '#68FFFF',  6: '#76BBFF',  7: '#767FFF',  8: '#A476FF',  9: '#FF76FF',
                      10: '#D20000', 11: '#C95000', 12: '#C9AE00', 13: '#78C900', 14: '#00C95E',
                      15: '#03ACAC', 16: '#145FAB', 17: '#353EB9', 18: '#6134B9', 19: '#CA46CA'}
                      #  0: 'salmon',  1: 'yellow',  2: 'lime',  3: 'orange',  4: 'dodgerblue',
                      #  5: 'skyblue',  6: 'violet',  7: 'cyan',  8: 'pink',  9: 'palegreen',
                      # 10: 'darkred', 11: 'tan', 12: 'limegreen', 13: 'darkorange', 14: 'steelblue',
                      # 15: 'royalblue', 16: 'darkviolet', 17: 'darkcyan', 18: 'deeppink', 19: 'darkseagreen'}
        else:
            colors = {-1: 'lightblue',
                       0: '#FFA680',  1: '#FFC980',  2: '#FFED80',  3: '#EFFF80',  4: '#CBFF80',  5: '#A6FF80',  6: '#82FF80',  7: '#80FFA2',  8: '#80FFC6',  9: '#80FFEB',
                      10: '#80F1FF', 11: '#80CDFF', 12: '#80A9FF', 13: '#8084FF', 14: '#A080FF', 15: '#C480FF', 16: '#E980FF', 17: '#FF80F3', 18: '#FF80CF', 19: '#FF80AB',
                      20: '#C00000', 21: '#C03600', 22: '#C06D00', 23: '#C0A300', 24: '#A6C000', 25: '#70C000', 26: '#3AC000', 27: '#03C000', 28: '#00C033', 29: '#00C06A',
                      30: '#00C0A0', 31: '#00AAC0', 32: '#0073C0', 33: '#003DC0', 34: '#0006C0', 35: '#3000C0', 36: '#6600C0', 37: '#9D00C0', 38: '#C000AD', 39: '#C00076'}

        unique_lbl = np.array(sorted(np.unique(y_dist)))
        num_gls = len(colors)-1 # lbl<10: 21-1; el: 41-1
        for i in range(-1, num_gls): # remove labels which are outside y_dist
            if i not in unique_lbl:
                colors.pop(i)
        # remap key and label, start_label: ?; last lbl: num_gls
        num_lbl = len(unique_lbl)
        if self.config.mu:
            g_min_lbl = num_gls - self.config.num_label + 1
        else:
            g_min_lbl = num_gls
        if self.config.te:
            print("unique_lbl: {}".format(unique_lbl))
            print("num: {}; g_num: {}; gml: {}".format(num_lbl, np.sum(unique_lbl >= 10), g_min_lbl))
        for i in range(num_lbl):
            ori_lbl = unique_lbl[num_lbl-1-i]
            new_lbl = num_gls-i
            colors[new_lbl] = colors[ori_lbl]
            del colors[ori_lbl]
            y_dist[y_dist == ori_lbl] = new_lbl
        co_keys = sorted(colors.keys())
        co_list = [colors[i] for i in co_keys]
        cm = LinearSegmentedColormap.from_list(
            'plotcm', co_list, N=len(co_list))
        for i in co_keys:
            g_mask = y_dist == i
            if i < g_min_lbl:
                plt.scatter(x_dist[g_mask, 0], x_dist[g_mask, 1], c=colors[i], cmap=cm, marker="x",
                            s=self.config.p_s, alpha=self.config.p_alpha)
            else:
                plt.scatter(x_dist[g_mask, 0], x_dist[g_mask, 1], c=colors[i], cmap=cm,
                            marker="o", facecolors='none', edgecolors=colors[i],
                            s=self.config.p_s, alpha=self.config.p_alpha)
        # sc = plt.scatter(x_dist[:, 0], x_dist[:, 1], c=y_dist, cmap=cm, marker=",",
        #                  s=self.config.p_s, alpha=self.config.p_alpha)
        x_min, x_max = x_dist[:, 0].min() - 1, x_dist[:, 0].max() + 1
        y_min, y_max = x_dist[:, 1].min() - 1, x_dist[:, 1].max() + 1
        plt.title('TSNE {}'.format(data_suffix))
        plt.axis((x_min, x_max, y_min, y_max))
        cax, _ = matplotlib.colorbar.make_axes(ax)
        normalize = matplotlib.colors.Normalize(vmin=min(unique_lbl), vmax=max(unique_lbl))
        cb = matplotlib.colorbar.ColorbarBase(cax, cmap=cm, norm=normalize)
        # cb = plt.colorbar(cm)
        cb.set_ticks([0, max(colors.keys())])
        cb.set_ticklabels(["", ""])
        fig.canvas.draw()
        # save as image
        save_path = os.path.join(self.config.save_dir,
                                 'Te{}.FM+VI.{}.tsne.{}.png'.format(self.config.dataset, self.config.suffix, data_suffix))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close('all')

        # split u, l ver
        if splitul:
            data_suffix += '2'
            colors = {0: 'lightblue', 1: 'g', 2: 'r'}
            # lbl_str = {0: 'unlabel', 1: 'label', 2: 'gen'}
            # from matplotlib.colors import LinearSegmentedColormap
            cm = LinearSegmentedColormap.from_list(
                'splitul', colors.values(), N=len(colors))
            ist = {0: 0, 1: data_num[0], 2: - data_num[1]}
            ied = {0: data_num[0], 2: None} # 0: u_lbls.shape[0], 1: u_lbls.shape[0] + l_lbls.shape[0]
            plt.ioff()
            fig = plt.figure(figsize=self.p_fs, dpi=self.config.p_d)
            fig.add_subplot(111)
            # draw the figure first...
            for g in colors.keys(): # u, g
                if g == 1:
                    if "_" in data_suffix:
                        continue
                    assert len(self.loaders) >= 4, "no label ind"
                    ix = self.loaders[3]
                    ix = ix[ix < x_dist.shape[0]]
                    y_dist[ix] = g
                else:
                    y_dist[ist[g]:ied[g]] = g
            sc = plt.scatter(x_dist[:, 0], x_dist[:, 1], c=y_dist, cmap=cm, marker=",",
                             s=self.config.p_s, alpha=self.config.p_alpha)
            plt.title('TSNE {}'.format(data_suffix))
            cb = plt.colorbar(sc)
            cb.set_ticks([0, colors.keys()[-1]])
            cb.set_ticklabels(["", ""])
            plt.axis((x_min, x_max, y_min, y_max))
            fig.canvas.draw()
            # save as image
            save_path = os.path.join(self.config.save_dir,
                                     'Te{}.FM+VI.{}.tsne.{}.png'.format(self.config.dataset, self.config.suffix, data_suffix))
            plt.savefig(save_path, bbox_inches='tight')
            plt.close('all')
            # plot ul
            data_suffix = data_suffix[:-1]+"3"
            colors.pop(2)
            cm = LinearSegmentedColormap.from_list(
                'gul3', colors.values(), N=len(colors))
            fig = plt.figure(figsize=self.p_fs, dpi=self.config.p_d)
            fig.add_subplot(111)
            g_mask = y_dist != 2
            x_dist = x_dist[g_mask]
            y_dist = y_dist[g_mask]
            sc = plt.scatter(x_dist[:, 0], x_dist[:, 1], c=y_dist, cmap=cm, marker=",",
                             s=self.config.p_s, alpha=self.config.p_alpha)
            plt.title('TSNE {}'.format(data_suffix))
            cb = plt.colorbar(sc)
            cb.set_ticks([0, colors.keys()[-1]])
            cb.set_ticklabels(["", ""])
            plt.axis((x_min, x_max, y_min, y_max))
            fig.canvas.draw()
            # save as image
            save_path = os.path.join(self.config.save_dir,
                                     'Te{}.FM+VI.{}.tsne.{}.png'.format(self.config.dataset, self.config.suffix, data_suffix))
            plt.savefig(save_path, bbox_inches='tight')
            plt.close('all')

    def eval_feats(self, data_loader, data="u"):
        if data == "u":
            if not hasattr(self, 'u_lbl'):
                self.u_feats, self.u_lbl = self.eval_feat(data_loader)
            return self.u_feats, self.u_lbl
        elif data == "l":
            if not hasattr(self, 'l_lbl'):
                self.l_feats, self.l_lbl = self.eval_feat(data_loader)
            return self.l_feats, self.l_lbl
        elif data == "d":
            if not hasattr(self, 'd_lbl'):
                self.d_feats, self.d_lbl = self.eval_feat(data_loader)
            return self.d_feats, self.d_lbl
        else:
            print("There is no data={}".format(data))

    def eval_feat(self, data_loader, max_batch=None, ema=False):
        if ema:
            dis = self.ema_dis
        else:
            dis = self.dis
        self.gen.eval()
        dis.eval()
        # self.enc.eval()
        # if self.config.te:
        #     max_batch = 1
        feats = lbls = None
        for i, (images, labels) in enumerate(data_loader.get_iter(shuffle=False)):
            images = Variable(images.cuda(), volatile=True)
            labels = Variable(labels.cuda(), volatile=True)
            feat = self.get_feat(images, ema=ema)
            if i == 0:
                feats = feat.data.clone()
                lbls  = labels.data.clone()
            else:
                feats = torch.cat((feats, feat.data.clone()), dim=0)
                lbls = torch.cat((lbls, labels.data.clone()), dim=0)
            if max_batch is not None and i >= max_batch - 1: break

        feats, lbls = feats.cpu().numpy(), lbls.cpu().numpy()

        return feats, lbls

    def calculate_remaining(self, t1, t2, progress):  # ta
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

        time_str = '| ' + time_str
        return time_str

    def load_model(self, net, net_label, epo_label):  # ta
        load_filename = 'VI.{}_{}_net_{}.pth'.format(self.config.suffix, epo_label, net_label)
        load_path = os.path.join(self.config.save_dir, load_filename)
        if not os.path.exists(load_path):
            print("There is no {}!".format(load_filename))
            return
        load_net = torch.load(load_path)
        net.cpu()
        model.load_my_state_dict(net, load_net)

        if torch.cuda.is_available():
            net.cuda()

    def resume(self, epo_label):  # ta
        # load old
        self.load_model(self.dis, 'D', epo_label)
        if hasattr(self.config, 'con_coef'):
            self.load_model(self.ema_dis, 'M', epo_label)
        if hasattr(self.config, 'gen_mode') and self.config.gen_mode != "non":
            self.load_model(self.gen, 'G', epo_label)
        if hasattr(self.config, 'gen_mode') and self.config.gen_mode == "z2i":
            self.load_model(self.enc, 'E', epo_label)
        if hasattr(self.config, 'dis_double') and self.config.dis_double:
            self.load_model(self.dis_dou, 'D2', epo_label)

    def test(self):
        config = self.config
        batch_per_epoch = int((len(self.loaders[0]) + config.train_batch_size - 1) / config.train_batch_size)

        iter_num = batch_per_epoch * (config.last_epochs - 1)
        if config.last_epo_lbl != 0:
            iter_num = config.last_epo_lbl
        if config.mu:
            config.suffix = "{}_s{}".format(config.suffix, config.train_step)
        self.resume(iter_num)

        # turn save_dir into another folder
        self.config.save_dir = os.path.join(self.config.save_dir, 'T{}'.format(self.config.suffix))
        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)

        gzlab = -1
        if config.gzlab:
            gzlab = config.num_label

        if config.te:
            config.gen_iter = 2
        if config.gen_iter == 1:
            if config.gen_mode == "z2i":
                self.visualize(gzlab=gzlab)
            else:
                self.visualize(self.loaders[0])
        else:
            if config.gen_mode == "z2i":
                self.visualize_iter(bsize=100, iters=config.gen_iter,
                                    data_suffix="g", gzlab=gzlab)
            else:
                self.visualize_iter(data_loader=self.loaders[0], bsize=self.loaders[0].batch_size,
                                    iters=config.gen_iter, data_suffix="g", gzlab=gzlab)

        if config.alldata:
            print("loader num: {}".format(len(self.loaders)))
            self.visualize_iter(self.loaders[1], 400, data_suffix="d")
            self.visualize_iter(self.loaders[0], 400, data_suffix="u")
            if len(self.loaders) >= 3:
                self.visualize_iter(self.loaders[2], 400, data_suffix="l")
        if config.tsne:
            if hasattr(self, 'gen'):
                if self.config.mu and self.config.declbl:
                    if config.gen_iter != -1:
                        assert config.gen_iter * self.loaders[0].batch_size <= len(self.loaders[0]), \
                            "out of dataset: {}*{} > {}".format(config.gen_iter, self.loaders[0].batch_size,
                                                                len(self.loaders[0]))
                        img_num = config.gen_iter * self.loaders[0].batch_size // config.num_label
                        self.img_per_cls = [img_num] * config.num_label
                        for i in range(config.num_label):
                            self.gen_feat[i] = self.gen_feat[i][:img_num]
                    self.gen_feat = torch.cat(self.gen_feat, 0)
                    print("# gen_feat: {}; ipc: {}".format(self.gen_feat.size(0), self.img_per_cls))
                self.gen_feat = self.gen_feat.cpu().numpy()
            if config.te:
                # self.tsne("g")  # t1\
                self.tsne("gul")
                # self.tsne("gu")
            else:
                if config.mu:
                    self.tsne("g")
                self.tsne("gu")
                self.tsne("gl")
                self.tsne("gul")
                self.tsne("gd")


if __name__ == '__main__':
    cc = config.cifarmg_config()
    parser = argparse.ArgumentParser(description='cmgmd_tester.py')
    parser.add_argument('-suffix', default='mg0', type=str, help="Suffix added to the save images.")
    parser.add_argument('-r', dest='resume', action='store_true')
    parser.add_argument('-dataset', default=cc.dataset, type=str,
                        help="dataset: cifar, stl10, coil20")
    parser.add_argument('-last_epochs', default=cc.last_epochs, type=int,
                        help="last epochs")
    parser.add_argument('-last_epo_lbl', default=0, type=int,
                        help="last epoch lbl")
    parser.add_argument('-gen_iter', default=1, type=int,
                        help="gen iteration times. def: 1; full: -1")
    parser.add_argument('-cmp', default='winter', type=str,
                        help="color map name")
    parser.add_argument('-alldata', dest='alldata', action='store_true',
                        help="plot dev, unl, lab image same as gen")
    parser.add_argument('-gzlab', dest='gzlab', action='store_true',
                        help="Gen z add label")
    parser.add_argument('-nsg', dest='nsg', action='store_true',
                        help="no save gen img")
    parser.add_argument('-nstf', dest='nstf', action='store_true',
                        help="no save gen true/fake ana")
    parser.add_argument('-tsne', dest='tsne', action='store_true',
                        help="plot tsne")
    parser.add_argument('-banl', default='', type=str,
                        help="the lab num be ignored on tsne. ex: 0,1,2")
    parser.add_argument('-showlab', default='', type=str,
                        help="the only lab num be shown on tsne. ex: 0,1")
    parser.add_argument('-p_fs', default='12,9', type=str,
                        help="plot fig size, def: 12,9")
    parser.add_argument('-p_d', default=300, type=int,
                        help="plot dpi, def: 300")
    parser.add_argument('-p_scale', default=20.0, type=float,
                        help="plot point scale, def: 20.0")
    parser.add_argument('-p_s', default=20.0, type=float,
                        help="plot s, def: 20.0")
    parser.add_argument('-p_alpha', default=0.5, type=float,
                        help="plot alpha, def:0.5")
    parser.add_argument('-te', dest='te', action='store_true',
                        help="just for test colorbar")
    parser.add_argument('-mu', dest='mu', action='store_true',
                        help="mu series: G(En De), D(Classifier)")
    parser.add_argument('-dl', dest='declbl', action='store_true',
                        help="mu series: decode same label image")
    parser.set_defaults(alldata=False)
    parser.set_defaults(gzlab=False)
    parser.set_defaults(tsne=False)
    parser.set_defaults(nsg=False)
    parser.set_defaults(nstf=False)
    parser.set_defaults(te=False)
    parser.set_defaults(mu=False)
    parser.set_defaults(declbl=False)

    parser.add_argument('-image_side', default="32", type=int,
                        help="cifar: 32, stl10: 96")
    parser.add_argument('-noise_size', default=cc.noise_size, type=int,
                        help="gen noise size")
    parser.add_argument('-train_batch_size', default=cc.train_batch_size, type=int,
                        help="labeled batch size")
    parser.add_argument('-train_batch_size_2', default=cc.train_batch_size_2, type=int,
                        help="unlabeled batch size")
    parser.add_argument('-gen_mode', default=cc.gen_mode, type=str,
                        help="gen model mode: z2i, i2i")

    parser.add_argument('-d', dest='double_input_size', action='store_true',
                        help="double input size")
    parser.add_argument('-f', dest='flip', action='store_true',
                        help="flip input or not")
    parser.add_argument('-dd', dest='dis_double', action='store_true',
                        help="dis double")
    parser.add_argument('-dt', dest='dis_triple', action='store_true',
                        help="dis tri")
    parser.set_defaults(resume=False)
    parser.set_defaults(double_input_size=cc.double_input_size)
    parser.set_defaults(flip=cc.flip)
    args = parser.parse_args()

    tester = Tester(cc, args)
    tester.test()
