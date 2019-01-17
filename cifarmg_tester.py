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

        for li in logger:
            if "|" in li: break
            key, val = li.split(" : ")
            val = self.int_float_bool(val[:-1])
            if hasattr(config, key) and key != 'last_epochs':
                setattr(self.config, key, val)
        logger.close()

        # self.labeled_loader, self.unlabeled_loader, self.dev_loader, self.special_set = data.get_cifar_loaders_test(
        #     config)
        self.loaders = data.get_cifar_loaders_test(config)
        self.dis = model.Discriminative(config).cuda()
        self.ema_dis = model.Discriminative(config).cuda()  # , ema=True).cuda()
        if hasattr(self.config, 'gen_mode') and self.config.gen_mode != "non":
            self.gen = model.generator(image_size=config.image_size, noise_size=config.noise_size,
                                       large=config.double_input_size, gen_mode=config.gen_mode).cuda()

        dis_para = [{'params': self.dis.parameters()}, ]
        if 'm' in config.dis_mode:  # svhn: 168; cifar:192
            self.m_criterion = FocalLoss(gamma=2)

        if config.dis_double:
            self.dis_dou = model.Discriminative_out(config).cuda()
            dis_para.append({'params': self.dis_dou.parameters()})

        if config.gen_mode == "z2i":
            self.enc = model.Encoder(config.image_size, noise_size=config.noise_size, output_params=True).cuda()

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
        else:  # 1
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

    def visualize_iter(self, data_loader=None, bsize=400, iters=1, data_suffix="g", gzlab=-1):
        assert data_loader is not None or "g" in data_suffix, "g or loader"
        iter_num = lambda a, b: int((len(a) + b - 1) / b)
        if data_loader is not None:
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
        # self.enc.eval()

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
        elif "g" not in data_suffix and data_loader is not None:
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
                cnt += labs.size(0)
                if cnt + labs.size(0) > vis_size:
                    break
        elif self.config.gen_mode == "i2i":
            gen_images = None
            i, cnt = 0, 0
            while (True):
                images, _ = data_loader.next()
                images = Variable(images.cuda(), volatile=True)
                gen_image = self.gen(images)
                if i == 0:
                    gen_images = gen_image
                else:
                    gen_images = torch.cat((gen_images, gen_image), 0)

                cnt += images.size(0)
                if cnt + images.size(0) > vis_size:
                    break

        # for tsne
        if "g" in data_suffix and self.config.tsne:
            feat = self.dis(gen_images, feat=True).data
            if self.gen_feat is None:
                self.gen_feat = feat
            else:
                self.gen_feat = torch.cat((self.gen_feat, feat), dim=0)
        if self.config.nsg:
            return
        save_path = os.path.join(self.config.save_dir,
                                 'Te{}.FM+VI.{}.{}.png'.format(self.config.dataset, self.config.suffix, data_suffix))
        vutils.save_image(gen_images.data.cpu(), save_path, normalize=True, range=(-1, 1), nrow=nrow)
        # dis true img
        gen_logits = self.dis(gen_images)
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
        topk = 10
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

        # get fake feature
        if "g" in data_suffix:
            # pdb.set_trace()
            # if self.config.te:
            #     g_lab = np.arange(10).repeat(10)
            # else:
            g_lab = np.array([self.config.num_label]).repeat(self.gen_feat.shape[0])
            if cifar_feats is None:
                cifar_feats = self.gen_feat
                lbls = g_lab
            else:
                cifar_feats = np.concatenate((cifar_feats, self.gen_feat), axis=0)
                lbls = np.concatenate((lbls, g_lab), axis=0)

        num_label = self.config.num_label

        data_num = [u_lbls.shape[0],  g_lab.shape[0]] \
            if "g" in data_suffix and "u" in data_suffix else None
        self.plot_scatter(cifar_feats, lbls, num_label+1, data_suffix, data_num)

        if not self.config.banl == "":
            # pdb.set_trace()
            data_suffix2 = data_suffix+"_" + self.config.banl
            banl = str(self.config.banl)
            ban_e = [int(i) for i in banl.split(",")]
            ban_ind = np.array([], dtype=int)
            for i in ban_e:
                ban_ind = np.concatenate((ban_ind, np.where(lbls == i)[0]), axis=0)
                num_label -= 1
            # ban_ind.sort()
            mask = np.ones(len(lbls), dtype=bool)
            mask[ban_ind] = False
            lbls2 = lbls[mask]
            cifar_feats2 = cifar_feats[mask]
            data_num = [(u_lbls.shape[0]//10) *num_label,  g_lab.shape[0]] \
                if "g" in data_suffix and "u" in data_suffix else None
            self.plot_scatter(cifar_feats2, lbls2, num_label+1, data_suffix2, data_num)

        if not self.config.showlab == "":
            data_num = [u_lbls.shape[0]//10,  g_lab.shape[0]] \
                if "g" in data_suffix and "u" in data_suffix else None
            for i in self.config.showlab.split(","):
                data_suffix2 = data_suffix+"_" + i
                show_e = int(i)
                mask = np.where(lbls == show_e)[0]
                mask = np.concatenate((mask, np.where(lbls == 10)[0]), axis=0)
                # ban_ind.sort()
                lbls2 = lbls[mask]
                cifar_feats2 = cifar_feats[mask]
                self.plot_scatter(cifar_feats2, lbls2, 2, data_suffix2, data_num)


    def plot_scatter(self, feats, y_dist, lbl_num, data_suffix, data_num=None):
        # x_dist = ts_feat
        # y_dist = lbls
        # lbl_num = num_label + 1
        print("Plot tsne {}".format(data_suffix))
        splitul = False
        if "u" in data_suffix and "l" in data_suffix:
            print("Also Plot tsne {}2".format(data_suffix))
            assert data_num is not None, "data_num is None"
            splitul = True
        if self.config.te:
            print("TSNE point num: {}".format(len(y_dist)))
        x_dist = TSNE(n_components=2).fit_transform(feats)
        if self.config.te:
            print("after TSNE transform")
        # plot
        plt.ioff()
        fig = plt.figure()
        fig.add_subplot(111)
        if lbl_num == 2:
            cm = plt.cm.get_cmap('bwr', lbl_num)
        else:
            cm = plt.cm.get_cmap('gist_rainbow', lbl_num)
        sc = plt.scatter(x_dist[:, 0], x_dist[:, 1], c=y_dist, cmap=cm, marker=",", s=0.55, alpha=0.3)
        # plt.scatter(x_dist[:, 0], x_dist[:, 1], c=y_dist, marker=",", s=0.5, alpha=0.3)
        x_min, x_max = x_dist[:, 0].min() - 1, x_dist[:, 0].max() + 1
        y_min, y_max = x_dist[:, 1].min() - 1, x_dist[:, 1].max() + 1
        plt.title('TSNE {}'.format(data_suffix))
        plt.colorbar(sc)    # , ticks=range(lbl_num))
        plt.axis((x_min, x_max, y_min, y_max))
        fig.canvas.draw()
        # save as image
        save_path = os.path.join(self.config.save_dir,
                                 'Te{}.FM+VI.{}.tsne.{}.png'.format(self.config.dataset, self.config.suffix, data_suffix))
        plt.savefig(save_path, bbox_inches='tight')
        plt.close('all')
        # Now we can save it to a numpy array.
        # npdata = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        # npdata = npdata.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # plt.close('all')
        # image_pil = Image.fromarray(npdata)
        # image_pil.save(save_path)
        # split u, l ver
        if splitul:
            data_suffix += '2'
            colors = {0: 'lightblue', 1: 'g', 2: 'r'}
            lbl_str = {0: 'unlabel', 1: 'label', 2: 'gen'}
            ist = {0: 0, 1: data_num[0], 2: - data_num[1]}
            ied = {0: data_num[0], 2: None} # 0: u_lbls.shape[0], 1: u_lbls.shape[0] + l_lbls.shape[0]
            # co = [colors[i] for i in lbls]
            plt.ioff()
            fig = plt.figure()
            fig.add_subplot(111)
            # draw the figure first...
            for g in colors.keys(): # u, g
                # ix = np.where(lbls == g)
                if g == 1:
                    if "_" in data_suffix:
                        continue
                    load_dir = os.path.dirname(self.config.save_dir)
                    ind_path = os.path.join(load_dir,
                                            '{}.FM+VI.{}.lind.npy'.format(
                                                self.config.dataset, self.config.suffix))
                    ix = np.load(ind_path)
                    x0 = x_dist[ix, 0]
                    x1 = x_dist[ix, 1]
                else:
                    x0 = x_dist[ist[g]:ied[g], 0]
                    x1 = x_dist[ist[g]:ied[g], 1]
                plt.scatter(x0, x1,
                            c=colors[g], label=lbl_str[g],
                            marker=",", s=0.55, alpha=0.3)
            plt.legend(loc=10, scatterpoints=1, bbox_to_anchor=(1.11, 0.91))
            plt.title('TSNE {}'.format(data_suffix))
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
        # self.gen.eval()
        dis.eval()
        # self.enc.eval()
        # if self.config.te:
        #     max_batch = 1
        feats = lbls = None
        for i, (images, labels) in enumerate(data_loader.get_iter(shuffle=False)):
            images = Variable(images.cuda(), volatile=True)
            labels = Variable(labels.cuda(), volatile=True)
            feat = dis(images, feat=True)
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
        net.load_my_state_dict(load_net)

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
        self.resume(iter_num)

        # turn save_dir into another folder
        self.config.save_dir = os.path.join(self.config.save_dir, 'T{}'.format(self.config.suffix))
        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)

        gzlab = -1
        if config.gzlab:
            gzlab = config.num_label

        if config.te:
            config.gen_iter = 1
        if config.gen_iter == 1:
            if config.gen_mode == "z2i":
                self.visualize(gzlab=gzlab)
            elif config.gen_mode == "i2i":
                self.visualize(self.loaders[1])
        else:
            self.visualize_iter(bsize=100, iters=config.gen_iter, data_suffix="g", gzlab=gzlab)

        if config.alldata:
            print("loader num: {}".format(len(self.loaders)))
            self.visualize_iter(self.loaders[1], 400, data_suffix="d")
            self.visualize_iter(self.loaders[0], 400, data_suffix="u")
            if len(self.loaders) == 3:
                self.visualize_iter(self.loaders[2], 400, data_suffix="l")
        if config.tsne:
            if hasattr(self, 'gen'):
                self.gen_feat = self.gen_feat.cpu().numpy()
            if config.te:
                # self.tsne("g")  # t1\
                # self.tsne("gul")
                self.tsne("gu")
            else:
                self.tsne("gu")
                self.tsne("gl")
                self.tsne("gul")
                self.tsne("gd")


if __name__ == '__main__':
    cc = config.cifarmg_config()
    parser = argparse.ArgumentParser(description='cifarmg_trainer.py')
    parser.add_argument('-suffix', default='mg0', type=str, help="Suffix added to the save images.")
    parser.add_argument('-r', dest='resume', action='store_true')
    parser.add_argument('-last_epochs', default=cc.last_epochs, type=int,
                        help="last epochs")
    parser.add_argument('-gen_iter', default=1, type=int,
                        help="gen iteration times")
    parser.add_argument('-cmp', default='winter', type=str,
                        help="color map name")
    parser.add_argument('-alldata', dest='alldata', action='store_true',
                        help="plot dev, unl, lab image same as gen")
    parser.add_argument('-gzlab', dest='gzlab', action='store_true',
                        help="Gen z add label")
    parser.add_argument('-nsg', dest='nsg', action='store_true',
                        help="no save gen img")
    parser.add_argument('-tsne', dest='tsne', action='store_true',
                        help="plot tsne")
    parser.add_argument('-banl', default='', type=str,
                        help="the lab num be ignored on tsne. ex: 0,1,2")
    parser.add_argument('-showlab', default='', type=str,
                        help="the only lab num be shown on tsne. ex: 0,1")
    parser.add_argument('-te', dest='te', action='store_true',
                        help="just for test colorbar")
    parser.set_defaults(alldata=False)
    parser.set_defaults(gzlab=False)
    parser.set_defaults(tsne=False)
    parser.set_defaults(nsg=False)
    parser.set_defaults(te=False)

    parser.add_argument('-noise_size', default=cc.noise_size, type=int,
                        help="gen noise size")
    parser.add_argument('-train_batch_size', default=cc.train_batch_size, type=int,
                        help="labeled batch size")
    parser.add_argument('-train_batch_size_2', default=cc.train_batch_size_2, type=int,
                        help="unlabeled batch size")
    parser.add_argument('-gen_mode', default=cc.gen_mode, type=str,
                        help="gen model mode: z2i, i2i")

    parser.add_argument('-d', dest='double_input_size', action='store_true')
    parser.add_argument('-f', dest='flip', action='store_true')
    parser.add_argument('-dd', dest='dis_double', action='store_true')
    parser.add_argument('-dt', dest='dis_triple', action='store_true')
    parser.set_defaults(resume=False)
    parser.set_defaults(double_input_size=cc.double_input_size)
    parser.set_defaults(flip=cc.flip)
    args = parser.parse_args()

    tester = Tester(cc, args)
    tester.test()
