
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
import glob # use '*' in path
import math
import argparse
from collections import OrderedDict

import numpy as np
from utils import *

class Tester(object):

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

        self.labeled_loader, self.unlabeled_loader, self.unlabeled_loader2, self.dev_loader, self.special_set = data.get_cifar_loaders(config)

        self.dis = model.Discriminative(config).cuda()
        self.gen = model.Generator(image_size=config.image_size, noise_size=config.noise_size).cuda()
        self.enc = model.Encoder(config.image_size, noise_size=config.noise_size, output_params=True).cuda()

        # load model    # ta
        self.load_network(self.dis, 'D', strict=False)
        self.load_network(self.gen, 'G', strict=False)
        self.load_network(self.enc, 'E', strict=False)

        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)

        log_path = os.path.join(self.config.save_dir, '{}.FM+VI.{}.txt'.format(self.config.dataset, self.config.suffix))
        self.logger = open(log_path, 'wb')
        self.logger.write(disp_str)

        print self.dis

    def _get_vis_images(self, labels):
        labels = labels.data.cpu()
        vis_images = self.special_set.index_select(0, labels)
        return vis_images

    def eval_true_fake(self, data_loader, max_batch=None):
        self.gen.eval()
        self.dis.eval()
        self.enc.eval()
        
        cnt = 0
        unl_acc, gen_acc, max_unl_acc, max_gen_acc = 0., 0., 0., 0.
        for i, (images, _) in enumerate(data_loader.get_iter()):
            images = Variable(images.cuda(), volatile=True)
            noise = Variable(torch.Tensor(images.size(0), self.config.noise_size).uniform_().cuda(), volatile=True)

            unl_feat = self.dis(images, feat=True)
            gen_feat = self.dis(self.gen(noise), feat=True)

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

    def eval(self, data_loader, max_batch=None):
        self.gen.eval()
        self.dis.eval()
        self.enc.eval()

        loss, incorrect, cnt = 0, 0, 0
        for i, (images, labels) in enumerate(data_loader.get_iter()):
            images = Variable(images.cuda(), volatile=True)
            labels = Variable(labels.cuda(), volatile=True)
            pred_prob = self.dis(images)
            loss += self.d_criterion(pred_prob, labels).data[0]
            cnt += 1
            incorrect += torch.ne(torch.max(pred_prob, 1)[1], labels).data.sum()
            if max_batch is not None and i >= max_batch - 1: break
        return loss / cnt, incorrect


    def visualize(self):
        self.gen.eval()
        self.dis.eval()
        self.enc.eval()

        vis_size = 100
        noise = Variable(torch.Tensor(vis_size, self.config.noise_size).uniform_().cuda())
        gen_images = self.gen(noise)

        save_path = os.path.join(self.config.save_dir, '{}.FM+VI.{}.png'.format(self.config.dataset, self.config.suffix))
        vutils.save_image(gen_images.data.cpu(), save_path, normalize=True, range=(-1,1), nrow=10)

    def param_init(self):
        def func_gen(flag):
            def func(m):
                if hasattr(m, 'init_mode'):
                    setattr(m, 'init_mode', flag)
            return func

        images = []
        for i in range(500 / self.config.train_batch_size):
            lab_images, _ = self.labeled_loader.next()
            images.append(lab_images)
        images = torch.cat(images, 0)

        self.gen.apply(func_gen(True))
        noise = Variable(torch.Tensor(images.size(0), self.config.noise_size).uniform_().cuda())
        gen_images = self.gen(noise)
        self.gen.apply(func_gen(False))

        self.enc.apply(func_gen(True))
        self.enc(gen_images)
        self.enc.apply(func_gen(False))

        self.dis.apply(func_gen(True))
        logits = self.dis(Variable(images.cuda()))
        self.dis.apply(func_gen(False))

    def calculate_remaining(self, t1, t2, epoch):   # ta
        progress = (epoch + 0.) / self.config.max_epochs
        elapsed_time = t2 - t1
        if (progress > 0):
            remaining_time = elapsed_time * (1/progress) - elapsed_time
        else:
            remaining_time = 0

        # return progress, remaining_time
        psec = int(remaining_time % 60)
        pmin = int((remaining_time // 60) % 60)
        phr = int(remaining_time / 3600)
        time_str = '[{:8.2%}], remain: {:3d}:{:2d}:{:2d} '.format(progress, phr, pmin, psec)

        time_str = '| ' + time_str + '\n'
        return time_str

    def load_network(self, net, net_label, strict=True):   # ta
        save_filename = glob.glob('VI.{}_*_net_{}.pth'.format(self.config.suffix, net_label))[0]
        save_path = os.path.join(self.config.save_dir, save_filename)
        net.cpu()
        net.load_state_dict(torch.load(save_path), strict=strict)
        net.cuda()

    def save_model(self, net, net_label, epo_label):   # ta
        save_filename = 'VI.{}_{}_net_{}.pth'.format(self.config.suffix, epo_label, net_label)
        save_path = os.path.join(self.config.save_dir, save_filename)
        torch.save(net.cpu().state_dict(), save_path)
        if torch.cuda.is_available():
            net.cuda()

    def del_model(self, net_label, epo_label):   # ta
        del_filename = 'VI.{}_{}_net_{}.pth'.format(self.config.suffix, epo_label, net_label)
        del_path = os.path.join(self.config.save_dir, del_filename)
        if os.path.exists(del_path):
            os.remove(del_path)
        else:
            print("The file does not exist, {}".format(del_path))

    def save(self, epo_label):   # ta
        # save new
        self.save_model(self.dis, 'D', epo_label)
        self.save_model(self.gen, 'G', epo_label)
        self.save_model(self.enc, 'E', epo_label)
        # del old
        if epo_label >= self.config.eval_period:
            epo_label -= self.config.eval_period
            self.del_model('D', epo_label)
            self.del_model('G', epo_label)
            self.del_model('E', epo_label)

    def test(self):
        config = self.config
        config.max_epochs = 1
        self.param_init()

        self.iter_cnt = 0
        iter, min_dev_incorrect = 0, 1e6
        
        batch_per_epoch = int((len(self.unlabeled_loader) + config.test_batch_size - 1) / config.train_batch_size)
        min_lr = config.min_lr if hasattr(config, 'min_lr') else 0.0
        start_time = time.time()
        while True:

            if iter % batch_per_epoch == 0:
                epoch = iter / batch_per_epoch
                if epoch >= config.max_epochs:
                    break
                epoch_ratio = float(epoch) / float(config.max_epochs)


            if iter % config.vis_period == 0:
                self.visualize()

            if iter % config.eval_period == 0:
                train_loss, train_incorrect = self.eval(self.labeled_loader)
                dev_loss, dev_incorrect = self.eval(self.dev_loader)

                unl_acc, gen_acc, max_unl_acc, max_gen_acc = self.eval_true_fake(self.dev_loader, 10)

                train_incorrect /= 1.0 * len(self.labeled_loader)
                dev_incorrect /= 1.0 * len(self.dev_loader)
                min_dev_incorrect = min(min_dev_incorrect, dev_incorrect)

                disp_str = '#{}\ttrain: {:.4f}, {:.4f} | dev: {:.4f}, {:.4f} | best: {:.4f}'.format(
                    iter, train_loss, train_incorrect, dev_loss, dev_incorrect, min_dev_incorrect)
                
                disp_str += ' | [Eval] unl acc: {:.4f}, gen acc: {:.4f}, max unl acc: {:.4f}, max gen acc: {:.4f}'.format(unl_acc, gen_acc, max_unl_acc, max_gen_acc)
                disp_str += '\n'

                # timer   # ta
                time_str = self.calculate_remaining(start_time, time.time(), iter / batch_per_epoch)
                # save model   # ta
                self.save(iter)

                self.logger.write(disp_str)
                sys.stdout.write(disp_str)
                sys.stdout.write(time_str)   # ta
                sys.stdout.flush()

            iter += 1
            self.iter_cnt += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cifar_tester.py')
    parser.add_argument('-suffix', default='run0', type=str, help="Suffix added to the save images.")
    parser.add_argument('-r', default='', type=str, help="Suffix added to the save images.")

    args = parser.parse_args()

    tester = Tester(config.cifar_config(), args)
    tester.test()


