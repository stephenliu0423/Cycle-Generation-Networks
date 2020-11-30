import torch
import torch.nn as nn
from torch.optim import RMSprop
from torch.optim import Adam

import numpy as np
from copy import deepcopy

from model import Generator
from memory import CycleMemory
from util import to_tensor


criterion = nn.MSELoss()


class CycleGAN(object):

    def __init__(self, interact, args, g_network, f_network, is_training=True, optimizer_type="adam"):
        self.dim = args.dim
        self.num_rec_items = args.num_rec_items
        self.interact = interact
        self.is_training = is_training
        self.batch_size = int(args.batch_size)
        self.memory_size = args.memory_size
        self.lambda_cyc = args.lambda_cyc

        torch.manual_seed(123)

        # create g_network and f_network
        if g_network is None:
            self.g_network = Generator(args.num_rec_items, args.dim)
            self.f_network = Generator(args.num_rec_items, args.dim)
        else:
            self.g_network = deepcopy(g_network)
            self.f_network = deepcopy(f_network)

        self.g_network_optim = Adam(self.g_network.parameters(), lr=1e-4)
        self.f_network_optim = Adam(self.f_network.parameters(), lr=1e-4)

        if torch.cuda.is_available():
            print('using gpu')
            self.g_network = self.g_network.cuda()
            self.f_network = self.f_network.cuda()
        else:
            print('no gpu')

        self.memory_1 = CycleMemory(self.memory_size)
        self.memory_2 = CycleMemory(self.memory_size)

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def mmd_rbf(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        source = source.view(-1, self.num_rec_items, self.dim)
        target = target.view(-1, self.num_rec_items, self.dim)
        batch_size = int(source.size()[1])
        loss_all = []
        for i in range(int(source.size()[0])):
            kernels = self.guassian_kernel(source[i], target[i], kernel_mul=kernel_mul, kernel_num=kernel_num,
                                      fix_sigma=fix_sigma)
            xx = kernels[:batch_size, :batch_size]
            yy = kernels[batch_size:, batch_size:]
            xy = kernels[:batch_size, batch_size:]
            yx = kernels[batch_size:, :batch_size]
            loss = torch.mean(xx + yy - xy - yx)
            loss_all.append(loss)
        return sum(loss_all) / len(loss_all)

    def select_action(self, items_1, items_2, user_emb_1, user_emb_2, select_action_type):
        """
        when testing:
        select_action_type = network_type
        """
        items_1_a, items_2_a = None, None
        if select_action_type == 1 or select_action_type == 3:
            items_1 = to_tensor(items_1[np.newaxis, :][np.newaxis, :])
            user_emb_2 = to_tensor(user_emb_2[np.newaxis, :][np.newaxis, :])

            items_1_a = self.g_network(items_1, user_emb_2)
            items_1_a = items_1_a.cpu().detach().numpy()
            items_1_a = np.resize(items_1_a, (self.num_rec_items, self.dim))
        if select_action_type == 2 or select_action_type == 3:
            items_2 = to_tensor(items_2[np.newaxis, :][np.newaxis, :])
            user_emb_1 = to_tensor(user_emb_1[np.newaxis, :][np.newaxis, :])

            items_2_a = self.f_network(items_2, user_emb_1)
            items_2_a = items_2_a.cpu().detach().numpy()
            items_2_a = np.resize(items_2_a, (self.num_rec_items, self.dim))

        return items_1_a, items_2_a

    def observe(self, items_1a, items_2a, items_1b, items_2b, user_emb_1, user_emb_2):
        if self.is_training:
            self.memory_1.append(items_1a, items_2a, items_1b, items_2b, user_emb_1, user_emb_2)
            # self.memory_2.append(items_2, items_1, user_emb_2, user_emb_1)

    def update_generator(self, is_debug):
        # print('here')

        # lambda_1 = 0.5  # 0.5
        # lambda_2 = 0.5  # 0.5

        items_1a, items_2a, items_1b, items_2b, user_emb_1, user_emb_2 = self.memory_1.sample_and_split(self.batch_size)

        # item_a = item_a * user_emb_a2
        # item_b = item_b * user_emb_b2

        self.g_network.zero_grad()
        self.f_network.zero_grad()

        # items_1a = to_tensor(items_1a[:, np.newaxis])
        # items_2a = to_tensor(items_2a[:, np.newaxis])
        items_1b = to_tensor(items_1b[:, np.newaxis])
        items_2b = to_tensor(items_2b[:, np.newaxis])

        user_emb_1 = to_tensor(user_emb_1)
        user_emb_2 = to_tensor(user_emb_2)

        # GAN loss
        item_g = self.g_network(items_1b, user_emb_2)
        item_g = item_g.view(-1, 1, self.num_rec_items, self.dim)
        # item_1_a = item_1_a / user_emb_a1

        g_loss = self.mmd_rbf(items_2b, item_g)
        # print('g_loss = ', g_loss)
        # g_gan_loss = - self.discriminator_g(item_1_a)

        item_f = self.f_network(items_2b, user_emb_1)
        item_f = item_f.view(-1, 1, self.num_rec_items, self.dim)
        # item_2_a = item_2_a / user_emb_b1

        f_loss = self.mmd_rbf(items_1b, item_f)
        # f_gan_loss = - self.discriminator_f(item_2_a)

        # cycle loss
        item_gf = self.f_network(self.g_network(items_1b, user_emb_2), user_emb_1)  # TODO think
        item_gf = item_gf.view(-1, 1, self.num_rec_items, self.dim)
        g_f = criterion(items_1b, item_gf)

        item_fg = self.g_network(self.f_network(items_2b, user_emb_1), user_emb_2)
        item_fg = item_fg.view(-1, 1, self.num_rec_items, self.dim)
        f_g = criterion(items_2b, item_fg)

        cyc_loss = g_f + f_g

        # UPDATE WIGHTS
        if is_debug:
            print('loss = ', g_loss, f_loss, g_f, f_g)
        loss = 2 * g_loss + f_loss + self.lambda_cyc * cyc_loss
        # print('loss = ', loss)
        loss = loss.mean()
        loss.backward()
        self.g_network_optim.step()
        self.f_network_optim.step()

