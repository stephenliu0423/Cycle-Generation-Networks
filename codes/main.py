import argparse
import psutil
import gc
import numpy as np
from tqdm import tqdm
import torch
import random

from interaction import Interaction
from cycle_gan import CycleGAN
from util import to_tensor
from mapping import mmr
import copy


np.random.seed(123)


def test_one_network(interact, agent, network_type):
    done, epi, cur_reward = 0, 0, 0
    precision = np.zeros(args.test_max_epi)
    precision_at = np.zeros(args.test_max_epi)
    recall = np.zeros(args.test_max_epi)
    recall_at = np.zeros(args.test_max_epi)
    hr = np.zeros(args.test_max_epi)
    predict = {}
    pre_all = []
    rec_all = []
    hr_all = []

    if network_type == 1:
        test_data = interact.test_user_dict_2
        test_domain = 2
    else:
        test_data = interact.test_user_dict_1
        test_domain = 1

    print('test_domain = ', test_domain)

    test_user_num = len(test_data)
    print('test_user_num = ', test_user_num)
    for cur_user_id in tqdm(test_data.keys()):

        interact.update()
        interact.create_test_id(cur_user_id, test_domain)
        # print('length1 = ', len(interact.test_user_dict_2[cur_user_id]))
        items_1, items_2, _, _, user_emb_1, user_emb_2 = interact.init(cur_user_id, 0, test_domain)

        items_1_a, items_2_a = agent.select_action(items_1, items_2, user_emb_1, user_emb_2, network_type)

        if test_domain == 1:
            pre_action = items_2_a
        else:
            pre_action = items_1_a
        id_action, vector = interact.mapping(pre_action, test_domain)

        reward, done, all_number = interact.result(cur_user_id, id_action, 0, domain=test_domain)
        predict[cur_user_id] = list(id_action)
        # print('id, vector = ', id_action, vector)
        # input('debug')

        precision[epi] += float(reward) / float(args.num_rec_items)
        pre_all.append(float(reward) / float(args.num_rec_items))
        precision_at[epi] += done / (float(args.num_rec_items) * (epi + 1))
        recall[epi] += float(reward) / float(all_number)
        rec_all.append(float(reward) / float(all_number))
        recall_at[epi] += float(done) / float(all_number)
        hr[epi] += float(min(reward, 1))
        hr_all.append(float(min(reward, 1)))

    precision = precision / test_user_num
    precision_at = precision_at / test_user_num
    recall = recall / test_user_num
    recall_at = recall_at / test_user_num
    hr = hr / test_user_num

    print('network type = ', network_type)
    print('hr = ', hr)
    print('precision = ', precision)
    print('precision@k = ', precision_at)
    print('recall = ', recall)
    print('recall_at = ', recall_at)
    print('ave pre = ', np.mean(precision))
    print()

    result = [hr[0], precision[0], recall[0]]

    return result, predict, np.array(pre_all), np.array(rec_all), np.array(hr_all)


def test_cycle_gan(g_network, f_network, accuracy):
    print('new testing...')
    interact = Interaction(args, False)
    agent = CycleGAN(interact, args, g_network=g_network, f_network=f_network, is_training=False)
    result1 = []
    result2 = []
    predict_1 = {}
    predict_2 = {}

    # testing network g
    # result1, predict_1, pre_all, rec_all, hr_all = test_one_network(interact, agent, network_type=1)

    # testing network f
    result2, predict_2, pre_all, rec_all, hr_all = test_one_network(interact, agent, network_type=2)

    for x in locals().keys():
        del x
    gc.collect()
    return result1, result2, predict_1, predict_2, pre_all, rec_all, hr_all


def train_cycle_gan(g_network, f_network):
    interact = Interaction(args, True)
    agent = CycleGAN(interact, args, g_network=g_network, f_network=f_network, is_training=True)
    cur_user_id = None
    accuracy = 0
    recall = 0
    done, epi = 0, 0
    partition_id = 0

    for step in tqdm(range(args.train_max_steps)):
        if cur_user_id is None:
            done, epi = 0, 0
            interact.update()
            cur_user_id = interact.create_train_id()
            # print('id = ', cur_user_id)

        items_1a, items_2a, debug_id_1, debug_id_2, user_emb_1, user_emb_2 = interact.init(cur_user_id, partition_id, 0)
        items_1b, items_2b, debug_id, _, _, _ = interact.init(cur_user_id, partition_id + 1, 0)
        partition_id += 1
        agent.observe(items_1a, items_2a, items_1b, items_2b, user_emb_1, user_emb_2)
        if partition_id == 0:
            agent.observe(None, None, items_1b, items_2b, user_emb_1, user_emb_2)

        epi += 1

        if step > args.warmup:
            if (step + 1) % 500 == 0:
                agent.update_generator(1)
            else:
                agent.update_generator(0)

        if epi >= args.train_max_epi or partition_id + 1 >= max(interact.train_user_dict_1[cur_user_id].keys()):
            cur_user_id = None
            partition_id = 0

        # test
        mem = psutil.virtual_memory()
        mem = int(round(mem.percent))
        if mem >= 90:
            input('memory warning!')

        if (step + 1) % 100 == 0 and step > 1000:  # and step > 1500

            print('current memory usage: ', mem, '%')
            result1, result2, predict_1, predict_2, pre_all, rec_all, hr_all = test_cycle_gan(agent.g_network, agent.f_network, accuracy)
            if result2[1] > accuracy:
                accuracy = result2[1]
                recall = result2[2]
                f = open('result_batch' + str(args.batch_size) + '_rec' + str(args.num_rec_items) + '_dim' + str(
                    args.dim) + '_lambda' + str(args.lambda_cyc) + '.txt', 'w')
                f.write(str(result2))
                f.close()
                f = open('predict2_crn' + str(args.num_rec_items) + '.txt', 'w')
                f.write(str(predict_2))
                f.close()
                torch.save(agent.g_network, 'g_network_model_best.pt')
                torch.save(agent.f_network, 'f_network_model_best.pt')

        for x in list(locals().keys()):
            del x  # locals()[x]
        gc.collect()


if __name__ == "__main__":
    # Initialize the parameters
    dim = 10
    parser = argparse.ArgumentParser(description="Cycle GAN for cross domain recommendation")

    parser.add_argument('--dim', default=dim, help='dimension of item embedding ')
    parser.add_argument('--num_rec_items', default=5, help='number of recommendation items')
    parser.add_argument('--lambda_cyc', default=0.5, help='number of recommendation items')

    parser.add_argument('--train_max_steps', default=10000000, help='max step for the whole training process')
    parser.add_argument('--init_actor_steps', default=2000, help='init_actor_steps')
    parser.add_argument('--init_actor_memory_size', default=10000, help='limited init actor memory size')
    parser.add_argument('--init_actor_batch_size', default=256, help='batch size for initializing actor network')
    parser.add_argument('--batch_size', default=64, help='the size of batch for updating th policy')
    parser.add_argument('--warmup', default=65, help='when the step is larger than warmup,start to update the policy')
    parser.add_argument('--train_max_epi', default=1, help='the longest prcess for training one specific user')
    parser.add_argument('--train_critic_steps', default=1000, help='the longest process for training critic network')
    parser.add_argument('--memory_size', default=10000, help='limited memory size')
    parser.add_argument('--test_max_epi', default=1, help='the longest process for testing one specific user')
    parser.add_argument('--data_type', default=0, help='1 for online data, 0 for offline data')

    # amazon homes data
    parser.add_argument('--train_file_1', default='../dataset/homes/homes_train.txt',
                        help='the training file 1')
    parser.add_argument('--test_file_1', default='../dataset/homes/homes_test.txt',
                        help='the testing file 1')
    parser.add_argument('--item_emb_file_1', default='../dataset/homes/bpr_homes_dim' + str(dim) + '_item_embs.npy',
                        help='the item embedding file 1')
    parser.add_argument('--user_emb_file_1', default='../dataset/homes/bpr_homes_dim' + str(dim) + '_user_embs.npy',
                        help='the user embedding file 1')

    # amazon clothing data
    parser.add_argument('--train_file_2', default='../dataset/clothing/clothing_train.txt',
                        help='the training file 2')
    parser.add_argument('--test_file_2', default='../dataset/clothing/clothing_test.txt',
                        help='the testing file 2')
    parser.add_argument('--item_emb_file_2', default='../dataset/clothing/bpr_clothing_dim' + str(dim) + '_item_embs.npy',
                        help='the item embedding file 2')
    parser.add_argument('--user_emb_file_2', default='../dataset/clothing/bpr_clothing_dim' + str(dim) + '_user_embs.npy',
                        help='the user embedding file 2')

    args = parser.parse_args()
    data = psutil.virtual_memory()
    print('memory = ', int(round(data.percent)))

    print('new training...')
    print('num_rec_items = ', args.num_rec_items)
    print('dim = ', args.dim)
    print('lambda_cyc = ', args.lambda_cyc)
    print('batch_size = ', args.batch_size)
    train_cycle_gan(None, None)
