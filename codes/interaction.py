"""
    This file incldues functions for loading the training and testing data.
"""
import pandas as pd
import numpy as np
import copy
from mapping import mmr
from sklearn import preprocessing


np.random.seed(123)


def load_file_dict(file_name, sep=" "):
    df = pd.read_csv(file_name, sep=sep, names=["users", "items", "ratings", "partition"])
    print('train set size = ', len(df))
    test = {}
    user_dict = {}
    temp_user_dict = {}
    for name, g in df.groupby('users'):

        for part, p in g.groupby('partition'):
            # print('part , g = ', part, p)
            movies = []
            test[name] = p.sort_values(by=["partition"], ascending=[1])["items"].values
            for i in test[name]:
                if i not in movies:
                    movies.append(i)
            temp_user_dict[part] = np.array(movies)

        # print('temp = ', temp_user_dict)
        user_dict[name] = temp_user_dict
        temp_user_dict = {}
        # print('user_dict = ', user_dict)
        # input('hi')

    return user_dict


def load_test_file_dict(file_name, sep=" "):
    df = pd.read_csv(file_name, sep=sep, names=["users", "items", "ratings", "partition"])
    print('test set size = ', len(df))
    test = {}
    user_dict = {}
    for name, g in df.groupby('users'):
        movies = []
        test[name] = g.sort_values(by=["partition"], ascending=[1])["items"].values
        for i in test[name]:
            if i not in movies:
                movies.append(i)
        user_dict[name] = np.array(movies)
    return user_dict


def load_neg_file(file_name, sep="\t"):
    f = open(file_name)
    line = f.readline()
    neg_data = {}
    while line:
        content = line.split(sep)
        t = content[0].replace('(', '').replace(')', '')
        t = list(map(int, t.split(', ')))
        neg_data[(t[0], t[1])] = list(map(int, content[1:len(content) - 1]))
        line = f.readline()
    f.close()
    return neg_data


class Interaction(object):

    def __init__(self, args, is_training):
        self.train_file_1 = args.train_file_1
        self.test_file_1 = args.test_file_1
        self.item_emb_file_1 = args.item_emb_file_1
        self.user_emb_file_1 = args.user_emb_file_1

        self.train_file_2 = args.train_file_2
        self.test_file_2 = args.test_file_2
        self.item_emb_file_2 = args.item_emb_file_2
        self.user_emb_file_2 = args.user_emb_file_2

        # self.neg_file_1 = args.neg_file_1
        # self.neg_file_2 = args.neg_file_2
        self.dim = args.dim
        self.num_rec_items = args.num_rec_items

        self.is_training = is_training

        self.data_type = args.data_type
        self.recommended_items = []
        self.cur_ratings = []
        self.cur_ratings_1 = []
        self.cur_ratings_2 = []
        self.initialize()

    def initialize(self):
        print('loading train set 1 ')
        self.train_user_dict_1 = load_file_dict(self.train_file_1)
        print('loading train set 2 ')
        self.train_user_dict_2 = load_file_dict(self.train_file_2)

        print('loading test set 1 ')
        self.test_user_dict_1 = load_test_file_dict(self.test_file_1)
        print('loading test set 2 ')
        self.test_user_dict_2 = load_test_file_dict(self.test_file_2)

        self.item_emb_1 = np.load(self.item_emb_file_1)
        self.item_emb_1 = preprocessing.minmax_scale(self.item_emb_1, feature_range=(-1, 1), axis=1, copy=True)

        self.item_emb_2 = np.load(self.item_emb_file_2)
        self.item_emb_2 = preprocessing.minmax_scale(self.item_emb_2, feature_range=(-1, 1), axis=1, copy=True)

        self.item_num_1 = self.item_emb_1.shape[0]
        self.item_num_2 = self.item_emb_2.shape[0]

        self.user_emb_1 = np.load(self.user_emb_file_1)
        self.user_emb_1 = preprocessing.minmax_scale(self.user_emb_1, feature_range=(-1, 1), axis=1, copy=True)
        self.user_emb_2 = np.load(self.user_emb_file_2)
        self.user_emb_2 = preprocessing.minmax_scale(self.user_emb_2, feature_range=(-1, 1), axis=1, copy=True)

        # the set of all items
        self.item_set_1 = set(range(self.item_num_1))
        self.item_set_2 = set(range(self.item_num_2))

        print('user num = ', len(set(self.test_user_dict_1.keys())))
        print('user num = ', len(set(self.test_user_dict_2.keys())))

        self.user_set = list(set(self.train_user_dict_1.keys()).intersection(set(self.train_user_dict_2.keys())))
        print('user num = ', len(self.user_set))

        print('item_set_1 num = ', len(self.item_set_1))
        print('item_set_2 num = ', len(self.item_set_2))

    def update(self):
        self.recommended_items = []
        self.cur_ratings_1 = []
        self.cur_ratings_2 = []

    def create_train_id(self):
        """
            randomly choose a training user
        """
        train_id = np.random.choice(self.user_set)
        self.cur_ratings_1 = self.train_user_dict_1[train_id]
        self.cur_ratings_2 = self.train_user_dict_2[train_id]
        return train_id

    def create_test_id(self, test_id, test_domain):
        if test_domain == 1:
            self.cur_ratings = self.test_user_dict_1[test_id]
        elif test_domain == 2:
            self.cur_ratings = self.test_user_dict_2[test_id]

    def find_items(self, user_id, partition_id, source_type, test_domain):
        if source_type == 1:
            data_dict = self.train_user_dict_1
            embs = self.item_emb_1
            user_emb = self.user_emb_1[user_id]
            test_data_dict = self.test_user_dict_1
        else:
            data_dict = self.train_user_dict_2
            embs = self.item_emb_2
            user_emb = self.user_emb_2[user_id]
            test_data_dict = self.test_user_dict_2
        # user_emb_cal = np.array([user_emb] * self.num_rec_items)
        if self.is_training:
            can_data = list(data_dict[user_id][partition_id])
            ii = np.random.choice(can_data, self.num_rec_items)
            result_items = embs[ii, :]
        else:
            if test_domain == source_type:
                partion_id = max(data_dict[user_id].keys())
                can_data = list(data_dict[user_id][partion_id])
                ii = np.random.choice(can_data, self.num_rec_items)
                result_items = embs[ii, :]
            else:
                if user_id in test_data_dict.keys():
                    can_data = list(test_data_dict[user_id])
                else:
                    can_data = []
                if len(can_data) < self.num_rec_items:
                    partion_id = max(data_dict[user_id].keys())
                    can2 = list(data_dict[user_id][partion_id])
                    can_data = can_data + can2
                ii = np.random.choice(can_data, self.num_rec_items)
                result_items = embs[ii, :]
        # result_items = result_items * user_emb_cal
        return result_items, ii, user_emb

    def init(self, user_id, partition_id, test_domain):
        # print('user_id, partition_id, 1, test_domain = ', user_id, partition_id, 1, test_domain)
        items_1, id_1, user_emb_1 = self.find_items(user_id, partition_id, 1, test_domain)
        # print('user_id, partition_id, 2, test_domain = ', user_id, partition_id, 2, test_domain)
        items_2, id_2, user_emb_2 = self.find_items(user_id, partition_id, 2, test_domain)
        return items_1, items_2, id_1, id_2, user_emb_1, user_emb_2

        # return items_1, items_2, id_1, id_2, user_emb_1, user_emb_2

    def mapping(self, pre_action, domain):
        # candidate item set
        if domain == 1:
            # if valid_item is None:
            #     can_set = copy.deepcopy(self.item_set_1) - set(self.recommended_items)
            # else:
            #     can_set = self.item_set_1
            can_set = copy.deepcopy(self.item_set_1)
            emb = self.item_emb_1
        else:
            # if valid_item is None:
            #     can_set = copy.deepcopy(self.item_set_2) - set(self.recommended_items)
            # else:
            #     can_set = self.item_set_2
            can_set = copy.deepcopy(self.item_set_2)
            emb = self.item_emb_2
            # print('length = ', self.item_emb_2.shape)
            # input('mapping')

        try:
            if self.is_training:
                idx, emb_action = mmr(pre_action, emb, K=self.num_rec_items, can_set=can_set, l=1)
                id_action = idx
            else:
                idx, emb_action = mmr(pre_action, emb, K=self.num_rec_items, can_set=can_set, l=10)
                id_action = idx
            # emb_action = np.resize(emb_action, (self.num_rec_items, self.dim))
        except:
            print('wrong in mapping')
            id_action, emb_action = [], []
        return id_action, emb_action

    def result(self, user_id, id_action, done, domain):
        reward = 0
        cur_done = done
        if domain == 1:
            true_data = self.test_user_dict_1[user_id]
        else:
            true_data = self.test_user_dict_2[user_id]

        if self.data_type == 0:
            all_number = max(1, len(set(self.cur_ratings)))
            for v in id_action:
                if v in true_data:
                    reward += 1
                # if (v in self.cur_ratings) and (v not in self.recommended_items):
                #     reward += 1
                #     item_emb = np.delete(item_emb, 0, 0)
                #     item_emb = np.insert(item_emb, self.num_rec_items-1, emb[v, :], 0)
                # if v not in self.recommended_items:
                #     self.recommended_items.append(v)
        else:
            reward, cur_done, all_number = None, None, None

        cur_done += reward

        return reward, cur_done, all_number
