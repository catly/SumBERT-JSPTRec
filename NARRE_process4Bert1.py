#coding=utf-8

import numpy as np
import re
import itertools
from collections import Counter
import tokenization
import tensorflow as tf
import csv
import os
import pickle
import pandas as pd

import sys
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from myutils import get_params

def clean_str(string):

    """
        Tokenization / string cleaning for all datasets except for SST.
    """
    string = re.sub(r"\d{1,}", " number ", string)
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " have", string)
    string = re.sub(r"n'\t", " not", string)
    string = re.sub(r"\'re", " are", string)
    string = re.sub(r"\'d", " would", string)
    string = re.sub(r"\'ll", " will", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

class NARREProcess:

    def __init__(self):

        self.parser = get_params()
        self.args = self.parser.parse_args()
        self.model_type = 'narre'

        self.root_data_dir = self.args.root_data_dir
        self.data_name = self.args.dataset
        self.data_dir = self.root_data_dir + '/' + self.data_name

        self.train_file = os.path.join(self.data_dir, self.data_name + '_train.csv')
        self.valid_file = os.path.join(self.data_dir, self.data_name + '_valid.csv')
        self.test_file = os.path.join(self.data_dir, self.data_name + '_test.csv')
        self.user_reviews_file = os.path.join(self.data_dir, 'user_review')
        self.item_reviews_file = os.path.join(self.data_dir, 'item_review')
        self.user_rids_file = os.path.join(self.data_dir, 'user_rid')
        self.item_rids_file = os.path.join(self.data_dir, 'item_rid')
        self.vocab_file_bert = os.path.join(self.data_dir, 'vocab.txt')
        self.vocab_file = os.path.join(self.data_dir, 'vocab.pk')

        self.u_max_num = self.args.u_max_num
        self.u_max_len = self.args.u_max_len
        self.i_max_num = self.args.i_max_num
        self.i_max_len = self.args.i_max_len

        self.model_dir = self.data_dir + '/' + self.model_type
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.train_input = os.path.join(self.model_dir, self.model_type + '.train')
        self.valid_input = os.path.join(self.model_dir, self.model_type + '.valid')
        self.test_input = os.path.join(self.model_dir, self.model_type + '.test')
        self.model_para = os.path.join(self.model_dir, self.model_type  + '.para')

    def convert_bert_input(self,max_seq_length,
                               tokenizer, text_a, text_b=None):
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = None
        if text_b:
            tokens_b = tokenizer.tokenize(text_b)  # 这里主要是将中文分字
        if tokens_b:
            # 如果有第二个句子，那么两个句子的总长度要小于 max_seq_length - 3
            # 因为要为句子补上[CLS], [SEP], [SEP]
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # 如果只有一个句子，只用在前后加上[CLS], [SEP] 所以句子长度要小于 max_seq_length - 3
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # 转换成bert的输入，注意下面的type_ids 在源码中对应的是 segment_ids
        # (a) 两个句子:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) 单个句子:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # 这里 "type_ids" 主要用于区分第一个第二个句子。
        # 第一个句子为0，第二个句子是1。在余训练的时候会添加到单词的的向量中，但这个不是必须的
        # 英文[SEP] 已经区分了第一个句子和第二个句子。但type_ids 会让学习变的简单

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)
        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)  # 将中文转换成ids
        # 创建mask
        input_mask = [1] * len(input_ids)
        # 对于输入进行补0
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        return input_ids, input_mask, segment_ids
    

    def pad_sentences(self, u_text, u_len, u2_len):

        """
        Pads all sentences to teh same length. The length is defined by the longest sentence.
        """
        review_num = u_len
        review_len = u2_len
        tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab_file_bert, do_lower_case=True)

        u_text2_ids = {}
        u_text2_mask ={}
        u_text2_segids = {}
        u_text_num = {}
        for i in u_text.keys():
            u_reviews = u_text[i]
            padded_u_train_ids = []
            padded_u_train_mask = []
            padded_u_train_segids = []
            u_text_num[i] = min(len(u_reviews), review_num)
            for ri in range(review_num):
                if ri < len(u_reviews):
                    sentence = u_reviews[ri]
                    ids,mask,segids=self.convert_bert_input(max_seq_length=review_len,tokenizer=tokenizer,text_a=" ".join(sentence))
                    padded_u_train_ids.append(ids)
                    padded_u_train_mask.append(mask)
                    padded_u_train_segids.append(segids)

                else:
                    sentence=[]
                    ids,mask,segids=self.convert_bert_input(max_seq_length=review_len,tokenizer=tokenizer,text_a=" ".join(sentence))
                    padded_u_train_ids.append(ids)
                    padded_u_train_mask.append(mask)
                    padded_u_train_segids.append(segids)
            u_text2_ids[i] = padded_u_train_ids
            u_text2_mask[i] = padded_u_train_mask
            u_text2_segids[i] = padded_u_train_segids
        return u_text2_ids,u_text2_mask,u_text2_segids, u_text_num

    def pad_reviewid(self, u_train, u_len, num):

        """
        num is the padding id
        """
        pad_u_train = []
        for i in range(len(u_train)):
            x = u_train[i]
            while u_len > len(x):
                x.append(num)
            if u_len < len(x):
                x = x[:u_len]
            pad_u_train.append(x)
        return pad_u_train

    def build_input_data(self, u_text, i_text, vocab):

        """
        Maps sentence and labels to vectors based on a vocabulary

        """
        u_text2 = {}
        for i in u_text.keys():
            u_reviews = u_text[i]
            u = np.array([[vocab[word] for word in words] for words in u_reviews])
            u_text2[i] = u

        i_text2 = {}
        for j in i_text.keys():
            i_reviews = i_text[j]
            i = np.array([[vocab[word] for word in words] for words in i_reviews])
            i_text2[j] = i
        return u_text2, i_text2

    def load_data_and_labels(self):

        """
            Loads data from files, splits the data into words and generate labels
        """

        # Load data from file
        f1 = open(self.user_reviews_file, "rb")
        f2 = open(self.item_reviews_file, "rb")
        f3 = open(self.user_rids_file, "rb")
        f4 = open(self.item_rids_file, "rb")

        user_reviews = pickle.load(f1)
        item_reviews = pickle.load(f2)
        user_rids = pickle.load(f3)
        item_rids = pickle.load(f4)

        print(" ====================== train =========================")
        f_train = pd.read_csv(self.train_file)
        train_user_ids = f_train["user_id"]
        train_item_ids = f_train["item_id"]
        train_ratings = f_train["ratings"]
        train_reviews = f_train["reviews"]

        train_length = len(train_user_ids)

        uid_train = []
        iid_train = []
        reid_user_train = []
        reid_item_train = []
        y_train = []
        reviews_train = []

        u_text = {}
        i_text = {}
        u_rid = {}
        i_rid = {}
        review_list = []

        for i in range(train_length):

            user_id = int(train_user_ids[i])
            item_id = int(train_item_ids[i])
            rating = int(train_ratings[i])
            review = str(train_reviews[i])
            review_list.append(review)

            uid_train.append(user_id)
            iid_train.append(item_id)
            y_train.append(float(rating))
            reviews_train.append(review)

            # add user id
            if user_id in u_text:
                reid_user_train.append(u_rid[user_id])

            else:
                u_text[user_id] = []
                for s in user_reviews[user_id]:
                    s1 = clean_str(s)
                    s1 = s1.split(" ")
                    u_text[user_id].append(s1)
                u_rid[user_id] = []
                for s in user_rids[user_id]:
                    u_rid[user_id].append(int(s))
                reid_user_train.append(u_rid[user_id])

            # add item id
            if item_id in i_text:
                reid_item_train.append(i_rid[item_id])

            else:
                i_text[item_id] = []
                for s in item_reviews[item_id]:
                    s1 = clean_str(s)
                    s1 = s1.split(" ")
                    i_text[item_id].append(s1)
                i_rid[item_id] = []
                for s in item_rids[item_id]:
                    i_rid[item_id].append(int(s))
                reid_item_train.append(i_rid[item_id])

        print("len of uid_train", len(uid_train))
        print("len of iid_train", len(iid_train))
        print("len of review_y_train", len(y_train))
        print("len of reviews_train", len(reviews_train))

        print("======================= valid =========================")
        f_valid = pd.read_csv(self.valid_file)
        valid_user_ids = f_valid["user_id"]
        valid_item_ids = f_valid["item_id"]
        valid_ratings = f_valid["ratings"]
        valid_reviews = f_valid["reviews"]
        valid_length = len(valid_user_ids)

        uid_valid = []
        iid_valid = []
        reid_user_valid = []
        reid_item_valid = []
        y_valid = []
        reviews_valid = []

        for i in range(valid_length):
            user_id = int(valid_user_ids[i])
            item_id = int(valid_item_ids[i])
            rating = int(valid_ratings[i])
            review = valid_reviews[i]

            review_list.append(review)
            uid_valid.append(user_id)
            iid_valid.append(item_id)
            y_valid.append(float(rating))
            reviews_valid.append(review)

            if user_id in u_text:
                reid_user_valid.append(u_rid[user_id])
            else:
                u_text[user_id] = [['number']]
                u_rid[user_id] = [int(0)]
                reid_user_valid.append(u_rid[user_id])
            if item_id in i_text:
                reid_item_valid.append(i_rid[item_id])
            else:
                i_text[item_id] = [['number']]
                i_rid[item_id] = [int(0)]
                reid_item_valid.append(i_rid[item_id])

        print("len of uid_valid", len(uid_valid))
        print("len of iid_valid", len(iid_valid))
        print("len of review_y_valid", len(y_valid))
        print("len of reviews_valid", len(reviews_valid))

        print("====================== test ==========================")
        f_test = pd.read_csv(self.test_file)
        test_user_ids = f_test["user_id"]
        test_item_ids = f_test["item_id"]
        test_ratings = f_test["ratings"]
        test_reviews = f_test["reviews"]
        test_length = len(test_user_ids)

        uid_test = []
        iid_test = []
        reid_user_test = []
        reid_item_test = []
        y_test = []
        reviews_test = []

        for i in range(test_length):

            user_id = int(test_user_ids[i])
            item_id = int(test_item_ids[i])
            rating = int(test_ratings[i])
            review = test_reviews[i]

            review_list.append(review)
            uid_test.append(user_id)
            iid_test.append(item_id)
            y_test.append(float(rating))
            reviews_test.append(review)

            if user_id in u_text:
                reid_user_test.append(u_rid[user_id])
            else:
                u_text[user_id] = [['number']]
                u_rid[user_id] = [int(0)]
                reid_user_test.append(u_rid[user_id])
            if item_id in i_text:
                reid_item_test.append(i_rid[item_id])

            else:
                i_text[item_id] = [['number']]
                i_rid[item_id] = [int(0)]
                reid_item_test.append(i_rid[item_id])

        print("len of uid_test", len(uid_test))
        print("len of iid_test", len(iid_test))
        print("len of review_y_test", len(y_test))
        print("len of reviews_test", len(reviews_test))

        uid_train, uid_valid, uid_test = np.array(uid_train), np.array(uid_valid), np.array(uid_test)
        iid_train, iid_valid, iid_test = np.array(iid_train), np.array(iid_valid), np.array(iid_test)
        reid_user_train, reid_user_valid, reid_user_test = np.array(reid_user_train), np.array(reid_user_valid), np.array(reid_user_test)
        reid_item_train, reid_item_valid, reid_item_test = np.array(reid_item_train), np.array(reid_item_valid), np.array(reid_item_test)
        y_train, y_valid, y_test = np.array(y_train), np.array(y_valid), np.array(y_test)
        reviews_train, reviews_valid, reviews_test = np.array(reviews_train), np.array(reviews_valid), np.array(reviews_test)

        train_set = list(
            zip(uid_train, iid_train, reid_user_train, reid_item_train, y_train)
        )
        valid_set = list(
            zip(uid_valid, iid_valid, reid_user_valid, reid_item_valid, y_valid)
        )
        test_set = list(
            zip(uid_test, iid_test, reid_user_test, reid_item_test, y_test)
        )
        print("================= data sample ================")
        print("len of review_list", len(review_list))
        print("len of training set", train_length)
        print("len of valid set", valid_length)
        print("len of test set", test_length)

        return u_text, i_text, u_rid, i_rid, train_set, valid_set, test_set, train_length, valid_length, test_length

    def get_padding_len(self, u_text, i_text):

        print("=================== parameters =====================")
        # user_rating_review
        review_num_u_list = np.array([len(x) for x in u_text.values()])
        x_u = np.sort(review_num_u_list)
        review_num_u = x_u[int(0.6 * len(review_num_u_list)) - 1]

        review_len_u_list = np.array([len(j) for i in u_text.values() for j in i])
        x2_u = np.sort(review_len_u_list)
        review_len_u = x2_u[int(0.6 * len(review_len_u_list)) - 1]

        # item_rating_review
        review_num_i_list = np.array([len(x) for x in i_text.values()])
        x_i = np.sort(review_num_i_list)
        review_num_i = x_i[int(0.6 * len(review_num_i_list)) - 1]

        review_len_i_list = np.array([len(j) for i in i_text.values() for j in i])
        x2_i = np.sort(review_len_i_list)
        review_len_i = x2_i[int(0.6 * len(review_len_i_list)) - 1]

        user_num = len(u_text)
        item_num = len(i_text)

        print("review_num_u", review_num_u)
        print("review_len_u", review_len_u)
        print("review_num_i", review_num_i)
        print("review_len_i", review_len_i)
        print("number of users", user_num)
        print("number of items", item_num)

        return review_num_u, review_len_u, review_num_i, review_len_i

    def run(self):

        # load data and labels
        u_text, i_text, u_rid, i_rid, train_set, valid_set, test_set, train_length, valid_length, test_length = self.load_data_and_labels()
        user_num = len(u_text)
        item_num = len(i_text)

        # get padding
        self.u_max_num = self.args.u_max_num
        self.u_max_len = self.args.u_max_len
        self.i_max_num = self.args.i_max_num
        self.i_max_len = self.args.i_max_len

        # get padding parameters
        if self.u_max_num < 1 or self.u_max_len < 1 or self.i_max_num < 1 or self.i_max_len:
            self.u_max_num, self.u_max_len, self.i_max_num, self.i_max_len = self.get_padding_len(u_text, i_text)

        # dataset samples
        uid_train, iid_train, reid_user_train, reid_item_train, y_train = zip(*train_set)
        uid_valid, iid_valid, reid_user_valid, reid_item_valid, y_valid = zip(*valid_set)
        uid_test, iid_test, reid_user_test, reid_item_test, y_test = zip(*test_set)

        # pad user review
        print("pad user")
        u_text_ids,u_text_mask,u_text_segids, u_text_num = self.pad_sentences(u_text, self.u_max_num, self.u_max_len)
        reid_u_train = self.pad_reviewid(reid_user_train, self.u_max_num, item_num + 1)
        reid_u_valid = self.pad_reviewid(reid_user_valid, self.u_max_num, item_num + 1)
        reid_u_test = self.pad_reviewid(reid_user_test, self.u_max_num, item_num + 1)

        # padding item review
        i_text_ids,i_text_mask,i_text_segids, i_text_num = self.pad_sentences(i_text, self.i_max_num, self.i_max_len)
        reid_i_train = self.pad_reviewid(reid_item_train, self.i_max_num, user_num + 1)
        reid_i_valid = self.pad_reviewid(reid_item_valid, self.i_max_num, user_num + 1)
        reid_i_test = self.pad_reviewid(reid_item_test, self.i_max_num, user_num + 1)

        # vocabulary
        vocab = pickle.load(open(self.vocab_file, 'rb'))
        print('len of vocabulary: {}'.format(len(vocab)))
        print("build vocabulary index")
        u_text, i_text = self.build_input_data(u_text, i_text, vocab)

        # to array
        y_train, y_valid, y_test = np.array(y_train), np.array(y_valid), np.array(y_test)
        uid_train, uid_valid, uid_test = np.array(uid_train), np.array(uid_valid), np.array(uid_test)
        iid_train, iid_valid, iid_test = np.array(iid_train), np.array(iid_valid), np.array(iid_test)
        print("iid_valid->", np.shape(iid_valid))

        reid_u_train, reid_u_valid, reid_u_test = np.array(reid_u_train), np.array(reid_u_valid), np.array(reid_u_test)
        reid_i_train, reid_i_valid, reid_i_test = np.array(reid_i_train), np.array(reid_i_valid), np.array(reid_i_test)

        y_train = y_train[:, np.newaxis]  # shape: (train_length, 1)
        y_valid = y_valid[:, np.newaxis]  # shape: (valid_length, 1)
        y_test = y_test[:, np.newaxis]  # shape: (test_length, 1)

        uid_train, uid_valid, uid_test = uid_train[:, np.newaxis], uid_valid[:, np.newaxis], uid_test[:, np.newaxis]
        iid_train, iid_valid, iid_test = iid_train[:, np.newaxis], iid_valid[:, np.newaxis], iid_test[:, np.newaxis]

        print("==================== shape =================")

        print("y_train {}, y_valid {}, y_test {}".format(np.shape(y_train), np.shape(y_valid), np.shape(y_test)))
        print("uid_train {}, uid_valid {}, uid_test {}".format(np.shape(uid_train), np.shape(uid_valid), np.shape(uid_test)))
        print("iid_train {}, iid_valid {}, iid_test {}".format(np.shape(iid_train), np.shape(iid_valid), np.shape(iid_test)))
        print("reid_u_train {}, reid_u_valid {}, reid_u_test {}".format(np.shape(reid_u_train), np.shape(reid_u_valid), np.shape(reid_u_test)))
        print("reid_i_train {}, reid_i_valid {}, reid_i_test {}".format(np.shape(reid_i_train), np.shape(reid_i_valid), np.shape(reid_i_test)))

        print("======================== write begin ==============================")
        batches_train = list(
            zip(uid_train, iid_train, reid_u_train, reid_i_train, y_train)
        )
        batches_valid = list(
            zip(uid_valid, iid_valid, reid_u_valid, reid_i_valid, y_valid)
        )
        batches_test = list(
            zip(uid_test, iid_test, reid_u_test, reid_i_test, y_test)
        )
        output = open(self.train_input, 'wb')
        pickle.dump(batches_train, output)
        output = open(self.valid_input, 'wb')
        pickle.dump(batches_valid, output)
        output = open(self.test_input, 'wb')
        pickle.dump(batches_test, output)

        para = {}
        para['user_num'] = user_num
        para['item_num'] = item_num
        para['review_num_u'] = self.u_max_num
        para['review_num_i'] = self.i_max_num
        para['review_len_u'] = self.u_max_len
        para['review_len_i'] = self.i_max_len
        para['vocab'] = vocab
        para['train_length'] = train_length
        para['valid_length'] = valid_length
        para['test_length'] = test_length
        para['u_text_ids'] = u_text_ids
        para['i_text_ids'] = i_text_ids
        para['u_text_mask'] = u_text_mask
        para['i_text_mask'] = i_text_mask
        para['u_text_segids'] = u_text_segids
        para['i_text_segids'] = i_text_segids
        para['u_text_num'] = u_text_num
        para['i_text_num'] = i_text_num

        output = open(self.model_para, 'wb')
        pickle.dump(para, output)
        print("write done!"+str(self.u_max_len)+" "+str(self.i_max_len))


if __name__ == '__main__':
    pro = NARREProcess()
    pro.run()