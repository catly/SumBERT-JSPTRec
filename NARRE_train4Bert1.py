# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner of classification for online prediction. input is a list. output is a label."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import modeling
import tokenization
import tensorflow as tf
import numpy as np

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
BERT_BASE_DIR="./checkpoint_finetuing_law512/"
flags.DEFINE_string("bert_config_file", BERT_BASE_DIR+"bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "sentence_pair", "The name of the task to train.")

flags.DEFINE_string("vocab_file", BERT_BASE_DIR+"vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("init_checkpoint", BERT_BASE_DIR, # model.ckpt-66870--> /model.ckpt-66870
    "Initial checkpoint (usually from a pre-trained BERT model).")
flags.DEFINE_integer("max_seq_length",256 ,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")
import pickle

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)


import datetime
import time

from narre.NARRE4Bert1 import NARRE
from myutils import get_params

np.random.seed(2019)
random_seed = 2019

class NARREExp:

    def __init__(self):
        self.parser = get_params()
        self.args = self.parser.parse_args()

        self.root_data_dir = self.args.root_data_dir
        self.data_name = self.args.dataset
        self.data_dir = self.root_data_dir + '/' + self.data_name

        print("gpu_index", self.args.gpu_index)
        tf.device('/gpu:{}'.format(self.args.gpu_index))

        os.environ['CUDA_VISIBLE_DEVICES']= "1,4,7,8"

        self.model_type = 'narre'
        self.model_dir = self.data_dir + '/' + self.model_type
        self.train_input = self.model_dir + '/' + self.model_type + '.train'
        self.valid_input = self.model_dir + '/' + self.model_type + '.valid'
        self.test_input = self.model_dir + '/' + self.model_type + '.test'
        self.model_para = self.model_dir + '/' + self.model_type + '.para'

        dtime = datetime.datetime.now()
        un_time = int(time.mktime(dtime.timetuple()))
        self.record_file = self.model_dir + '/' + self.model_type + '_' + str(un_time) + '.record'

        self.batch_size = self.args.batch_size
        self.num_epoches = self.args.num_epoches
        self.dropout_keep_prob = self.args.dropout_keep_prob

    def train_step(self, narre, sess,u_ids_batch, i_ids_batch,u_mask_batch,i_mask_batch,u_sgeids_batch,i_segids_batch, u_batch_num, i_batch_num, uid, iid, reuid, reiid, y_batch, batch_num):
        """
        A single training step
        """
        feed_dict = {
            narre.input_u_ids: u_ids_batch,
            narre.input_u_mask: u_mask_batch,
            narre.input_u_segids: u_sgeids_batch,
            narre.input_i_ids: i_ids_batch,
            narre.input_i_mask: i_mask_batch,
            narre.input_i_segids: i_segids_batch,
            narre.input_u_num: u_batch_num,
            narre.input_i_num: i_batch_num,
            narre.input_uid: uid,
            narre.input_iid: iid,
            narre.input_y: y_batch,
            narre.input_reuid: reuid,
            narre.input_reiid: reiid,
            narre.dropout_keep_prob: self.args.dropout_keep_prob
        }
        _, step, loss, rmse, mse, mae, u_a, i_a, fm = sess.run(
            [self.train_op, self.global_step, narre.loss, narre.rmse, narre.mse, narre.mae, narre.u_a, narre.i_a, narre.score],
            feed_dict
        )
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, rmse {:g}, mse {:g}, mae {:g}".format(time_str, batch_num, loss, rmse, mse, mae))
        self.record.write("{}: step {}, loss {:g}, rmse {:g}, mse {:g}, mae {:g}\n".format(time_str, batch_num, loss, rmse, mse, mae))
        return rmse, mse, mae, u_a, i_a, fm

    def dev_step(self, narre, sess, u_ids_batch, i_ids_batch,u_mask_batch,i_mask_batch,u_sgeids_batch,i_segids_batch, u_batch_num, i_batch_num, uid, iid, reuid, reiid, y_batch):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            narre.input_u_ids: u_ids_batch,
            narre.input_u_mask: u_mask_batch,
            narre.input_u_segids: u_sgeids_batch,
            narre.input_i_ids: i_ids_batch,
            narre.input_i_mask: i_mask_batch,
            narre.input_i_segids: i_segids_batch,
            narre.input_u_num: u_batch_num,
            narre.input_i_num: i_batch_num,
            narre.input_y: y_batch,
            narre.input_uid: uid,
            narre.input_iid: iid,
            narre.input_reuid: reuid,
            narre.input_reiid: reiid,
            narre.dropout_keep_prob: 1.0
        }
        step, loss, rmse, mse, mae = sess.run(
            [self.global_step, narre.loss, narre.rmse, narre.mse, narre.mae],
            feed_dict
        )
        return loss, rmse, mse, mae

    def run(self):


        print("Loading data...")
        pkl_file = open(self.model_para, 'rb')
        para = pickle.load(pkl_file)
        user_num = para['user_num']
        item_num = para['item_num']
        review_num_u = para['review_num_u']
        review_num_i = para['review_num_i']
        review_len_u = para['review_len_u']
        review_len_i = para['review_len_i']
        vocab = para['vocab']
        u_text_ids = para['u_text_ids']
        i_text_ids = para['i_text_ids']
        u_text_mask = para['u_text_mask']
        i_text_mask = para['i_text_mask']
        u_text_segids = para['u_text_segids']
        i_text_segids = para['i_text_segids']
        u_text_num = para['u_text_num']
        i_text_num = para['i_text_num']

        print("user_num", user_num)
        print("item_num", item_num)
        print("review_num_u", review_num_u)
        print("review_len_u", review_len_u)
        print("review_num_i", review_num_i)
        print("review_len_i", review_len_i)

        epoch_stop_loss = 99999
        epoch_stopping = 5

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False
            )
            # session_conf.gpu_options.allow_growth = True
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                narre = NARRE(
                    args = self.args,
                    review_num_u=review_num_u,
                    review_num_i=review_num_i,
                    review_len_u=review_len_u,
                    review_len_i=review_len_i,
                    user_num=user_num,
                    item_num=item_num,
                    vocab_size=len(vocab),
                    n_latent=self.args.narre_n_latent,
                    embedding_id=self.args.narre_embedding_id,
                    attention_size=self.args.narre_attention_size,
                    embedding_size=self.args.embed_dim,
                    filter_sizes=list(map(int,self.args.narre_filter_sizes.split(","))),
                    num_filters=self.args.narre_num_filters,
                    l2_reg_lambda=self.args.l2_reg_lambda
                )
                tf.set_random_seed(random_seed)
                print(user_num)
                print(item_num)

                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                optimizer = tf.train.AdamOptimizer(self.args.lr, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(narre.loss)
                self.train_op = optimizer
                sess.run(tf.initialize_all_variables())
                # saver = tf.train.Saver()

                best_valid_loss = 99999
                best_valid_mae = 5
                best_valid_rmse = 5
                best_valid_mse = 5

                best_test_loss = 99999
                best_test_mae = 5
                best_test_rmse = 5
                best_test_mse = 5

                pkl_file = open(self.train_input, 'rb')
                train_data = pickle.load(pkl_file)
                train_data = np.array(train_data)
                pkl_file.close()

                pkl_file = open(self.valid_input, 'rb')
                valid_data = pickle.load(pkl_file)
                valid_data = np.array(valid_data)
                pkl_file.close()

                pkl_file = open(self.test_input, 'rb')
                test_data = pickle.load(pkl_file)
                test_data = np.array(test_data)
                pkl_file.close()

                data_size_train = len(train_data)
                data_size_valid = len(valid_data)
                data_size_test = len(test_data)

                l_train = int(len(train_data) / self.batch_size)

                self.record = open(self.record_file, 'w+')
                self.record.write(str(self.args) + '\n')

                for epoch in range(1, self.num_epoches + 1, 1):
                    # Shuffle the data at each epoch
                    shuffle_indices = np.random.permutation(np.arange(data_size_train))
                    shuffled_data = train_data[shuffle_indices]

                    for batch_num in range(l_train):

                        start_index = batch_num * self.batch_size
                        end_index = min((batch_num + 1) * self.batch_size, data_size_train)
                        data_train = shuffled_data[start_index: end_index]

                        uid, iid, reuid, reiid, y_batch = zip(*data_train)
                        u_ids_batch = []
                        i_ids_batch = []
                        u_mask_batch = []
                        i_mask_batch = []
                        u_segids_batch = []
                        i_segids_batch = []
                        u_batch_num = []
                        i_batch_num = []
                        for i in range(len(uid)):
                            u_ids_batch.append(u_text_ids[uid[i][0]])
                            i_ids_batch.append(i_text_ids[iid[i][0]])
                            u_mask_batch.append(u_text_mask[uid[i][0]])
                            i_mask_batch.append(i_text_mask[iid[i][0]])
                            u_segids_batch.append(u_text_segids[uid[i][0]])
                            i_segids_batch.append(i_text_segids[iid[i][0]])
                            u_batch_num.append(u_text_num[uid[i][0]])
                            i_batch_num.append(i_text_num[iid[i][0]])
                        u_ids_batch = np.array(u_ids_batch)
                        i_ids_batch = np.array(i_ids_batch)
                        u_mask_batch = np.array(u_mask_batch)
                        i_mask_batch = np.array(i_mask_batch)
                        u_segids_batch = np.array(u_segids_batch)
                        i_segids_batch = np.array(i_segids_batch)
                        u_batch_num = np.array(u_batch_num)
                        i_batch_num = np.array(i_batch_num)

                        self.train_step(narre, sess, u_ids_batch, i_ids_batch,u_mask_batch,i_mask_batch,u_segids_batch,i_segids_batch, u_batch_num, i_batch_num, uid, iid, reuid, reiid, y_batch, batch_num)
                        current_step = tf.train.global_step(sess, self.global_step)

                        if batch_num % self.args.evaluate_every == 0 and batch_num > 1:
                            print("\nEvaluation valid:")
                            self.record.write("\nEvaluation valid:\n")

                            loss_s = 0.0
                            rmse_s = 0.0
                            mae_s = 0.0
                            mse_s = 0.0


                            ll_valid = int(len(valid_data) / self.batch_size) + 1
                            for batch_num_valid in range(ll_valid):
                                start_index = batch_num_valid * self.batch_size
                                end_index = min((batch_num_valid + 1) * self.batch_size, data_size_valid)
                                data_valid = valid_data[start_index: end_index]

                                userid_valid, itemid_valid, reuid, reiid, y_valid = zip(*data_valid)
                                u_ids_valid = []
                                i_ids_valid = []
                                u_mask_valid = []
                                i_mask_valid = []
                                u_segids_valid = []
                                i_segids_valid = []
                                u_valid_num = []
                                i_valid_num = []
                                for i in range(len(userid_valid)):
                                    u_ids_valid.append(u_text_ids[userid_valid[i][0]])
                                    i_ids_valid.append(i_text_ids[itemid_valid[i][0]])
                                    u_mask_valid.append(u_text_ids[userid_valid[i][0]])
                                    i_mask_valid.append(i_text_ids[itemid_valid[i][0]])
                                    u_segids_valid.append(u_text_ids[userid_valid[i][0]])
                                    i_segids_valid.append(i_text_ids[itemid_valid[i][0]])
                                    u_valid_num.append(u_text_num[userid_valid[i][0]])
                                    i_valid_num.append(i_text_num[itemid_valid[i][0]])
                                u_ids_valid = np.array(u_ids_valid)
                                i_ids_valid = np.array(i_ids_valid)
                                u_mask_valid = np.array(u_mask_valid)
                                i_mask_valid = np.array(i_mask_valid)
                                u_segids_valid = np.array(u_segids_valid)
                                i_segids_valid = np.array(i_segids_valid)
                                u_valid_num = np.array(u_valid_num)
                                i_valid_num = np.array(i_valid_num)

                                loss, rmse, mse, mae = self.dev_step(narre, sess, u_ids_valid, i_ids_valid,u_mask_valid, i_mask_valid ,u_segids_valid,i_segids_valid, u_valid_num, i_valid_num, userid_valid, itemid_valid, reuid, reiid, y_valid)

                                loss_s = loss_s + len(u_ids_valid) * loss
                                rmse_s = rmse_s + len(u_ids_valid) * np.square(rmse)
                                mae_s = mae_s + len(u_ids_valid) * mae
                                mse_s = mse_s + len(u_ids_valid) * mse

                            print("loss_valid {:g}, rmse_valid {:g}, mse_valid {:g}, mae_valid {:g}".format(loss_s / len(valid_data),
                                                                                            np.sqrt(rmse_s / len(
                                                                                                valid_data)),
                                                                                            mse_s / len(valid_data),
                                                                                            mae_s / len(valid_data)))
                            self.record.write("loss_valid {:g}, rmse_valid {:g}, mse_valid {:g}, mae_valid {:g}\n".format(loss_s / len(valid_data),
                                                                                            np.sqrt(rmse_s / len(
                                                                                                valid_data)),
                                                                                            mse_s / len(valid_data),
                                                                                            mae_s / len(valid_data)))
                            rmse = np.sqrt(rmse_s / len(valid_data))
                            mae = mae_s / len(valid_data)
                            loss_s = loss_s / len(valid_data)
                            mse = mse_s / len(valid_data)

                            if best_valid_loss > loss_s:
                                best_valid_loss = loss_s
                            if best_valid_rmse > rmse:
                                best_valid_rmse = rmse
                            if best_valid_mae > mae:
                                best_valid_mae = mae
                            if best_valid_mse > mse:
                                best_valid_mse = mse

                            print("best_valid_loss {:g}, best_valid_rmse {:g}, best_valid_mse {:g}, best_valid_mae {:g}".format(best_valid_loss, best_valid_rmse, best_valid_mse, best_valid_mae))
                            self.record.write("best_valid_loss {:g}, best_valid_rmse {:g}, best_valid_mse {:g}, best_valid_mae {:g}\n".format(best_valid_loss, best_valid_rmse, best_valid_mse, best_valid_mae))
                            print('===================================================================')
                            self.record.write('===================================================================\n')

                        if batch_num % self.args.evaluate_every == 0 and batch_num > 1:
                            print("\nEvaluation test:")
                            self.record.write("\nEvaluation test:\n")

                            loss_s = 0.0
                            rmse_s = 0.0
                            mae_s = 0.0
                            mse_s = 0.0

                            ll_test = int(len(test_data) / self.batch_size) + 1
                            for batch_num_test in range(ll_test):
                                start_index = batch_num_test * self.batch_size
                                end_index = min((batch_num_test + 1) * self.batch_size, data_size_test)
                                data_test = test_data[start_index: end_index]

                                userid_test, itemid_test, reuid, reiid, y_test = zip(*data_test)
                                u_ids_test = []
                                i_ids_test = []
                                u_mask_test = []
                                i_mask_test = []
                                u_segids_test = []
                                i_segids_test = []
                                u_test_num = []
                                i_test_num = []
                                for i in range(len(userid_test)):
                                    u_ids_test.append(u_text_ids[userid_test[i][0]])
                                    i_ids_test.append(i_text_ids[itemid_test[i][0]])
                                    u_mask_test.append(u_text_mask[userid_test[i][0]])
                                    i_mask_test.append(i_text_mask[itemid_test[i][0]])
                                    u_segids_test.append(u_text_segids[userid_test[i][0]])
                                    i_segids_test.append(i_text_segids[itemid_test[i][0]])
                                    u_test_num.append(u_text_num[userid_test[i][0]])
                                    i_test_num.append(i_text_num[itemid_test[i][0]])
                                u_ids_test = np.array(u_ids_test)
                                i_ids_test = np.array(i_ids_test)
                                u_mask_test = np.array(u_mask_test)
                                i_mask_test = np.array(i_mask_test)
                                u_segids_test = np.array(u_segids_test)
                                i_segids_test = np.array(i_segids_test)
                                u_test_num = np.array(u_test_num)
                                i_test_num = np.array(i_test_num)

                                loss, rmse, mse, mae = self.dev_step(narre, sess, u_ids_test, i_ids_test,  u_mask_test, i_mask_test, u_segids_test, i_segids_test,u_test_num, i_test_num, userid_test, itemid_test, reuid, reiid, y_test)

                                loss_s = loss_s + len(u_ids_test) * loss
                                rmse_s = rmse_s + len(u_ids_test) * np.square(rmse)
                                mae_s = mae_s + len(u_ids_test) * mae
                                mse_s = mse_s + len(u_ids_test) * mse

                            print("loss_test {:g}, rmse_test {:g}, mse_test {:g}, mae_test {:g}".format(loss_s / len(test_data),
                                                                                            np.sqrt(rmse_s / len(
                                                                                                test_data)),
                                                                                            mse_s / len(test_data),
                                                                                            mae_s / len(test_data)))

                            self.record.write("loss_test {:g}, rmse_test {:g}, mse_test {:g}, mae_test {:g}\n".format(loss_s / len(test_data),
                                                                                            np.sqrt(rmse_s / len(
                                                                                                test_data)),
                                                                                            mse_s / len(test_data),
                                                                                            mae_s / len(test_data)))
                            rmse = np.sqrt(rmse_s / len(test_data))
                            mae = mae_s / len(test_data)
                            loss_s = loss_s / len(test_data)
                            mse = mse_s / len(test_data)
                            if best_test_loss > loss_s:
                                best_test_loss = loss_s
                            if best_test_rmse > rmse:
                                best_test_rmse = rmse
                            if best_test_mae > mae:
                                best_test_mae = mae
                            if best_test_mse > mse:
                                best_test_mse = mse

                            print("best_test_loss {:g}, best_test_rmse {:g}, best_test_mse {:g}, best_test_mae {:g}".format(
                                best_test_loss, best_test_rmse, best_test_mse, best_test_mae))
                            self.record.write("best_test_loss {:g}, best_test_rmse {:g}, best_test_mse {:g}, best_test_mae {:g}\n".format(
                                    best_test_loss, best_test_rmse, best_test_mse, best_test_mae))
                            print('===================================================================')
                            self.record.write('===================================================================\n')

                    print("epoch " + str(epoch) + ":")
                    self.record.write("epoch " + str(epoch) + ":\n")
                    print("best_valid_loss {:g}, best_valid_rmse {:g}, best_valid_mse {:g}, best_valid_mae {:g}".format(best_valid_loss,
                    best_valid_rmse, best_valid_mse, best_valid_mae))
                    print("best_test_loss {:g}, best_test_rmse {:g}, best_valid_mse {:g}, best_test_mae {:g}".format(best_test_loss,
                    best_test_rmse, best_test_mse, best_test_mae))
                    print('===================================================================')
                    self.record.write("best_valid_loss {:g}, best_valid_rmse {:g}, best_valid_mse {:g}, best_valid_mae {:g}\n".format(best_valid_loss,
                     best_valid_rmse, best_valid_mse, best_valid_mae))
                    self.record.write("best_test_loss {:g}, best_test_rmse {:g}, best_test_mse {:g}, best_test_mae {:g}\n".format(best_test_loss,
                     best_test_rmse, best_test_mse, best_test_mae))
                    self.record.write('===================================================================\n')

                    # early stopping
                    if best_valid_loss < epoch_stop_loss:
                        epoch_stop_loss = best_valid_loss
                        epoch_stopping = 5
                    else:
                        epoch_stopping = epoch_stopping - 1
                    if epoch_stopping <= 0:
                        break
                self.record.close()

if __name__ == '__main__':
    ex = NARREExp()
    ex.run()

