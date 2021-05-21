# coding=utf-8



import tensorflow as tf
import pickle
import modeling

def Mask(inputs, seq_len=None, max_len=None, mode='mul'):
    if seq_len == None:
        return inputs
    else:
        mask = tf.cast(tf.sequence_mask(seq_len, maxlen=max_len), tf.float32)
        for _ in range(len(inputs.shape) - 2):
            mask = tf.expand_dims(mask, 2)
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12

class NARRE(object):

    def __init__(self, args, review_num_u, review_num_i, review_len_u, review_len_i, user_num, item_num,
            vocab_size, n_latent, embedding_id, attention_size,
            embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        #self.input_u = tf.placeholder(tf.int32, [None, review_num_u, review_len_u], name="input_u")
        #self.input_i = tf.placeholder(tf.int32, [None, review_num_i, review_len_i], name="input_i")
        self.input_reuid = tf.placeholder(tf.int32, [None, review_num_u], name="input_reuid")
        self.input_reiid = tf.placeholder(tf.int32, [None, review_num_i], name="input_reiid")
        self.input_y = tf.placeholder(tf.float32, [None, 1], name="input_y")
        self.input_uid = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.input_iid = tf.placeholder(tf.int32, [None, 1], name="input_iid")

        self.input_u_ids = tf.placeholder(tf.int32, [None, review_num_u, review_len_u], name="input_u_ids")
        self.input_i_ids = tf.placeholder(tf.int32, [None, review_num_i, review_len_i], name="input_i_ids")
        self.input_u_mask = tf.placeholder(tf.int32, [None, review_num_u, review_len_u], name="input_u_mask")
        self.input_i_mask = tf.placeholder(tf.int32, [None, review_num_i, review_len_i], name="input_i_mask")
        self.input_u_segids = tf.placeholder(tf.int32, [None, review_num_u, review_len_u], name="input_u_segids")
        self.input_i_segids = tf.placeholder(tf.int32, [None, review_num_i, review_len_i], name="input_i_segids")

        self.input_u_num = tf.placeholder(tf.int32, [None, ], name="input_u_num")
        self.input_i_num = tf.placeholder(tf.int32, [None, ], name="input_i_num")
        self.Bert_config=modeling.BertConfig.from_json_file("/users4/yli/RR/uncased_L-12_H-768_A-12/bert_config.json")

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        iidW = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.01, 0.01), name="iidW")
        uidW = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.01, 0.01), name="uidW")

        l2_loss = tf.constant(0.0)

        with tf.name_scope("Bert_user"):
            model = modeling.BertModel(
                config=self.Bert_config,
                is_training=True,
                input_ids=tf.reshape(self.input_u_ids,[-1,review_len_u]),
                input_mask=tf.reshape(self.input_u_mask,[-1,review_len_u]),
                token_type_ids=tf.reshape(self.input_u_segids,[-1,review_len_u]),
                use_one_hot_embeddings=False
            )
            output_u = tf.reshape(model.get_pooled_output(),[-1,review_num_u, review_len_u,768])
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool_u = tf.concat(output_u, 3)
            self.h_pool_flat_u = tf.reshape(self.h_pool_u, [-1, review_num_u, num_filters_total])

        with tf.name_scope("Bert_item"):
            model = modeling.BertModel(
                config=self.Bert_config,
                is_training=True,
                input_ids=tf.reshape(self.input_i_ids,[-1,review_len_i]),
                input_mask=tf.reshape(self.input_i_mask,[-1,review_len_i]),
                token_type_ids=tf.reshape(self.input_i_segids,[-1,review_len_i]),
                use_one_hot_embeddings=False
            )
            output_i = tf.reshape(model.get_pooled_output(),[-1,review_num_i, review_len_i,768])
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool_i = tf.concat(output_i, 3)
            self.h_pool_flat_i = tf.reshape(self.h_pool_i, [-1, review_num_i, num_filters_total])

        with tf.name_scope("dropout"):
            self.h_drop_u = tf.nn.dropout(self.h_pool_flat_u, 1.0)
            self.h_drop_i = tf.nn.dropout(self.h_pool_flat_i, 1.0)
            print("h_drop_u", self.h_drop_u)

        with tf.name_scope("attention"):

            # User attention
            Wau = tf.Variable(
                tf.random_uniform([num_filters_total, attention_size], -0.01, 0.01),
                name="Wau"
            )
            Wru = tf.Variable(
                tf.random_uniform([embedding_id, attention_size], -0.01, 0.01),
                name="Wru"
            )
            Wpu = tf.Variable(
                tf.random_uniform([attention_size, 1], -0.01, 0.01),
                name="Wpu"
            )
            bau = tf.Variable(
                tf.constant(0.1, shape=[attention_size]),
                name="bau"
            )
            bbu = tf.Variable(
                tf.constant(0.1, shape=[1]),
                name="bbu"
            )

            self.iid_a = tf.nn.relu(tf.nn.embedding_lookup(iidW, self.input_reuid))
            self.u_j = tf.einsum('ajk,kl->ajl', tf.nn.relu(
                tf.einsum('ajk,kl->ajl', self.h_drop_u, Wau) + tf.einsum('ajk,kl->ajl', self.iid_a, Wru) + bau),
                Wpu) + bbu
            self.u_j = Mask(self.u_j, seq_len=self.input_u_num, max_len=review_num_u, mode='add')
            self.u_a = tf.nn.softmax(self.u_j, 1)

            # item attention
            Wai = tf.Variable(
                tf.random_uniform([num_filters_total, attention_size], -0.01, 0.01),
                name="Wai"

            )
            Wri = tf.Variable(
                tf.random_uniform([embedding_id, attention_size], -0.01, 0.01),
                name="Wri"
            )
            Wpi = tf.Variable(
                tf.random_uniform([attention_size, 1], -0.01, 0.01),
                name="Wpi"
            )
            bai = tf.Variable(tf.constant(0.1, shape=[attention_size]), name="bai")
            bbi = tf.Variable(tf.constant(0.1, shape=[1]), name="bbi")
            self.uid_a = tf.nn.relu(tf.nn.embedding_lookup(uidW, self.input_reiid))
            self.i_j = tf.einsum('ajk,kl->ajl', tf.nn.relu(
                tf.einsum('ajk,kl->ajl', self.h_drop_i, Wai) + tf.einsum('ajk,kl->ajl', self.uid_a, Wri) + bai),
                Wpi) + bbi
            self.i_j = Mask(self.i_j, seq_len=self.input_i_num, max_len=review_num_i, mode='add')
            self.i_a = tf.nn.softmax(self.i_j, 1)


            l2_loss += tf.nn.l2_loss(Wau)
            l2_loss += tf.nn.l2_loss(Wru)
            l2_loss += tf.nn.l2_loss(Wri)
            l2_loss += tf.nn.l2_loss(Wai)



        with tf.name_scope("add_reviews"):
            self.u_feas = tf.reduce_sum(tf.multiply(self.u_a, self.h_drop_u), 1)
            self.u_feas = tf.nn.dropout(self.u_feas, self.dropout_keep_prob)
            self.i_feas = tf.reduce_sum(tf.multiply(self.i_a, self.h_drop_i), 1)
            self.i_feas = tf.nn.dropout(self.i_feas, self.dropout_keep_prob)

        with tf.name_scope("get_fea"):
            # user fusion (text review + dmf)
            uidmf = tf.Variable(tf.random_uniform([user_num + 2, embedding_id], -0.01, 0.01), name="uidmf")
            self.uid = tf.nn.embedding_lookup(uidmf, self.input_uid)
            self.uid = tf.reshape(self.uid, [-1, embedding_id])
            Wu = tf.Variable(
                tf.random_uniform([num_filters_total, n_latent], -0.01, 0.01),
                name="Wu"
            )
            bu = tf.Variable(
                tf.constant(0.1, shape=[n_latent]),
                name="bu"

            )
            self.u_feas = tf.matmul(self.u_feas, Wu) + self.uid + bu

            # item fusion (text review + dmf)
            iidmf = tf.Variable(tf.random_uniform([item_num + 2, embedding_id], -0.01, 0.01), name="iidmf")
            self.iid = tf.nn.embedding_lookup(iidmf, self.input_iid)
            self.iid = tf.reshape(self.iid, [-1, embedding_id])
            Wi = tf.Variable(
                tf.random_uniform([num_filters_total, n_latent], -0.01, 0.01),
                name="Wi"
            )
            bi = tf.Variable(
                tf.constant(0.1, shape=[n_latent]),
                name="bi"
            )
            self.i_feas = tf.matmul(self.i_feas, Wi) + self.iid + bi

        with tf.name_scope("ncf"):
            self.FM = tf.multiply(self.u_feas, self.i_feas)
            self.FM = tf.nn.relu(self.FM)
            self.FM = tf.nn.dropout(self.FM, self.dropout_keep_prob)

            Wmul = tf.Variable(
                tf.random_uniform([n_latent, 1], -0.01, 0.01),
                name='wmul'
            )

            self.mul = tf.matmul(self.FM, Wmul)
            self.score = tf.reduce_sum(self.mul, 1, keep_dims=True)

            self.uidW2 = tf.Variable(tf.constant(0.1, shape=[user_num + 2]), name="uidW2")
            self.iidW2 = tf.Variable(tf.constant(0.1, shape=[item_num + 2]), name="iidW2")
            self.u_bias = tf.gather(self.uidW2, self.input_uid)
            self.i_bias = tf.gather(self.iidW2, self.input_iid)
            self.Feature_bias = self.u_bias + self.i_bias

            self.bised = tf.Variable(tf.constant(0.1), name='bias')
            self.predictions = self.score + self.Feature_bias + self.bised

        with tf.name_scope("loss"):
            # losses = tf.nn.l2_loss(tf.subtract(self.predictions, self.input_y))
            losses = tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y)))
            self.loss = losses + 0 * l2_loss

        with tf.name_scope("accuracy"):
            self.mae = tf.reduce_mean(tf.abs(tf.subtract(self.predictions, self.input_y)))
            self.rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y))))
            self.mse = tf.reduce_mean(tf.square(tf.subtract(self.predictions, self.input_y)))