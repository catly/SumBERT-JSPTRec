# coding=utf-8

import argparse

def get_params():

    parser = argparse.ArgumentParser()
    ps = parser.add_argument


    # dataset
    ps("--root_data_dir", dest="root_data_dir", type=str, default="./data", help="root dataset dir")
    ps("--dataset", dest="dataset", type=str, default="music", help='dataset')
    ps("--data_type", dest="data_type", type=str, default="amazon", help="data type")
    ps("--gpu_index", dest="gpu_index", type=str, default='0', help="gpu index")

    # word2vec
    ps("--w2v_file", dest="w2v_file", type=str, default="w2v/w2v_amazon_100.txt", help="word2vec embeddings")
    ps("--embed_dim", dest='embed_dim', type=int, default=100, help="word embedding dim")
    ps("--embed_type", dest="embed_type", type=str, default='trained', help="embed type (random or pre-trained)")
    ps("--embed_way", dest="embed_way", type=str, default="sep", help="ways for user embedding and item embedding (use united or seperated embedding)")

    # model
    ps("--model_type", dest="model_type", type=str, default="", help="model type")
    ps("--batch_size",dest="batch_size", type=int, default=100, help='batch size')
    ps("--dropout_keep_prob", dest="dropout_keep_prob", type=float, default=0.5, help="dropout keep prob")
    ps("--l2_reg_lambda", dest="l2_reg_lambda", type=float, default=0, help="L2 regularization lambda")
    ps("--lr", dest="lr", type=float, default=0.001, help="learning rate") # 0.005, 0.01, 0.02, 0.05
    ps("--init_type", dest="init_type", type=str, metavar='<str>', default='uniform', help="Init Type(xavier, normal ,uniform)")
    ps("--init", dest="init", type=float, metavar='<float>', default=0.01, help="Init Paramater (maxval)")

    # training
    ps("--num_epoches", dest="num_epoches", type=int, default=40, help="number of training epoches")
    ps("--evaluate_every", dest="evaluate_every", type=int ,default=100, help="Evaluate model on dev set after many steps")

    # deepcnn parameters
    ps("--u_doc_len", dest="u_doc_len", type=int, default=0, help="max document length for user reviews")
    ps("--i_doc_len", dest="i_doc_len", type=int, default=0, help="max document length for item reviews")
    ps("--deepconn_filter_sizes", dest="deepconn_filter_sizes", type=str, default="3", help="filter sizes for deepconn")
    ps("--deepconn_num_filters", dest="deepconn_num_filters", type=int ,default=100, help="number of filters")
    ps("--deepconn_n_latent", dest='deepconn_n_latent', type=int, default=32, help="latent size")
    ps("--deepconn_fm_k", dest='deepconn_fm_k', type=int ,default=8, help="fm hidden numbers")

    # narre parameters
    ps("--u_max_num", dest="u_max_num", type=int, default=0, help="max number for user reviews")
    ps("--u_max_len", dest="u_max_len", type=int, default=0, help="max length for each user review")
    ps("--i_max_num", dest='i_max_num', type=int, default=0, help="max number for item reviews")
    ps("--i_max_len", dest="i_max_len", type=int, default=0, help="max length for each item review")
    ps("--narre_n_latent", dest='narre_n_latent', type=int, default=32, help="latent size for narre")
    ps("--narre_embedding_id", dest='narre_embedding_id', type=int, default=32, help="embeddding id for narre")
    ps("--narre_attention_size", dest='narre_attention_size', type=int, default=32, help="attention size for narre")
    ps("--narre_filter_sizes", dest="narre_filter_sizes", type=str, default="3", help="filter sizes for narre")
    ps("--narre_num_filters", dest='narre_num_filters', type=int, default=768, help="number of filters for narre")
    # summary
    ps("--summ_proportion", dest='summ_proportion', type=float, default=0.6, help="the proportion of the simmary")

    # DATT parameters
    ps("--datt_L_num_filters", dest="datt_L_num_filters", type=int, default=200, help="local number of filters for DATT")
    ps("--datt_G_num_filters", dest="datt_G_num_filters", type=int, default=100, help="Global number of filters for DATT")
    ps("--datt_fc1_size", dest="datt_fc1_size", type=int, default=500, help="first fully connected layer size for DATT")
    ps("--datt_fc2_size", dest="datt_fc2_size", type=int, default=50, help="second fully conneceted layer size fo DATT")

    # TransNET parameters
    ps("--review_length", dest="review_length", type=int, default=0, help="max length for review")
    ps("--tnet_num_filters", dest="tnet_num_filters", type=int, default=100, help="number of filters for TransNET")
    ps("--tnet_filter_sizes", dest="tnet_filter_sizes", type=str, default="3", help="filter sizes for TransNET")
    ps("--tnet_fm_k", dest="tnet_fm_k", type=int, default=8, help="fm_k for TransNET")
    ps("--tnet_hidden_size", dest="tnet_hidden_size", default=100, help="hidden size for TransNET")
    ps("--tnet_trans_layers", dest="tnet_trans_layers", default=2, help="trans layers for TransNET")

    # MPCN parameters
    ps("--mpcn_num_heads", dest="mpcn_num_heads", type=int, default=3, help="number headers for mpcn")
    ps("--mpcn_ensemble_type", dest="mpcn_ensemble_type", type=str, default='FN',help="ensemble type for mpcn (FN, ADD)")
    ps("--mpcn_num_layers", dest="mpcn_num_layers", type=int, default=2, help='number layers for mpcn')
    ps("--mpcn_fm_k", dest="mpcn_fm_k", type=int, default=8, help="fm_k for mpcn")

    # RDMH parameters
    ps("--rdmh_gate_size", dest="rdmh_gate_size", type=int, default=100, help="gate size for rdmh")
    ps("--rdmh_block_list_u", dest="rdmh_block_list_u", type=str, default="10", help="block list user for rdmh")
    ps("--rdmh_block_list_i", dest="rdmh_block_list_i", type=str, default="10", help="block list item for rdmh")
    ps("--rdmh_rnn_type", dest="rdmh_rnn_type", type=str, default='gru', help="rnn type (gru or lstm, cnn)")
    ps("--rdmh_rnn_size", dest="rdmh_rnn_size", type=int, default=50, help='rnn hidden size for rdmh')
    ps("--rdmh_num_headers", dest="rdmh_num_headers", type=int, default=3, help="number headers for rdmh")
    ps("--rdmh_fusion_size", dest="rdmh_fusion_size", type=int, default=100, help="fusion size for rdmh")
    ps("--rdmh_predict_type", dest="rdmh_predict_type", type=str, default="FM", help="predict type for rdmh (FM, NCF)")
    ps("--rdmh_ncf_latent", dest="rdmh_ncf_latent", type=int, default=32, help="ncf latent size for rdmh")
    ps("--rdmh_fm_k", dest="rdmh_fm_k", type=int, default=8, help="fm_k for rdmh")
    ps("--rdmh_review_type", dest="rdmh_review_type", type=str, default="avg_pooling", help="aggregation layer for reviews (max_pooling, avg_pooling)")
    ps("--rdmh_block_type", dest="rdmh_block_type", type=str, default="reduce_mean", help="aggregation layer for block (reduce_max, reduce_mean, self_attention)")

     # Inter parameters
    ps("--inter_n_latent", dest='inter_n_latent', type=int, default=32, help="latent size for interact")
    ps("--inter_embedding_id", dest='inter_embedding_id', type=int, default=32, help="embeddding id for interact")
    ps("--inter_attention_size", dest='inter_attention_size', type=int, default=32, help="attention size for interact")
    ps("--inter_filter_sizes", dest="inter_filter_sizes", type=str, default="3", help="filter sizes for interact")
    ps("--inter_num_filters", dest='inter_num_filters', type=int, default=100, help="number of filters for interact")
    ps("--inter_latent_size", dest="inter_latent_size", type=int, default=100, help="latent size for interact")
    ps("--inter_nb_head", dest="inter_nb_head", type=int, default=1, help="number of headers for interact")

    return parser
