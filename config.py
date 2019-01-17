
import numpy as np


class mnist_config:
    dataset = 'mnist'
    image_size = 28 * 28
    num_label = 10

    gen_emb_size = 20
    noise_size = 100

    dis_lr = 3e-3
    enc_lr = 1e-3
    gen_lr = 1e-3

    eval_period = 600
    vis_period = 100

    data_root = 'data'

    size_labeled_data = 100

    train_batch_size = 100
    train_batch_size_2 = 100
    dev_batch_size = 200

    seed = 13

    feature_match = True
    top_k = 5
    top1_weight = 1.

    supervised_only = False
    feature_match = True
    p_loss_weight = 1e-4
    p_loss_prob = 0.1
    
    max_epochs = 2000

    pixelcnn_path = 'model/mnist.True.3.best.pixel'


class svhn_config:
    dataset = 'svhn'
    image_size = 3 * 32 * 32
    num_label = 10

    gen_emb_size = 20
    noise_size = 100

    dis_lr = 1e-3
    enc_lr = 1e-3
    gen_lr = 1e-3
    min_lr = 1e-4

    eval_period = 730
    vis_period = 730

    data_root = 'data'

    size_labeled_data = 1000

    train_batch_size = 64
    train_batch_size_2 = 64
    dev_batch_size = 200

    max_epochs = 900
    ent_weight = 0.1
    pt_weight = 0.8

    p_loss_weight = 1e-4
    p_loss_prob = 0.1

class cifar_config:
    dataset = 'cifar'
    image_size = 3 * 32 * 32
    num_label = 10

    gen_emb_size = 20
    noise_size = 100

    dis_lr = 6e-4
    enc_lr = 3e-4
    gen_lr = 3e-4

    eval_period = 500
    vis_period = 500

    data_root = 'data'

    size_labeled_data = 4000

    train_batch_size = 100
    train_batch_size_2 = 100
    dev_batch_size = 200

    max_epochs = 1200
    vi_weight = 1e-2


class cifarh_config:
    dataset = 'cifarh'
    image_size = 3 * 32 * 32
    num_label = 10

    gen_emb_size = 20
    noise_size = 100

    dis_lr = 6e-4
    enc_lr = 3e-4
    gen_lr = 3e-4

    eval_period = 500
    vis_period = 500

    data_root = 'data'

    size_labeled_data = 10000

    train_batch_size = 100
    train_batch_size_2 = 100
    dev_batch_size = 200

    max_epochs = 1200
    vi_weight = 1e-2


class cifarmg_config:
    dataset = 'cifar'
    image_size = 3 * 32 * 32
    num_label = 10

    gen_emb_size = 20
    noise_size = 100

    dis_lr = 6e-4
    enc_lr = 3e-4
    gen_lr = 3e-4

    eval_period = 500
    vis_period = 500

    data_root = 'data'

    size_labeled_data = 4000

    train_batch_size = 100
    train_batch_size_2 = 100
    dev_batch_size = 200

    max_epochs = 1000    # 1200
    last_epochs = 1000
    dg_start = 0
    vi_weight = 1e-2
    gf_weight = 0.    # #lab / #unl: 4k / 50k
    dl_weight = 1.    # dis lab loss
    du_weight = 1.    # dis unlabeled loss
    dg_weight = 1.    # dis gen loss
    dgl_weight = 0.    # dis gen lab loss
    da_weight = 0.    # dis arcface loss
    dt_weight = 0.    # dis triple loss; no d in dis_mode no tri
    ut_weight = 0.    # dis triple gan loss; no d in dis_mode no tri
    tv_weight = 1.    # tv
    st_weight = 0.    # style loss
    gl_weight = 0.    # gen label loss
    im_weight = 0.    # unl m loss
    cim_weight = 0.    # cosine unl m loss
    il_weight = 0.    # lab loss
    gen_mode = 'z2i'    # z2i; i2i
    dis_mode = 'd'    # dm; d; m
    cd_mode_iter = 0    # change mode epoch

    ema_decay = 0.999
    consistency_type = 'mse'    # mse; kl
    con_coef = 10.0
    nei_coef = 1.0
    nei_margin = 1.0
    nei_top = 1
    c_rampup = 5    # consistency_rampup
    ini_lr = 0.    # lr rampup ini
    lr_rampup = 0    # lr_rampup fin epoch
    lr_rampdn = None    # lr_rampdn
    t_forget_coef = 2.
    t_start = 70.
    drop = False
    double_input_size = False
    flip = False
    dis_double = False
    dis_triple = False

    gop = 'Adam'
    momentum = 0.9
    weight_decay = 2e-4
    nesterov = True  # use nesterov momentum

class cifarmt_config:
    dataset = 'cifar'
    image_size = 3 * 32 * 32
    num_label = 10

    gen_emb_size = 20
    noise_size = 100

    dis_lr = 6e-4
    enc_lr = 3e-4
    gen_lr = 3e-4

    eval_period = 500
    vis_period = 500

    data_root = 'data'

    size_labeled_data = 4000

    train_batch_size = 100
    train_batch_size_2 = 100
    dev_batch_size = 200

    max_epochs = 500    # 1200
    vi_weight = 1e-2

    ema_decay = 0.99
    consistency_type = 'mse'    # mse; kl
    consistency = 50.0
    consistency_rampup = 5
    momentum = 0.9
    weight_decay = 2e-4
    nesterov = True  # use nesterov momentum


class pixelcnn_config:
    dataset = 'mnist'
    image_wh = 28 if dataset == 'mnist' else 32
    n_channel = 1 if dataset == 'mnist' else 3
    image_size = 28 * 28 if dataset == 'mnist' else 32 * 32

    if dataset == 'cifar':
        train_batch_size = 20 * 4
        test_batch_size = 20 * 4
        lr = 1e-3 * 96 / train_batch_size
        disable_third = False
        nr_resnet = 5
        dropout_p = 0.5
    elif dataset == 'svhn':
        train_batch_size = 30 * 4
        test_batch_size = 30 * 4
        lr = 2e-4
        disable_third = True
        nr_resnet = 3
        dropout_p = 0.0
    elif dataset == 'mnist':
        train_batch_size = 40 * 1
        test_batch_size = 40 * 1
        lr = 2e-4
        disable_third = True
        nr_resnet = 3
        dropout_p = 0.0

    eval_period = 30
    save_period = 5
