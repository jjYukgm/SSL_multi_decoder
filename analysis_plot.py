import sys
import os
import numpy as np
# plot
from laplotter import LossAccPlotter, MultiPlotter
from collections import OrderedDict
import time
import pdb
# pdb.set_trace()
# pdb.set_trace = lambda: None

# import matplotlib
# matplotlib.use('Agg')


def str2flo(str):
    if 'nan' in str:
        flo = 0.
    else:
        flo = float(str)
    return flo

def calculate_remaining(t1, t2, i, total):
    progress = (i + 0.) / total
    elapsed_time = t2 - t1
    if progress > 0:
        remaining_time = elapsed_time * (1/progress) - elapsed_time
    else:
        remaining_time = 0

    # return progress, remaining_time
    psec = int(remaining_time % 60)
    pmin = int(remaining_time // 60)
    time_str = '[{:8.2%}], remain: {:2d}:{:2d} '.format(progress, pmin, psec)
    time_str = '| ' + time_str
    return time_str

def check_val(data, suffix, b_size=None):
    dict = locals()
    ema = None
    tri = None
    # plot_data = {'X': [], 'Y': [], 'legend': []}
    if not b_size:
        b_size = 100
    batch_per_epoch = int((50000 + 100 - 1) / b_size) # in data_loader: unlabel_size = tr_size
    # batch_per_epoch /= 500  # vis_period
    ## load loss data
    log_path = os.path.join('{}_log'.format(data), '{}.FM+VI.{}.txt'.format(data, suffix))
    log_file2 = open(log_path, "r")
    st_time = time.time()
    fi_len = 0
    for li in log_file2:
        li_or = li.split(" | ")
        if len(li_or) == 1:
            continue
        if ema is None and "ema" in li_or[3]:
            ema = True
        if tri is None and "tri" in li_or[4]:
            tri = True
        fi_len += 1
    # print("file len: {}".format(fi_len))
    if not ema:
        ema = False
    if not tri:
        tri = False

    dict["batch_per_epoch"] = batch_per_epoch
    dict["log_path"] = log_path
    dict["fi_len"] = fi_len
    dict["ema"] = ema
    dict["tri"] = tri
    return dict

def plot_loss_err(data, suffix, fi_len, ema, batch_per_epoch, **kwargs):
    ## load loss data
    log_path = kwargs["log_path"]
    log_file2 = open(log_path, "r")
    st_time = time.time()

    # plot settings
    save_to_filepath = os.path.join("{}_log".format(data), "{}_plot_loss_err.png".format(suffix))
    plotter = LossAccPlotter(title="{} loss over time".format(suffix),
                             save_to_filepath=save_to_filepath,
                             show_regressions=True,
                             show_averages=True,
                             show_loss_plot=True,
                             show_err_plot=True,
                             show_ema_plot=ema,
                             show_plot_window=False,
                             x_label="Epoch")
    i = 0
    for li in log_file2:
        li_or = li.split(" | ")
        if len(li_or) == 1:
            continue
        iter = li_or[0].split("\t")[0][1:]
        loss_train = str2flo(li_or[0].split(":")[1].split(",")[0])
        err_train = str2flo(li_or[0].split(",")[1])
        loss_val = str2flo(li_or[1].split(":")[1].split(",")[0])
        err_val = str2flo(li_or[1].split(",")[1])
        ema_err_train = ema_err_val = None
        if ema:
            ema_err_train = li_or[3].split(":")[1].split(",")[0]
            ema_err_val = li_or[3].split(",")[1]
            if "None" not in ema_err_train:
                ema_err_train = str2flo(ema_err_train)
                ema_err_val = str2flo(ema_err_val)
            else:
                ema_err_train = ema_err_val = None

        float_epoch = str2flo(iter) / batch_per_epoch
        plotter.add_values(float_epoch,
                           loss_train=loss_train, loss_val=loss_val,
                           err_train=err_train, err_val=err_val,
                           ema_err_train=ema_err_train, ema_err_val=ema_err_val,
                           redraw=False)
        i += 1
        time_str = "{}\r".format(calculate_remaining(st_time, time.time(), i, fi_len))
        sys.stdout.write(time_str)
        sys.stdout.flush()
    sys.stdout.write("\n")
    log_file2.close()
    plotter.redraw()    # save as image
    # plotter.block()

def plot_losses(data, suffix, fi_len, batch_per_epoch, **kwargs):
    # plot_data = {'X': [], 'Y': [], 'legend': []}
    other_loss = OrderedDict()

    # plot settings
    save_to_filepath = os.path.join("{}_log".format(data), "{}_plot_losses.png".format(suffix))
    plotter = LossAccPlotter(title="{} loss over time".format(suffix),
                             save_to_filepath=save_to_filepath,
                             show_regressions=False,
                             show_averages=False,
                             show_other_loss=True,
                             show_log_loss=True,
                             show_loss_plot=True,
                             show_err_plot=True,
                             show_plot_window=False,
                             epo_max=1000,
                             x_label="Epoch")
    ## load loss data
    log_path = kwargs["log_path"]
    log_file2 = open(log_path, "r")
    st_time = time.time()
    i = 0
    for li in log_file2:
        li_or = li.split(" | ")
        if len(li_or) == 1:
            continue
        iter = li_or[0].split("\t")[0][1:]
        loss_train = str2flo(li_or[0].split(":")[1].split(",")[0])
        err_train = str2flo(li_or[0].split(",")[1])
        loss_val = str2flo(li_or[1].split(":")[1].split(",")[0])
        err_val = str2flo(li_or[1].split(",")[1])
        for li2 in li_or:
            if "loss" not in li2:
                continue
            # pdb.set_trace()
            key = li2.split(": ")[0]
            value = str2flo(li2.split(": ")[1])
            if key == 'vi loss':
                value *= 1e-2
            other_loss[key] = value

        float_epoch = str2flo(iter) / batch_per_epoch
        plotter.add_values(float_epoch,
                           loss_train=loss_train, loss_val=loss_val,
                           err_train=err_train, err_val=err_val,
                           redraw=False, other_loss = other_loss)
        i += 1
        time_str = "{}\r".format(calculate_remaining(st_time, time.time(), i, fi_len))
        # print(time_string, end = '\r')
        sys.stdout.write(time_str)
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    log_file2.close()
    plotter.redraw()    # save as image
    # plotter.block()

def plot_values(data, suffix, fi_len, batch_per_epoch, **kwargs):
    # plot gen acc and unl acc
    # plot_data = {'X': [], 'Y': [], 'legend': []}
    other_value = OrderedDict()

    # plot settings
    save_to_filepath = os.path.join("{}_log".format(data), "{}_plot_values.png".format(suffix))
    plotter = MultiPlotter(title="{} Accuracy over time".format(suffix),
                             save_to_filepath=save_to_filepath,
                             show_regressions=False,
                             show_averages=False,
                             show_plot_window=False,
                             epo_max=1000,
                             x_label="Epoch")
    ## load loss data
    log_path = kwargs["log_path"]
    log_file2 = open(log_path, "r")
    st_time = time.time()
    i = 0
    for li in log_file2:
        li_or = li.split(" | ")
        if len(li_or) == 1:
            continue
        iter = li_or[0].split("\t")[0][1:]
        li_ev = li.split("[Eval]")[1].split(" | ")[0].split(",")
        for li2 in li_ev:
            if "acc" not in li2:
                continue
            # pdb.set_trace()
            key = li2.split(": ")[0]
            value = str2flo(li2.split(": ")[1])
            other_value[key] = value

        float_epoch = str2flo(iter) / batch_per_epoch
        plotter.add_values(float_epoch, other_value = other_value,
                           redraw=False)
        i += 1
        time_str = "{}\r".format(calculate_remaining(st_time, time.time(), i, fi_len))
        # print(time_string, end = '\r')
        sys.stdout.write(time_str)
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    log_file2.close()
    plotter.redraw()    # save as image
    # plotter.block()

def plot_tri(data, suffix, fi_len, batch_per_epoch, tri, ema, **kwargs):
    assert tri, "tri Something Wrong"
    # plot gen acc and unl acc
    # plot_data = {'X': [], 'Y': [], 'legend': []}
    other_value = OrderedDict()

    # plot settings
    save_to_filepath = os.path.join("{}_log".format(data), "{}_plot_tri.png".format(suffix))
    plotter = MultiPlotter(title="{} ErrorRate over time".format(suffix),
                             save_to_filepath=save_to_filepath,
                             show_regressions=False,
                             show_averages=False,
                             show_plot_window=False,
                             epo_max=1000,
                             x_label="Epoch")
    ## load loss data
    log_path = kwargs["log_path"]
    log_file2 = open(log_path, "r")
    st_time = time.time()
    i = 0
    for li in log_file2:
        li_or = li.split(" | ")
        if len(li_or) == 1:
            continue
        iter = li_or[0].split("\t")[0][1:]

        other_value["err_train"] = str2flo(li_or[0].split(",")[1])
        other_value["err_val"] = str2flo(li_or[1].split(",")[1])
        if ema:
            ema_err_train = li_or[3].split(":")[1].split(",")[0]
            ema_err_val = li_or[3].split(",")[1]
            if "None" not in ema_err_train:
                ema_err_train = str2flo(ema_err_train)
                ema_err_val = str2flo(ema_err_val)
            else:
                ema_err_train = ema_err_val = None
            other_value["ema_err_train"] = ema_err_train
            other_value["ema_err_val"] = ema_err_val
        # tri
        tri_2 = li_or[4].split(":")[1].split(",")[0]
        tri_3 = li_or[4].split(",")[1]
        other_value["tri_2"] = str2flo(tri_2)
        other_value["tri_3"] = str2flo(tri_3)

        float_epoch = str2flo(iter) / batch_per_epoch
        plotter.add_values(float_epoch, other_value = other_value,
                           redraw=False)
        i += 1
        time_str = "{}\r".format(calculate_remaining(st_time, time.time(), i, fi_len))
        # print(time_string, end = '\r')
        sys.stdout.write(time_str)
        sys.stdout.flush()
    sys.stdout.write("\n")
    sys.stdout.flush()
    log_file2.close()
    plotter.redraw()    # save as image
    # plotter.block()

def main():
    if len(sys.argv) < 3: # 1
        print("Usage:{} <DATA> <WHO> [batchsize]".format(sys.argv[0]))
        sys.exit(1)       # 2

    if len(sys.argv) > 3:
        b_size = int(sys.argv[3])
    else:
        b_size = None
    dict = {"data": sys.argv[1], "suffix": sys.argv[2], "b_size": b_size}
    dict = check_val(**dict)
    plot_loss_err(**dict)
    plot_losses(**dict)
    plot_values(**dict)
    if dict["tri"]: plot_tri(**dict)


if __name__ == "__main__":
    main()
