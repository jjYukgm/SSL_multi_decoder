import sys
import os
import numpy as np
# plot
from laplotter import LossAccPlotter
from collections import OrderedDict
import time
import pdb
# pdb.set_trace()
# pdb.set_trace = lambda: None

# import matplotlib
# matplotlib.use('Agg')


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

def plot_loss_err(data, suffix, fi_len=None):
    # plot_data = {'X': [], 'Y': [], 'legend': []}
    batch_per_epoch = int((50000 + 100 - 1) / 100) # in data_loader: unlabel_size = tr_size
    # batch_per_epoch /= 500  # vis_period
    # plot settings
    save_to_filepath = os.path.join("{}_log".format(data), "{}_plot_loss_err.png".format(suffix))
    plotter = LossAccPlotter(title="{} loss over time".format(suffix),
                             save_to_filepath=save_to_filepath,
                             show_regressions=True,
                             show_averages=True,
                             show_loss_plot=True,
                             show_err_plot=True,
                             show_plot_window=False,
                             x_label="Epoch")
    ## load loss data
    log_path = os.path.join('{}_log'.format(data), '{}.FM+VI.{}.txt'.format(data, suffix))
    log_file2 = open(log_path, "r")
    st_time = time.time()
    if fi_len is None:
        fi_len = 0
        for li in log_file2:
            li_or = li.split(" | ")
            if len(li_or) == 1:
                continue
            fi_len += 1
        log_file2.seek(0)
        # print("file len: {}".format(fi_len))
    i = 0

    for li in log_file2:
        li_or = li.split(" | ")
        if len(li_or) == 1:
            continue
        iter = li_or[0].split("\t")[0][1:]
        loss_train = float(li_or[0].split(":")[1].split(",")[0])
        err_train = float(li_or[0].split(",")[1])
        loss_val = float(li_or[1].split(":")[1].split(",")[0])
        err_val = float(li_or[1].split(",")[1])

        float_epoch = float(iter) / batch_per_epoch
        plotter.add_values(float_epoch,
                           loss_train=loss_train, loss_val=loss_val,
                           err_train=err_train, err_val=err_val,
                           redraw=False)
        i += 1
        time_str = "{}\r".format(calculate_remaining(st_time, time.time(), i, fi_len))
        sys.stdout.write(time_str)
        sys.stdout.flush()
    sys.stdout.write("\n")
    log_file2.close()
    plotter.redraw()    # save as image
    # plotter.block()
    return fi_len

def plot_losses(data, suffix, fi_len=None):
    # plot_data = {'X': [], 'Y': [], 'legend': []}
    batch_per_epoch = int((50000 + 100 - 1) / 100) # in data_loader: unlabel_size = tr_size
    # batch_per_epoch /= 500  # vis_period
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
                             x_label="Epoch")
    ## load loss data
    log_path = os.path.join('{}_log'.format(data), '{}.FM+VI.{}.txt'.format(data, suffix))
    log_file2 = open(log_path, "r")
    st_time = time.time()
    if fi_len is None:
        fi_len = 0
        for li in log_file2:
            li_or = li.split(" | ")
            if len(li_or) == 1:
                continue
            fi_len += 1
        log_file2.seek(0)
        # print("file len: {}".format(fi_len))
    i = 0
    for li in log_file2:
        li_or = li.split(" | ")
        if len(li_or) == 1:
            continue
        iter = li_or[0].split("\t")[0][1:]
        loss_train = float(li_or[0].split(":")[1].split(",")[0])
        err_train = float(li_or[0].split(",")[1])
        loss_val = float(li_or[1].split(":")[1].split(",")[0])
        err_val = float(li_or[1].split(",")[1])
        for li2 in li_or:
            if not "loss" in li2:
                continue
            # pdb.set_trace()
            key = li2.split(": ")[0]
            value = float(li2.split(": ")[1])
            if key == 'vi loss':
                value *= 1e-2
            other_loss[key] = value

        float_epoch = float(iter) / batch_per_epoch
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
    return fi_len

def main():
    if len(sys.argv) < 3: # 1
        print("Usage:{}<WHO>".format(sys.argv[0]))
        sys.exit(1)       # 2

    data_name = sys.argv[1]
    suffix_name = sys.argv[2]

    fi_len = plot_loss_err(data_name, suffix_name)
    plot_losses(data_name, suffix_name, fi_len=fi_len)


if __name__ == "__main__":
    main()
