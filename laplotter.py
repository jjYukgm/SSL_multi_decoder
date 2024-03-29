"""A class to generate plots for the results of applied loss functions and/or
error of models trained with machine learning methods.
# reference: https://github.com/aleju/LossAccPlotter/blob/master/laplotter.py

Example:
    plotter = LossAccPlotter()
    for epoch in range(100):
        loss_train, err_train = your_model.train()
        loss_val, err_val = your_model.validate()
        plotter.add_values(epoch,
                           loss_train=loss_train, err_train=err_train,
                           loss_val=loss_val, err_val=err_val)
    plotter.block()

Example, no error chart:
    plotter = LossAccPlotter(show_err_plot=False)
    for epoch in range(100):
        loss_train = your_model.train()
        loss_val = your_model.validate()
        plotter.add_values(epoch, loss_train=loss_train, loss_val=loss_val)
    plotter.block()

Example, update the validation line only every 10th epoch:
    plotter = LossAccPlotter(show_err_plot=False)
    for epoch in range(100):
        loss_train = your_model.train()
        if epoch % 10 == 0:
            loss_val = your_model.validate()
        else:
            loss_val = None
        plotter.add_values(epoch, loss_train=loss_train, loss_val=loss_val)
    plotter.block()
"""
from __future__ import absolute_import
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import warnings
import math
from collections import OrderedDict

def ignore_nan_and_inf(value, label, x_index):
    """Helper function that creates warnings on NaN/INF and converts them to None.
    Args:
        value: The value to check for NaN/INF.
        label: For which line the value was used (usually "loss train", "loss val", ...)
            This is used in the warning message.
        x_index: At which x-index the value was used (e.g. 1 as in Epoch 1).
            This is used in the warning message.
    Returns:
        value, but None if value is NaN or INF.
    """
    if value is None:
        return None
    elif math.isnan(value):
        warnings.warn("Got NaN for value '%s' at x-index %d" % (label, x_index))
        return None
    elif math.isinf(value):
        warnings.warn("Got INF for value '%s' at x-index %d" % (label, x_index))
        return None
    else:
        return value

class LossAccPlotter(object):
    """Class to plot loss and error charts (for training and validation data)."""

    def __init__(self,
                 title=None,
                 save_to_filepath=None,
                 show_regressions=True,
                 show_averages=True,
                 show_other_loss=False,
                 show_log_loss=False,
                 fix_ylim=True,
                 show_loss_plot=True,
                 show_err_plot=True,
                 show_ema_plot=False,
                 show_plot_window=True,
                 epo_max=None,
                 x_label="Epoch"):
        """Constructs the plotter.

        Args:
            title: An optional title which will be shown at the top of the
                plot. E.g. the name of the experiment or some info about it.
                If set to None, no title will be shown. (Default is None.)
            save_to_filepath: The path to a file in which the plot will be saved,
                e.g. "/tmp/last_plot.png". If set to None, the chart will not be
                saved to a file. (Default is None.)
            show_regressions: Whether or not to show a regression, indicating
                where each line might end up in the future.
            show_averages: Whether to plot moving averages in the charts for
                each line (so for loss train, loss val, ...). This value may
                only be True or False. To change the interval (default is 20
                epochs), change the instance variable "averages_period" to the new
                integer value. (Default is True.)
            show_loss_plot: Whether to show the chart for the loss values. If
                set to False, only the error chart will be shown. (Default
                is True.)
            show_err_plot: Whether to show the chart for the error value. If
                set to False, only the loss chart will be shown. (Default is True.)
            show_plot_window: Whether to show the plot in a window (True)
                or hide it (False). Hiding it makes only sense if you
                set save_to_filepath. (Default is True.)
            x_label: Label on the x-axes of the charts. Reasonable choices
                would be: "Epoch", "Batch" or "Example". (Default is "Epoch".)
        """
        assert show_loss_plot or show_err_plot
        assert save_to_filepath is not None or show_plot_window

        self.title = title
        self.title_fontsize = 14
        self.show_regressions = show_regressions
        self.show_averages = show_averages
        self.show_other_loss = show_other_loss
        self.show_log_loss = show_log_loss
        self.fix_ylim = fix_ylim
        self.show_loss_plot = show_loss_plot
        self.show_err_plot = show_err_plot
        self.show_ema_plot = show_ema_plot
        self.show_plot_window = show_plot_window
        self.epo_max = epo_max
        self.save_to_filepath = save_to_filepath
        self.x_label = x_label

        # alpha values
        # 0.8 = quite visible line
        # 0.5 = moderately visible line
        # thick is used for averages and regression (also for the main values,
        # if there are no averages),
        # thin is used for the main values
        self.alpha_thick = 0.8
        self.alpha_thin = 0.5

        # the interval for the moving averages, e.g. 20 = average over 20 points
        self.averages_period = 20

        # these values deal with the regression
        self.poly_forward_perc = 0.1
        self.poly_backward_perc = 0.2
        self.poly_n_forward_min = 5
        self.poly_n_backward_min = 10
        self.poly_n_forward_max = 100
        self.poly_n_backward_max = 100
        self.poly_degree = 1

        # whether to show grids in both charts
        self.grid = True

        # the styling of the lines
        # sma = simple moving average
        self.linestyles = {
            "loss_train": "r-",
            "loss_train_sma": "r-",
            "loss_train_regression": "r:",
            "loss_val": "b-",
            "loss_val_sma": "b-",
            "loss_val_regression": "b:",
            "err_train": "r-",
            "err_train_sma": "r-",
            "err_train_regression": "r:",
            "err_val": "b-",
            "err_val_sma": "b-",
            "err_val_regression": "b:"
        }
        # different linestyles for the first epoch (if only one value is available),
        # because no line can then be drawn (needs 2+ points) and only symbols will
        # be shown.
        # No regression here, because regression always has at least at least
        # two xy-points (last real value and one (or more) predicted values).
        # No averages here, because the average over one value would be identical
        # to the value anyways.
        self.linestyles_one_value = {
            "loss_train": "rs-",
            "loss_val": "b^-",
            "err_train": "rs-",
            "err_val": "b^-"
        }

        # these values will be set in _initialize_plot() upon the first call
        # of redraw()
        # fig: the figure of the whole plot
        # ax_loss: loss chart (left)
        # ax_err: error chart (right)
        self.fig = None
        self.ax_loss = None
        self.ax_err = None

        # dictionaries with x, y values for each line
        self.values_loss_train = OrderedDict()
        self.values_loss_val = OrderedDict()
        self.values_err_train = OrderedDict()
        self.values_err_val = OrderedDict()
        if self.show_other_loss:
            self.other_loss = OrderedDict()
        if self.show_ema_plot:
            self.ema_value = OrderedDict()

    def add_values(self, x_index, loss_train=None, loss_val=None,
                   err_train=None, err_val=None,
                   ema_err_train=None, ema_err_val=None,
                   redraw=True, other_loss=None):
        """Function to add new values for each line for a specific x-value (e.g.
        a specific epoch).

        Meaning of the values / lines:
         - loss_train: y-value of the loss function applied to the training set.
         - loss_val:   y-value of the loss function applied to the validation set.
         - err_train:  y-value of the error (e.g. 0.0 to 1.0) when measured on
                       the training set.
         - err_val:    y-value of the error (e.g. 0.0 to 1.0) when measured on
                       the validation set.

        Values that are None will be ignored.
        Values that are INF or NaN will be ignored, but create a warning.

        It is currently assumed that added values follow logically after
        each other (progressive order), so the first x_index might be 1 (first entry),
        then 2 (second entry), then 3 (third entry), ...
        Not allowed would be e.g.: 10, 11, 5, 7, ...
        If that is not the case, you will get a broken line graph.

        Args:
            x_index: The x-coordinate, e.g. x_index=5 might represent Epoch 5.
            loss_train: The y-value of the loss train line at the given x_index.
                If None, no value for the loss train line will be added at
                the given x_index. (Default is None.)
            loss_val: Same as loss_train for the loss validation line.
                (Default is None.)
            err_train: Same as loss_train for the error train line.
                (Default is None.)
            err_val: Same as loss_train for the error validation line.
                (Default is None.)
            redraw: Whether to redraw the plot immediatly after receiving the
                new values. This is reasonable if you add values once at the end
                of every epoch. If you add many values in a row, set this to
                False and call redraw() at the end (significantly faster).
                (Default is True.)
        """
        assert isinstance(x_index, (int, float))
        if self.epo_max and x_index > self.epo_max:
            return

        loss_train = ignore_nan_and_inf(loss_train, "loss train", x_index)
        loss_val = ignore_nan_and_inf(loss_val, "loss val", x_index)
        err_train = ignore_nan_and_inf(err_train, "err train", x_index)
        err_val = ignore_nan_and_inf(err_val, "err val", x_index)
        if self.show_other_loss and other_loss is not None:
            if x_index == 0.:
                for key in other_loss.keys():
                    self.other_loss[key] = OrderedDict()
            for key, val in other_loss.items():
                val2 = ignore_nan_and_inf(val, key, x_index)
                if val2 is not None:
                    self.other_loss[key][x_index] = val2
        if self.show_ema_plot and x_index == 0.:
            self.ema_value["ema train"] = OrderedDict()
            self.ema_value["ema val"] = OrderedDict()
        if self.show_ema_plot and ema_err_train is not None:
            ema_err_train = ignore_nan_and_inf(ema_err_train, "ema train", x_index)
            ema_err_val = ignore_nan_and_inf(ema_err_val, "ema val", x_index)
            if ema_err_train is not None:
                self.ema_value["ema train"][x_index] = ema_err_train
            if ema_err_val is not None:
                self.ema_value["ema val"][x_index] = ema_err_val

        if loss_train is not None:
            self.values_loss_train[x_index] = loss_train
        if loss_val is not None:
            self.values_loss_val[x_index] = loss_val
        if err_train is not None:
            self.values_err_train[x_index] = err_train
        if err_val is not None:
            self.values_err_val[x_index] = err_val

        if redraw:
            self.redraw()

    def block(self):
        """Function to show the plot in a blocking way.

        This should be called at the end of your program. Otherwise the
        chart will be closed automatically (at the end).
        By default, the plot is shown in a non-blocking way, so that the
        program continues execution, which causes it to close automatically
        when the program finishes.

        This function will silently do nothing if show_plot_window was set
        to False in the constructor.
        """
        if self.show_plot_window:
            plt.figure(self.fig.number)
            plt.show()

    def save_plot(self, filepath):
        """Saves the current plot to a file.

        Args:
            filepath: The path to the file, e.g. "/tmp/last_plot.png".
        """
        self.fig.savefig(filepath, bbox_inches="tight")

    def _initialize_plot(self):
        """Creates empty figure and axes of the plot and shows it in a new window.
        """
        if self.show_loss_plot and self.show_err_plot:
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(24, 8))
            self.fig = fig
            self.ax_loss = ax1
            self.ax_err = ax2
        else:
            fig, ax = plt.subplots(ncols=1, figsize=(12, 8))
            self.fig = fig
            self.ax_loss = ax if self.show_loss_plot else None
            self.ax_err = ax if self.show_err_plot else None

        if self.ax_loss is not None:
            if self.show_log_loss:  # semilogy; plot
                self.ax1_plot = self.ax_loss.semilogy
                # self.ax_loss.set_ylim([1e-3, 10.])
            else:
                self.ax1_plot = self.ax_loss.plot
                # self.ax_loss.set_ylim([0., 1.])

        # if self.ax_err is not None:
        #     self.ax_err.set_ylim([0., 1.])

        # set_position is neccessary here in order to make space at the bottom
        # for the legend
        for ax in [self.ax_loss, self.ax_err]:
            if ax is not None:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                 box.width, box.height * 0.9])

        # draw the title
        # it seems to be necessary to set the title here instead of in redraw(),
        # otherwise the title is apparently added again and again with every
        # epoch, making it ugly and bold
        if self.title is not None:
            self.fig.suptitle(self.title, fontsize=self.title_fontsize)

        if self.show_plot_window:
            plt.show(block=False)

    def redraw(self):
        """Redraws the plot with the current values.

        This is a full redraw and includes recalculating averages and regressions.
        It should not be called many times per second as that would be slow.
        Calling it every couple seconds should create no noticeable slowdown though.

        Args:
            epoch: The index of the current epoch, starting at 0.
            train_loss: All of the training loss values of each
                epoch (list of floats).
            train_err: All of the training error values of each
                epoch (list of floats).
            val_loss: All of the validation loss values of each
                epoch (list of floats).
            val_err: All of the validation error values of each
                epoch (list of floats).
        """
        # initialize the plot if it's the first redraw
        if self.fig is None:
            self._initialize_plot()

        # activate the plot, in case another plot was opened since the last call
        plt.figure(self.fig.number)

        # shorter local variables
        ax1 = self.ax_loss
        ax2 = self.ax_err

        # set chart titles, x-/y-labels and grid
        for ax, label in zip([ax1, ax2], ["Loss", "ErrorRate"]):
            if ax:
                ax.clear()
                ax.set_title(label)
                ax.set_ylabel(label)
                ax.set_xlabel(self.x_label)
                ax.grid(self.grid)

        # Plot main lines, their averages and the regressions (predictions)
        self._redraw_main_lines()
        self._redraw_averages()
        self._redraw_regressions()
        self._redraw_otherloss()
        self._redraw_ema()

        # Add legends (below both chart)
        ncol = 1
        labels = ["$CHART train", "$CHART val."]
        if self.show_averages:
            labels.extend(["$CHART train (avg %d)" % (self.averages_period,),
                           "$CHART val. (avg %d)" % (self.averages_period,)])
            ncol += 1
        if self.show_regressions:
            labels.extend(["$CHART train (regression)",
                           "$CHART val. (regression)"])
            ncol += 1

        labels1 = list(labels)
        ncol1 = ncol
        if self.show_other_loss:
            for key in self.other_loss.keys():
                labels1.extend([key])
                ncol1 += 1

        if ax1:
            ax1.legend([label.replace("$CHART", "loss") for label in labels1],
                       loc="upper center",
                       bbox_to_anchor=(0.5, -0.08),
                       ncol=ncol1)
        if ax2:
            ax2.legend([label.replace("$CHART", "err.") for label in labels],
                       loc="upper center",
                       bbox_to_anchor=(0.5, -0.08),
                       ncol=ncol)

        if self.fix_ylim:
            if ax1:
                if self.show_log_loss:  # semilogy; plot
                    ax1.set_ylim([1e-3, 10.])
                else:
                    ax1.set_ylim([0., 3.])

            if ax2:
                ax2.set_ylim([0., 1.])

        # if self.epo_max:
        #     if ax1:
        #         ax1.set_xlim([0., self.epo_max])
        #     if ax2:
        #         ax2.set_xlim([0., self.epo_max])

        plt.draw()

        # save the redrawn plot to a file upon every redraw.
        if self.save_to_filepath is not None:
            self.save_plot(self.save_to_filepath)

    def _redraw_main_lines(self):
        """Draw the main lines of values (i.e. loss train, loss val, err train, err val).

        Returns:
            List of handles (one per line).
        """
        handles = []
        ax1 = self.ax_loss
        ax2 = self.ax_err

        # Set the styles of the lines used in the charts
        # Different line style for epochs after the first one, because
        # the very first epoch has only one data point and therefore no line
        # and would be invisible without the changed style.
        ls_loss_train = self.linestyles["loss_train"]
        ls_loss_val = self.linestyles["loss_val"]
        ls_err_train = self.linestyles["err_train"]
        ls_err_val = self.linestyles["err_val"]
        if len(self.values_loss_train) == 1:
            ls_loss_train = self.linestyles_one_value["loss_train"]
        if len(self.values_loss_val) == 1:
            ls_loss_val = self.linestyles_one_value["loss_val"]
        if len(self.values_err_train) == 1:
            ls_err_train = self.linestyles_one_value["err_train"]
        if len(self.values_err_val) == 1:
            ls_err_val = self.linestyles_one_value["err_val"]

        # Plot the lines
        alpha_main = self.alpha_thin if self.show_averages else self.alpha_thick
        if ax1:
            h_lt, = self.ax1_plot(self.values_loss_train.keys(), self.values_loss_train.values(),
                             ls_loss_train, label="loss train", alpha=alpha_main)
            h_lv, = self.ax1_plot(self.values_loss_val.keys(), self.values_loss_val.values(),
                             ls_loss_val, label="loss val.", alpha=alpha_main)
            handles.extend([h_lt, h_lv])
        if ax2:
            h_at, = ax2.plot(self.values_err_train.keys(), self.values_err_train.values(),
                             ls_err_train, label="err. train", alpha=alpha_main)
            h_av, = ax2.plot(self.values_err_val.keys(), self.values_err_val.values(),
                             ls_err_val, label="err. val.", alpha=alpha_main)
            handles.extend([h_at, h_av])

        return handles

    def _redraw_ema(self):
        """Draw the moving averages of each line.

        If moving averages has been deactived in the constructor, this function
        will do nothing.

        Returns:
            List of handles (one per line).
        """
        # abort if moving averages have been deactivated
        if not self.show_ema_plot:
            return []

        handles = []
        ax2 = self.ax_err
        if not ax2:
            return handles
        # for loss chart
        for key, od in self.ema_value.items():
            # plot the xy-values
            h_lt, = ax2.plot(od.keys(), od.values(),
                             label=key)
            handles.extend([h_lt])

        return handles

    def _redraw_otherloss(self):
        """Draw the moving averages of each line.

        If moving averages has been deactived in the constructor, this function
        will do nothing.

        Returns:
            List of handles (one per line).
        """
        # abort if moving averages have been deactivated
        if not self.show_other_loss:
            return []

        handles = []
        ax1 = self.ax_loss
        if not ax1:
            return handles
        # for loss chart
        for key, od in self.other_loss.items():
            # plot the xy-values
            h_lt, = self.ax1_plot(od.keys(), od.values(),
                             label=key)
            handles.extend([h_lt])

        return handles

    def _redraw_averages(self):
        """Draw the moving averages of each line.

        If moving averages has been deactived in the constructor, this function
        will do nothing.

        Returns:
            List of handles (one per line).
        """
        # abort if moving averages have been deactivated
        if not self.show_averages:
            return []

        handles = []
        ax1 = self.ax_loss
        ax2 = self.ax_err

        # calculate the xy-values
        if ax1:
            # for loss chart
            (lt_sma_x, lt_sma_y) = self._calc_sma(self.values_loss_train.keys(),
                                                  self.values_loss_train.values())
            (lv_sma_x, lv_sma_y) = self._calc_sma(self.values_loss_val.keys(),
                                                  self.values_loss_val.values())
        if ax2:
            # for error chart
            (at_sma_x, at_sma_y) = self._calc_sma(self.values_err_train.keys(),
                                                  self.values_err_train.values())
            (av_sma_x, av_sma_y) = self._calc_sma(self.values_err_val.keys(),
                                                  self.values_err_val.values())

        # plot the xy-values
        alpha_sma = self.alpha_thick
        if ax1:
            # for loss chart
            h_lt, = self.ax1_plot(lt_sma_x, lt_sma_y, self.linestyles["loss_train_sma"],
                             label="train loss (avg %d)" % (self.averages_period,),
                             alpha=alpha_sma)
            h_lv, = self.ax1_plot(lv_sma_x, lv_sma_y, self.linestyles["loss_val_sma"],
                             label="val loss (avg %d)" % (self.averages_period,),
                             alpha=alpha_sma)
            handles.extend([h_lt, h_lv])
        if ax2:
            # for error chart
            h_at, = ax2.plot(at_sma_x, at_sma_y, self.linestyles["err_train_sma"],
                             label="train err (avg %d)" % (self.averages_period,),
                             alpha=alpha_sma)
            h_av, = ax2.plot(av_sma_x, av_sma_y, self.linestyles["err_val_sma"],
                             label="err. val. (avg %d)" % (self.averages_period,),
                             alpha=alpha_sma)
            handles.extend([h_at, h_av])

        return handles

    def _redraw_regressions(self):
        """Draw the moving regressions of each line, i.e. the predictions of
        future values.

        If regressions have been deactived in the constructor, this function
        will do nothing.

        Returns:
            List of handles (one per line).
        """
        if not self.show_regressions:
            return []

        handles = []
        ax1 = self.ax_loss
        ax2 = self.ax_err

        # calculate future values for loss train (lt), loss val (lv),
        # err train (at) and err val (av)
        if ax1:
            # for loss chart
            lt_regression = self._calc_regression(self.values_loss_train.keys(),
                                                  self.values_loss_train.values())
            lv_regression = self._calc_regression(self.values_loss_val.keys(),
                                                  self.values_loss_val.values())
        # predicting error values isnt necessary if theres no err chart
        if ax2:
            # for error chart
            at_regression = self._calc_regression(self.values_err_train.keys(),
                                                  self.values_err_train.values())
            av_regression = self._calc_regression(self.values_err_val.keys(),
                                                  self.values_err_val.values())

        # plot the predicted values
        alpha_regression = self.alpha_thick
        if ax1:
            # for loss chart
            h_lt, = self.ax1_plot(lt_regression[0], lt_regression[1],
                             self.linestyles["loss_train_regression"],
                             label="loss train regression",
                             alpha=alpha_regression)
            h_lv, = self.ax1_plot(lv_regression[0], lv_regression[1],
                             self.linestyles["loss_val_regression"],
                             label="loss val. regression",
                             alpha=alpha_regression)
            handles.extend([h_lt, h_lv])
        if ax2:
            # for error chart
            h_at, = ax2.plot(at_regression[0], at_regression[1],
                             self.linestyles["err_train_regression"],
                             label="err train regression",
                             alpha=alpha_regression)
            h_av, = ax2.plot(av_regression[0], av_regression[1],
                             self.linestyles["err_val_regression"],
                             label="err val. regression",
                             alpha=alpha_regression)
            handles.extend([h_at, h_av])

        return handles

    def _calc_sma(self, x_values, y_values):
        """Calculate the moving average for one line (given as two lists, one
        for its x-values and one for its y-values).

        Args:
            x_values: x-coordinate of each value.
            y_values: y-coordinate of each value.

        Returns:
            Tuple (x_values, y_values), where x_values are the x-values of
            the line and y_values are the y-values of the line.
        """
        result_y, last_ys = [], []
        running_sum = 0
        period = self.averages_period
        # use a running sum here instead of avg(), should be slightly faster
        for y_val in y_values:
            last_ys.append(y_val)
            running_sum += y_val
            if len(last_ys) > period:
                poped_y = last_ys.pop(0)
                running_sum -= poped_y
            result_y.append(float(running_sum) / float(len(last_ys)))
        return (x_values, result_y)

    def _calc_regression(self, x_values, y_values):
        """Calculate the regression for one line (given as two lists, one
        for its x-values and one for its y-values).

        Args:
            x_values: x-coordinate of each value.
            y_values: y-coordinate of each value.

        Returns:
            Tuple (x_values, y_values), where x_values are the predicted x-values
            of the line and y_values are the predicted y-values of the line.
        """
        if not x_values or len(x_values) < 2:
            return ([], [])

        # This currently assumes that the last added x-value for the line
        # was indeed that highest x-value.
        # This could be avoided by tracking the max value for each line.
        last_x = x_values[-1]
        nb_values = len(x_values)

        # Compute regression lines based on n_backwards epochs
        # in the past, e.g. based on the last 10 values.
        # n_backwards is calculated relative to the current epoch
        # (e.g. at epoch 100 compute based on the last 10 values,
        # at 200 based on the last 20 values...). It has a minimum (e.g. never
        # use less than 5 epochs (unless there are only less than 5 epochs))
        # and a maximum (e.g. never use more than 1000 epochs).
        # The minimum prevents bad predictions.
        # The maximum
        #   a) is better for performance
        #   b) lets the regression react faster in case you change something
        #      in the hyperparameters after a long time of training.
        n_backward = int(nb_values * self.poly_backward_perc)
        n_backward = max(n_backward, self.poly_n_backward_min)
        n_backward = min(n_backward, self.poly_n_backward_max)

        # Compute the regression lines for the n_forward future epochs.
        # n_forward also has a reletive factor, as well as minimum and maximum
        # values (see n_backward).
        n_forward = int(nb_values * self.poly_forward_perc)
        n_forward = max(n_forward, self.poly_n_forward_min)
        n_forward = min(n_forward, self.poly_n_forward_max)

        # return nothing of the values turn out too low
        if n_backward <= 1 or n_forward <= 0:
            return ([], [])

        # create/train the regression model
        fit = np.polyfit(x_values[-n_backward:], y_values[-n_backward:],
                         self.poly_degree)
        poly = np.poly1d(fit)

        # calculate future x- and y-values
        # we use last_x to last_x+n_forward here instead of
        #        last_x+1 to last_x+1+n_forward
        # so that the regression line is better connected to the current line
        # (no visible gap)
        future_x = [i for i in np.arange(last_x, last_x + n_forward, 1.)]
        future_y = [poly(x_idx) for x_idx in future_x]

        return (future_x, future_y)
class MultiPlotter(object):
    """Class to plot loss and error charts (for training and validation data)."""

    def __init__(self,
                 title=None,
                 save_to_filepath=None,
                 show_regressions=False,
                 show_averages=False,
                 fix_ylim=True,
                 show_plot_window=True,
                 epo_max=None,
                 y_label="Accuracy",
                 x_label="Epoch"):
        """Constructs the plotter.

        Args:
            title: An optional title which will be shown at the top of the
                plot. E.g. the name of the experiment or some info about it.
                If set to None, no title will be shown. (Default is None.)
            save_to_filepath: The path to a file in which the plot will be saved,
                e.g. "/tmp/last_plot.png". If set to None, the chart will not be
                saved to a file. (Default is None.)
            show_regressions: Whether or not to show a regression, indicating
                where each line might end up in the future.
            show_averages: Whether to plot moving averages in the charts for
                each line (so for loss train, loss val, ...). This value may
                only be True or False. To change the interval (default is 20
                epochs), change the instance variable "averages_period" to the new
                integer value. (Default is True.)
            show_loss_plot: Whether to show the chart for the loss values. If
                set to False, only the error chart will be shown. (Default
                is True.)
            show_err_plot: Whether to show the chart for the error value. If
                set to False, only the loss chart will be shown. (Default is True.)
            show_plot_window: Whether to show the plot in a window (True)
                or hide it (False). Hiding it makes only sense if you
                set save_to_filepath. (Default is True.)
            x_label: Label on the x-axes of the charts. Reasonable choices
                would be: "Epoch", "Batch" or "Example". (Default is "Epoch".)
        """
        assert save_to_filepath is not None or show_plot_window

        self.title = title
        self.title_fontsize = 14
        self.show_regressions = show_regressions
        self.show_averages = show_averages
        self.fix_ylim = fix_ylim
        self.show_plot_window = show_plot_window
        self.epo_max = epo_max
        self.save_to_filepath = save_to_filepath
        self.x_label = x_label
        self.y_label = y_label

        # alpha values
        # 0.8 = quite visible line
        # 0.5 = moderately visible line
        # thick is used for averages and regression (also for the main values,
        # if there are no averages),
        # thin is used for the main values
        self.alpha_thick = 0.8
        self.alpha_thin = 0.5

        # the interval for the moving averages, e.g. 20 = average over 20 points
        self.averages_period = 20

        # these values deal with the regression
        self.poly_forward_perc = 0.1
        self.poly_backward_perc = 0.2
        self.poly_n_forward_min = 5
        self.poly_n_backward_min = 10
        self.poly_n_forward_max = 100
        self.poly_n_backward_max = 100
        self.poly_degree = 1

        # whether to show grids in both charts
        self.grid = True

        # the styling of the lines
        # sma = simple moving average
        self.linestyles = {
            "loss_train": "r-",
            "loss_train_sma": "r-",
            "loss_train_regression": "r:",
            "loss_val": "b-",
            "loss_val_sma": "b-",
            "loss_val_regression": "b:",
            "err_train": "r-",
            "err_train_sma": "r-",
            "err_train_regression": "r:",
            "err_val": "b-",
            "err_val_sma": "b-",
            "err_val_regression": "b:"
        }
        # different linestyles for the first epoch (if only one value is available),
        # because no line can then be drawn (needs 2+ points) and only symbols will
        # be shown.
        # No regression here, because regression always has at least at least
        # two xy-points (last real value and one (or more) predicted values).
        # No averages here, because the average over one value would be identical
        # to the value anyways.
        self.linestyles_one_value = {
            "loss_train": "rs-",
            "loss_val": "b^-",
            "err_train": "rs-",
            "err_val": "b^-"
        }

        # these values will be set in _initialize_plot() upon the first call
        # of redraw()
        # fig: the figure of the whole plot
        # ax_loss: loss chart (left)
        # ax_err: error chart (right)
        self.fig = None
        self.ax_err = None

        # dictionaries with x, y values for each line
        self.other_value = OrderedDict()

    def add_values(self, x_index, other_value=None,
                   redraw=True):
        """Function to add new values for each line for a specific x-value (e.g.
        a specific epoch).

        Meaning of the values / lines:

        Values that are None will be ignored.
        Values that are INF or NaN will be ignored, but create a warning.

        It is currently assumed that added values follow logically after
        each other (progressive order), so the first x_index might be 1 (first entry),
        then 2 (second entry), then 3 (third entry), ...
        Not allowed would be e.g.: 10, 11, 5, 7, ...
        If that is not the case, you will get a broken line graph.

        Args:
            x_index: The x-coordinate, e.g. x_index=5 might represent Epoch 5.
            redraw: Whether to redraw the plot immediatly after receiving the
                new values. This is reasonable if you add values once at the end
                of every epoch. If you add many values in a row, set this to
                False and call redraw() at the end (significantly faster).
                (Default is True.)
        """
        assert isinstance(x_index, (int, float))
        if self.epo_max and x_index > self.epo_max:
            return

        if other_value is not None:
            if x_index == 0.:
                for key in other_value.keys():
                    self.other_value[key] = OrderedDict()
            for key, val in other_value.items():
                val2 = ignore_nan_and_inf(val, key, x_index)
                if val2 is not None:
                    self.other_value[key][x_index] = val2

        if redraw:
            self.redraw()

    def block(self):
        """Function to show the plot in a blocking way.

        This should be called at the end of your program. Otherwise the
        chart will be closed automatically (at the end).
        By default, the plot is shown in a non-blocking way, so that the
        program continues execution, which causes it to close automatically
        when the program finishes.

        This function will silently do nothing if show_plot_window was set
        to False in the constructor.
        """
        if self.show_plot_window:
            plt.figure(self.fig.number)
            plt.show()

    def save_plot(self, filepath):
        """Saves the current plot to a file.

        Args:
            filepath: The path to the file, e.g. "/tmp/last_plot.png".
        """
        self.fig.savefig(filepath, bbox_inches="tight")

    def _initialize_plot(self):
        """Creates empty figure and axes of the plot and shows it in a new window.
        """
        fig, ax = plt.subplots(ncols=1, figsize=(12, 8))
        self.fig = fig
        self.ax_err = ax

        if self.ax_err is not None:
            self.ax_err.set_ylim([0., 1.])

        # set_position is neccessary here in order to make space at the bottom
        # for the legend
        for ax in [self.ax_err]:
            if ax is not None:
                box = ax.get_position()
                ax.set_position([box.x0, box.y0 + box.height * 0.1,
                                 box.width, box.height * 0.9])

        # draw the title
        # it seems to be necessary to set the title here instead of in redraw(),
        # otherwise the title is apparently added again and again with every
        # epoch, making it ugly and bold
        if self.title is not None:
            self.fig.suptitle(self.title, fontsize=self.title_fontsize)

        if self.show_plot_window:
            plt.show(block=False)

    def redraw(self):
        """Redraws the plot with the current values.

        This is a full redraw and includes recalculating averages and regressions.
        It should not be called many times per second as that would be slow.
        Calling it every couple seconds should create no noticeable slowdown though.

        Args:
            epoch: The index of the current epoch, starting at 0.
            val_err: All of the validation error values of each
                epoch (list of floats).
        """
        # initialize the plot if it's the first redraw
        if self.fig is None:
            self._initialize_plot()

        # activate the plot, in case another plot was opened since the last call
        plt.figure(self.fig.number)

        # shorter local variables
        ax2 = self.ax_err

        # set chart titles, x-/y-labels and grid
        for ax, label in zip([ax2], [self.y_label]):
            if ax:
                ax.clear()
                ax.set_title(label)
                ax.set_ylabel(label)
                ax.set_xlabel(self.x_label)
                ax.grid(self.grid)

        # Plot main lines, their averages and the regressions (predictions)
        self._redraw_othervalue()
        # self._redraw_averages()
        # self._redraw_regressions()

        # Add legends (below both chart)
        ncol = 1
        labels = []
        for key in self.other_value.keys():
            labels.extend([key])
            ncol += 1

        # if self.show_averages:
        #     labels.extend(["$CHART train (avg %d)" % (self.averages_period,),
        #                    "$CHART val. (avg %d)" % (self.averages_period,)])
        #     ncol += 1
        # if self.show_regressions:
        #     labels.extend(["$CHART train (regression)",
        #                    "$CHART val. (regression)"])
        #     ncol += 1

        if ax2:
            ax2.legend([label.replace("$CHART", "err.") for label in labels],
                       loc="upper center",
                       bbox_to_anchor=(0.5, -0.08),
                       ncol=ncol)

        if self.fix_ylim:
            ax2.set_ylim([0., 1.])

        # if self.epo_max:
        #     if ax1:
        #         ax1.set_xlim([0., self.epo_max])
        #     if ax2:
        #         ax2.set_xlim([0., self.epo_max])

        plt.draw()

        # save the redrawn plot to a file upon every redraw.
        if self.save_to_filepath is not None:
            self.save_plot(self.save_to_filepath)

    def _redraw_othervalue(self):
        """Draw the moving averages of each line.

        If moving averages has been deactived in the constructor, this function
        will do nothing.

        Returns:
            List of handles (one per line).
        """
        # abort if moving averages have been deactivated

        handles = []
        ax1 = self.ax_err
        if not ax1:
            return handles
        # for loss chart
        for key, od in self.other_value.items():
            # plot the xy-values
            h_lt, = ax1.plot(od.keys(), od.values(),
                             label=key)
            handles.extend([h_lt])

        return handles

    def _redraw_averages(self):
        """Draw the moving averages of each line.

        If moving averages has been deactived in the constructor, this function
        will do nothing.

        Returns:
            List of handles (one per line).
        """
        # abort if moving averages have been deactivated
        if not self.show_averages:
            return []

        handles = []
        ax2 = self.ax_err

        # calculate the xy-values
        if ax2:
            # for error chart
            (at_sma_x, at_sma_y) = self._calc_sma(self.values_err_train.keys(),
                                                  self.values_err_train.values())
            (av_sma_x, av_sma_y) = self._calc_sma(self.values_err_val.keys(),
                                                  self.values_err_val.values())

        # plot the xy-values
        alpha_sma = self.alpha_thick
        if ax2:
            # for error chart
            h_at, = ax2.plot(at_sma_x, at_sma_y, self.linestyles["err_train_sma"],
                             label="train err (avg %d)" % (self.averages_period,),
                             alpha=alpha_sma)
            h_av, = ax2.plot(av_sma_x, av_sma_y, self.linestyles["err_val_sma"],
                             label="err. val. (avg %d)" % (self.averages_period,),
                             alpha=alpha_sma)
            handles.extend([h_at, h_av])

        return handles

    def _redraw_regressions(self):
        """Draw the moving regressions of each line, i.e. the predictions of
        future values.

        If regressions have been deactived in the constructor, this function
        will do nothing.

        Returns:
            List of handles (one per line).
        """
        if not self.show_regressions:
            return []

        handles = []
        ax2 = self.ax_err

        # calculate future values for loss train (lt), loss val (lv),
        # err train (at) and err val (av)
        # predicting error values isnt necessary if theres no err chart
        if ax2:
            # for error chart
            at_regression = self._calc_regression(self.values_err_train.keys(),
                                                  self.values_err_train.values())
            av_regression = self._calc_regression(self.values_err_val.keys(),
                                                  self.values_err_val.values())

        # plot the predicted values
        alpha_regression = self.alpha_thick
        if ax2:
            # for error chart
            h_at, = ax2.plot(at_regression[0], at_regression[1],
                             self.linestyles["err_train_regression"],
                             label="err train regression",
                             alpha=alpha_regression)
            h_av, = ax2.plot(av_regression[0], av_regression[1],
                             self.linestyles["err_val_regression"],
                             label="err val. regression",
                             alpha=alpha_regression)
            handles.extend([h_at, h_av])

        return handles

    def _calc_sma(self, x_values, y_values):
        """Calculate the moving average for one line (given as two lists, one
        for its x-values and one for its y-values).

        Args:
            x_values: x-coordinate of each value.
            y_values: y-coordinate of each value.

        Returns:
            Tuple (x_values, y_values), where x_values are the x-values of
            the line and y_values are the y-values of the line.
        """
        result_y, last_ys = [], []
        running_sum = 0
        period = self.averages_period
        # use a running sum here instead of avg(), should be slightly faster
        for y_val in y_values:
            last_ys.append(y_val)
            running_sum += y_val
            if len(last_ys) > period:
                poped_y = last_ys.pop(0)
                running_sum -= poped_y
            result_y.append(float(running_sum) / float(len(last_ys)))
        return (x_values, result_y)

    def _calc_regression(self, x_values, y_values):
        """Calculate the regression for one line (given as two lists, one
        for its x-values and one for its y-values).

        Args:
            x_values: x-coordinate of each value.
            y_values: y-coordinate of each value.

        Returns:
            Tuple (x_values, y_values), where x_values are the predicted x-values
            of the line and y_values are the predicted y-values of the line.
        """
        if not x_values or len(x_values) < 2:
            return ([], [])

        # This currently assumes that the last added x-value for the line
        # was indeed that highest x-value.
        # This could be avoided by tracking the max value for each line.
        last_x = x_values[-1]
        nb_values = len(x_values)

        # Compute regression lines based on n_backwards epochs
        # in the past, e.g. based on the last 10 values.
        # n_backwards is calculated relative to the current epoch
        # (e.g. at epoch 100 compute based on the last 10 values,
        # at 200 based on the last 20 values...). It has a minimum (e.g. never
        # use less than 5 epochs (unless there are only less than 5 epochs))
        # and a maximum (e.g. never use more than 1000 epochs).
        # The minimum prevents bad predictions.
        # The maximum
        #   a) is better for performance
        #   b) lets the regression react faster in case you change something
        #      in the hyperparameters after a long time of training.
        n_backward = int(nb_values * self.poly_backward_perc)
        n_backward = max(n_backward, self.poly_n_backward_min)
        n_backward = min(n_backward, self.poly_n_backward_max)

        # Compute the regression lines for the n_forward future epochs.
        # n_forward also has a reletive factor, as well as minimum and maximum
        # values (see n_backward).
        n_forward = int(nb_values * self.poly_forward_perc)
        n_forward = max(n_forward, self.poly_n_forward_min)
        n_forward = min(n_forward, self.poly_n_forward_max)

        # return nothing of the values turn out too low
        if n_backward <= 1 or n_forward <= 0:
            return ([], [])

        # create/train the regression model
        fit = np.polyfit(x_values[-n_backward:], y_values[-n_backward:],
                         self.poly_degree)
        poly = np.poly1d(fit)

        # calculate future x- and y-values
        # we use last_x to last_x+n_forward here instead of
        #        last_x+1 to last_x+1+n_forward
        # so that the regression line is better connected to the current line
        # (no visible gap)
        future_x = [i for i in np.arange(last_x, last_x + n_forward, 1.)]
        future_y = [poly(x_idx) for x_idx in future_x]

        return (future_x, future_y)
