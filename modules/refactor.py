from typing import List, Union
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output, display

class NeuralNetwork:
    FONT_SIZE = 16
    FIGURE_SIZE = (16, 6)
    RATIO_FIGURE_SIZE = (16, 3)
    RATIO_LINE_SPACING = 0.05

    def __init__(self):
        self.print_cache = []
        self.epoch_performance = []
        self.epoch_index = 0

    def label_plt(self, title: str, xlab: str, ylab: str) -> None:
        plt.rcParams['font.size'] = self.FONT_SIZE
        plt.title(title)
        plt.xlabel(xlab)
        plt.ylabel(ylab)

    def print_progress(self) -> None:
        self.clear_and_display_output()
        self.print_epoch_performance()

        if self.epoch_index > 0:
            self.plot_epoch_progress()
            self.plot_overfitting_ratios()

    def clear_and_display_output(self) -> None:
        clear_output()
        for line in self.print_cache:
            if isinstance(line, pd.core.frame.DataFrame):
                display(line)
            elif isinstance(line, str):
                print(line)

    def print_epoch_performance(self) -> None:
        print('Performance By Epoch:')
        display(pd.DataFrame(self.epoch_performance))

    def plot_epoch_progress(self) -> None:
        plt.figure(figsize=self.FIGURE_SIZE)

        min_epoch, epochs, ep, misses_train, misses_val = self.get_plot_data()

        self.plot_lines(epochs, [ep.loss_train, ep.loss_val, misses_train, misses_val],
                        ['Loss: Training', 'Loss: Validation', 'Misclassif Rate: Training', 'Misclassif Rate: Validation'],
                        ['b', 'b', 'r', 'r'], ['--', None, '--', None])

        plt.legend(loc='upper left')
        plt.ylim(0, 1.05*ep.loss_val[min_epoch+1])
        plt.xticks(epochs)
        self.label_plt('Loss and Misclassification Rate vs. Epoch', 'After Epoch', 'Value')
        plt.show()

    def plot_overfitting_ratios(self) -> None:
        plt.figure(figsize=self.RATIO_FIGURE_SIZE)

        min_epoch, epochs, ep, misses_train, misses_val = self.get_plot_data()

        ovf_ratio_loss = ep.loss_val/ep.loss_train
        ovf_ratio_misses = misses_val/misses_train

        self.plot_lines(epochs, [ovf_ratio_loss, ovf_ratio_misses],
                        ['Loss', 'Misclassif Rate'], ['b', 'r'], ['dotted', 'dotted'])

        plt.legend(loc='upper left')
        max_ratio = np.amax([ovf_ratio_loss, ovf_ratio_misses])
        self.plot_ratio_grid(max_ratio)
        plt.ylim(1, 1.01*max_ratio)
        plt.xticks(epochs)
        self.label_plt('Overfitting Ratios vs. Epoch', 'After Epoch', 'Validation to Training Ratio')
        plt.show()

    def get_plot_data(self):
        min_epoch = np.maximum(0, self.epoch_index-5)
        epochs = range(min_epoch, self.epoch_index+1)
        ep = self.epoch_performance[min_epoch:self.epoch_index+1]
        misses_train = 1-ep.acc_train
        misses_val = 1-ep.acc_val
        return min_epoch, epochs, ep, misses_train, misses_val

    def plot_lines(self, epochs, values, labels, colors, styles):
        for value, label, color, style in zip(values, labels, colors, styles):
            plt.plot(epochs, value, label=label, c=color, linestyle=style)

    def plot_ratio_grid(self, max_ratio) -> None:
        for i in np.arange(1, max_ratio, self.RATIO_LINE_SPACING):
            plt.axhline(y=i, c='gray', linestyle='dotted', alpha=0.5)