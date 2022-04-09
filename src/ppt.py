# This program was used to make the plots for the PowerPoint presentation.
from transformers import set_seed
import matplotlib.pyplot as plt
import numpy as np


def scatter(x, y, r=1.0, x_shift=0.0, y_shift=0.0):
    radii = r * np.sqrt(x)
    rotations = 2 * np.pi * y
    x = x_shift + np.cos(rotations) * radii
    y = y_shift + np.sin(rotations) * radii
    for add_on in [None, 'pairwise', 'centroid']:
        plt.figure()
        plt.plot(x, y, 'ro')
        if add_on == 'pairwise':
            plt.plot(x, y, 'w')
        elif add_on == 'centroid':
            x_mean, y_mean = np.mean(x), np.mean(y)
            for x_point, y_point in zip(x, y):
                plt.annotate("", xy=(x_mean, y_mean), xytext=(x_point, y_point),
                             arrowprops=dict(color='w', headwidth=1,
                                             headlength=1, width=0.1))
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.xticks(np.arange(0.125, 1, 0.125))
        plt.yticks(np.arange(0.125, 1, 0.125))
        ax = plt.gca()
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        plt.tight_layout()
        plt.savefig(f"scatter-{r}-{x_shift}-{y_shift}-{add_on}.pdf")
        plt.close()


def ts(metric):
    x = [0, 1, 2, 3, 4, 5, 6]
    y = [0.2, 0.4, 0.3, 0.5, 0.6, 0.5, 0.7]
    plt.figure()
    plt.annotate("", xy=(x[-1], y[-1]), xytext=(x[-2], y[-2]),
                 arrowprops=dict(color='orange', headwidth=10, headlength=10,
                                 width=0.1))
    plt.plot(x, y, color='orange', alpha=1.0)
    plt.plot(x[:-1], y[:-1], 'o', color='orange')
    plt.ylabel(metric)
    plt.xlabel("Time Index")
    plt.ylim(0, 1)
    plt.title(f"{metric} Time Series")
    plt.savefig(f"ts-{metric}.pdf")
    plt.close()


if __name__ == '__main__':
    set_seed(0)
    rand_x, rand_y = np.random.rand(100), np.random.rand(100)
    with plt.style.context('dark_background'):
        scatter(rand_x, rand_y, r=0.1, x_shift=0.5, y_shift=0.5)
        scatter(rand_x, rand_y, r=0.2, x_shift=0.5, y_shift=0.5)
        scatter(rand_x, rand_y, r=0.25, x_shift=0.5, y_shift=0.5)
        scatter(rand_x, rand_y, r=0.2, x_shift=0.25, y_shift=0.25)
        scatter(rand_x, rand_y, r=0.2, x_shift=0.75, y_shift=0.75)

        ts("APD")
        ts("SD")
