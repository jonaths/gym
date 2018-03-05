#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


class PlotHistogramRT:
    """
    Crea un histograma y lo actualiza cada que se llama update en la misma figura.
    """

    def __init__(self, series_number, bins, pause=0.01):
        self.bins = bins
        self.pause = pause
        self.series_number = series_number
        self.width = 0.7
        plt.ion()

    def update(self, new_x, new_y):
        """
        Actualiza el histograma
        :param new_y: numpy con los nuevos valores en y [[1, 2, 3], [4, 5, 6]]
        :param new_x: numpy con los nuevos valores en x [1, 2, 3]
        :return:
        """

        new_y = np.asarray(new_y)

        # valida que las medidas nuevas sean las mismas que las establecidas en el constructor
        if new_y.shape[0] != self.series_number \
                or new_y.shape[1] != self.bins \
                or new_x.shape[0] != self.bins:
            raise ValueError('Incorrect new_y or new_x shape')

        # print np.asarray(new_x).shape
        #
        plt.clf()
        plt.cla()
        for ind in range(len(new_y)):
            plt.bar(new_x, new_y[ind])
        plt.draw()
        plt.pause(self.pause)


if __name__ == "__main__":
    plotter = PlotHistogramRT(3, 51)
    x = np.arange(51)

    # plotter.update(x, y)
    while True:
        y = np.random.randn(3, 51)
        plotter.update(x, y)
