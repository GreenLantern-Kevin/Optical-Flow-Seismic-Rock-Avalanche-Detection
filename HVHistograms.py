import matplotlib.pyplot as plt
import numpy as np
import cv2


def plot_channel_histogram(channel, channel_name="Channel",color='b'):
    plt.figure(figsize=(8, 6))
    plt.title(f"{channel_name} Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    hist = cv2.calcHist([channel], [0], None, [256], [1,256])

    x = np.arange(256)

    plt.fill_between(x, hist.flatten(), color=color, alpha=0.6)
    plt.plot(x, hist, color=color)

    range = (1, 100)
    plt.xlim(range)
    plt.grid()
    plt.show()



h_channel = cv2.imread('HChannel_Figure_Path')
v_channel = cv2.imread('VChannel_Figure_Path')

plot_channel_histogram(h_channel, "Hue",color='r')
plot_channel_histogram(v_channel, "Value",color='b')
