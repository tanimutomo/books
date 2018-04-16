import numpy as np
import matplotlib.pyplot as plt


def create_dataset(N, function=np.sin):
    train_x = np.random.uniform(0, 7, N)
    train_y = function(train_x)

    train_y += 0.5 * np.random.randn(*train_y.shape)

    return train_x, train_y


def plotting(ground_truth, training_data, prediction):
    plt.plot(training_data[0], training_data[1], "o")
    plt.plot(ground_truth[0], ground_truth[1], color='r')
    plt.plot(ground_truth[0], prediction[0], color='b')
    plt.plot(ground_truth[0], prediction[1], linestyle='--', color='c')
    plt.plot(ground_truth[0], prediction[2], linestyle='--', color='c')
    plt.xlim([-0.5, 7])
    plt.ylim([-3, 3])
    plt.show()


