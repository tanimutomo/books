import numpy as np
import matplotlib.pyplot as plt

def create_dataset(N, function=np.sin):
    # train_data
    train_x = np.random.uniform(0, 7, N)
    train_y = function(train_x)
    # ground_truth
    gt_x = np.linspace(0, 7)
    gt_y = function(gt_x)

    return [train_x, train_y], [gt_x, gt_y]

def plotting(ground_truth, train_data, prediction):
    plt.plot(train_data[0],train_data[1], "o")
    plt.plot(ground_truth[0], ground_truth[1], color='r')
    plt.plot(ground_truth[0], prediction[0], color='b')
    plt.plot(ground_truth[0], prediction[1], linestyle='--', color='c')
    plt.plot(ground_truth[0], prediction[2], linestyle='--', color='c')
    
    plt.xlim([-0.5, 7])
    plt.ylim([-3, 3])

    plt.show()

