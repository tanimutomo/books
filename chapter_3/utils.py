import torch
import matplotlib.pyplot as plt

def create_dataset(N, function=torch.sin):
    # train_data
    train_x = 7 * torch.rand(10)
    train_y = function(train_x)
    # ground_truth
    gt_x = torch.linspace(0, 7)
    gt_y = function(gt_x)

    return [train_x, train_y], [gt_x, gt_y]

def plotting(ground_truth, train_data, prediction):
    plt.plot(list(train_data[0]), list(train_data[1]), "o")
    plt.plot(list(ground_truth[0]), list(ground_truth[1]), color='r')
    plt.plot(list(ground_truth[0]), list(prediction[0]), color='b')
    plt.plot(list(ground_truth[0]), list(prediction[1]), linestyle='--', color='c')
    plt.plot(list(ground_truth[0]), list(prediction[2]), linestyle='--', color='c')
    
    plt.xlim([-0.5, 7])
    plt.ylim([-3, 3])

    plt.show()

