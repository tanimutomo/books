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
    plt.plot(train_data[0].numpy(), train_data[1].numpy(), "o")
    plt.plot(ground_truth[0].numpy(), ground_truth[1].numpy(), color='r')
    plt.plot(ground_truth[0].numpy(), prediction[0].numpy(), color='b')
    plt.plot(ground_truth[0].numpy(), prediction[1].numpy(), linestyle='--', color='c')
    plt.plot(ground_truth[0].numpy(), prediction[2].numpy(), linestyle='--', color='c')
    
    plt.xlim([-0.5, 7])
    plt.ylim([-3, 3])

    plt.show()

