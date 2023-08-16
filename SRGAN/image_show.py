import torch
from matplotlib import pyplot as plt
import cv2
def image_show(hr,lr,outputs):
    # print(hr.shape)
    # chang the image from 3x224x224 to 224x224x3
    # hr = torch.from_numpy(hr)
    # lr = torch.from_numpy(lr)
    # outputs = torch.from_numpy(outputs)
    # # hr = hr.permute(2, 0, 1)
    # # lr = lr.permute(2, 0, 1)
    # # outputs = outputs.permute(2, 0, 1)
    # hr = hr.permute(2, 0, 1)
    # lr = lr.permute(2, 0, 1)
    # outputs = outputs.permute(2, 0, 1)
    # hr = hr.numpy()
    # lr = lr.numpy()
    # outputs = outputs.numpy()

    # show the hr,lr and outputs images
    # print(hr.shape)
    fig, axs = plt.subplots(1, 3, figsize=(20, 8))
    axs[0].imshow(cv2.cvtColor(hr, cv2.COLOR_BGR2RGB))
    #axs[0].imshow(hr)
    axs[0].set_title('HR')
    axs[1].imshow(cv2.cvtColor(lr, cv2.COLOR_BGR2RGB))
    #axs[0].imshow(lr)
    axs[1].set_title('LR')
    axs[2].imshow(cv2.cvtColor(outputs, cv2.COLOR_BGR2RGB))
    #axs[0].imshow(outputs)
    axs[2].set_title('SRGAN')

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])