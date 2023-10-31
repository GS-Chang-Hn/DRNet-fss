"""Util functions"""
import random

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import scipy.misc
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import imageio
loader = transforms.Compose([
    transforms.ToTensor()
])

# @GL query pred 覆盖
def pred_query_cover(query_id, pred):
    # <_io.BufferedReader name='Z:\\czb\\COCO2014\\train2014\\train2014\\COCO_train2014_000000078000.jpg'>
    # query_path = r"Z:\czb\COCO2014\train2014\val2014\COCO_val2014_"+"0"*(12-len(query_id)) + query_id + ".jpg"
    query_path = "Z:\\changzb\\data_copy\\data\\VOC2012\\JPEGImages\\" + query_id + ".jpg"
    query_img = Image.open(query_path).convert('RGB')
    query_img = F.interpolate(loader(query_img).unsqueeze(0), size=pred.shape[-2:], mode='bilinear')

    # generalized_imshow(query_img.squeeze(0), "query_util")
    # generalized_imshow(pred.cpu(), "pred_util")

    # @GL merge
    # img = cv2.imread('./Cyc/pred_max.jpg')  # 读取照片
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imwrite("./Cyc/hsv.jpg", hsv)
    # lower_blue = np.array([110, 100, 100])
    # upper_blue = np.array([130, 255, 255])
    # mask = cv2.inRange(hsv, lower_blue, upper_blue)
    #
    # rows, cols, channels = img.shape
    # for i in range(rows):
    #     for j in range(cols):
    #         if mask[i, j] == 255:  # 像素点: 255 = 白色
    #             # img[i,j]=(0,0,255) # 白色 -> 红色
    #             # img2[i,j]=(0,255,0) # 白色 -> 绿色
    #             # img2[i,j]=(255,0,0) # 白色 -> 蓝色
    #             # img2[i,j]=(255,0,255) # 白色 -> 品红色
    #             # img2[i,j]=(0,255,255) # 白色 -> 黄色
    #             img[i, j] = (255, 255, 0)  # 白色 -> 青色
    #
    # generalized_imshow(img, "img")
    # dst = cv2.bitwise_not(img)
    # cv2.imwrite("./Cyc/hsv0.jpg", dst)
    # dst = cv2.bitwise_not(dst)
    # cv2.imwrite("./Cyc/hsv1.jpg", dst)
    # cv2.waitKey(0)

    plt.imshow(query_img.squeeze().permute(1, 2, 0), alpha=0.7)
    # plt.show()

    plt.imshow(pred.permute(1, 2, 0).cpu(), alpha=0.2)
    # plt.show()
    plt.savefig("./CCG_Test/" + query_id + "_merge.jpg", bbox_inches='tight', pad_inches=0)
# @GL 可视化tensor @czb 去刻度和空白
def generalized_imshow(arr, img_name):
    # print(arr.shape, img_name)
    # fig = plt.figure(frameon=False)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)

    if isinstance(arr, torch.Tensor) and arr.shape[0] == 3:
        # if isinstance(arr, torch.Tensor) and (arr.shape[0] == 3 or arr.shape[0] == 1):
        arr = arr.permute(1, 2, 0)
        plt.imshow(arr)
    if arr.shape[0] == 1:
        # plt.imshow(arr.squeeze(), cmap="gray")
        arr = arr.squeeze()
        plt.imshow(arr, cmap="gray")
    plt.axis('off')
    plt.savefig("./CCG_Test/" + img_name + ".jpg", bbox_inches='tight', pad_inches=0)


# @GL 可视化特征图
def show_feature_map(feature_map):
    # print(feature_map.shape)
    # feature_map = feature_map.squeeze(0)
    feature_map = feature_map.detach().cpu().numpy()
    feature_map_num = feature_map.shape[0]
    row_num = np.ceil(np.sqrt(feature_map_num))
    plt.figure()
    for index in range(1, feature_map_num + 1):
        plt.subplot(row_num, row_num, index)
        plt.imshow(feature_map[index - 1], cmap='gray')
        plt.axis('off')
        imageio.imsave("./savefile/" + str(index) + ".png", feature_map[index - 1])
    plt.show()


# @GL 计算余弦相似度
def cosine_similarity(x, y):
    x = x.detach()
    y = y.detach()
    score = np.dot(x, y.T)
    score = np.diag(score)
    score_under = []
    for i in range(len(score)):
        result_face = sum([c * c for c in x[i][:]])
        result_background = sum([d * d for d in y[i][:]])
        score_under.append((result_face * result_background) ** 0.5)
    out = score / score_under
    return out


def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


CLASS_LABELS = {
    'VOC': {
        'all': set(range(1, 21)),
        0: set(range(1, 21)) - set(range(1, 6)),
        1: set(range(1, 21)) - set(range(6, 11)),
        2: set(range(1, 21)) - set(range(11, 16)),
        3: set(range(1, 21)) - set(range(16, 21)),
    },
    'COCO': {
        'all': set(range(1, 81)),
        0: set(range(1, 81)) - set(range(1, 21)),
        1: set(range(1, 81)) - set(range(21, 41)),
        2: set(range(1, 81)) - set(range(41, 61)),
        3: set(range(1, 81)) - set(range(61, 81)),
    }
}


def get_bbox(fg_mask, inst_mask):
    """
    Get the ground truth bounding boxes
    """

    fg_bbox = torch.zeros_like(fg_mask, device=fg_mask.device)
    bg_bbox = torch.ones_like(fg_mask, device=fg_mask.device)

    inst_mask[fg_mask == 0] = 0
    area = torch.bincount(inst_mask.view(-1))
    cls_id = area[1:].argmax() + 1
    cls_ids = np.unique(inst_mask)[1:]

    mask_idx = np.where(inst_mask[0] == cls_id)
    y_min = mask_idx[0].min()
    y_max = mask_idx[0].max()
    x_min = mask_idx[1].min()
    x_max = mask_idx[1].max()
    fg_bbox[0, y_min:y_max + 1, x_min:x_max + 1] = 1

    for i in cls_ids:
        mask_idx = np.where(inst_mask[0] == i)
        y_min = max(mask_idx[0].min(), 0)
        y_max = min(mask_idx[0].max(), fg_mask.shape[1] - 1)
        x_min = max(mask_idx[1].min(), 0)
        x_max = min(mask_idx[1].max(), fg_mask.shape[2] - 1)
        bg_bbox[0, y_min:y_max + 1, x_min:x_max + 1] = 0
    return fg_bbox, bg_bbox
