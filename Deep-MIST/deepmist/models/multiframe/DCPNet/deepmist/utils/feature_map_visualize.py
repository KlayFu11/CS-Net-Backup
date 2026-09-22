import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from deepmist.utils.data_processing import rgb_loader


def featuremap_2_heatmap(feature_map):
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:, 0, :, :] * 0
    for c in range(feature_map.shape[1]):
        heatmap += feature_map[:, c, :, :]
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap


def draw_feature_map(feature_maps, img_path='', save_dir='', name=None):
    # img = rgb_loader(img_path)  # 测试用PIL还是cv2
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if not isinstance(feature_maps, (list, tuple)):
        feature_maps = [feature_maps]
    for idx, feature_map in enumerate(feature_maps):
        heatmap = featuremap_2_heatmap(feature_map)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.5 + img * 0.3
        cv2.imwrite(os.path.join(save_dir, name + '_' + str(idx) + '.png'), superimposed_img)
        # plt.imshow(superimposed_img)
        # plt.show()

        # cv2.imshow("1", superimposed_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
