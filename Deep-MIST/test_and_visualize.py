import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import auc
from tqdm import tqdm

from deepmist.datasets import build_dataset, DataLoaderX
from deepmist.metrics import mIoUMetric, nIoUMetric, PdFaMetric, ROCMetric
from deepmist.models import build_model, run_model
from deepmist.utils import ordered_yaml, make_dir, draw_feature_map


class Trainer(object):
    def __init__(self, args):
        # config
        with open(args.config, mode='r') as f:
            self.cfg = yaml.load(f, Loader=ordered_yaml()[0])

        # device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # dataloader
        self.val_dataset = build_dataset(self.cfg['dataset'], mode='val')
        self.val_loader = DataLoaderX(self.val_dataset, batch_size=1,
                                      num_workers=self.cfg['test']['num_workers'], pin_memory=True)

        # model
        self.model, self.model_name = build_model(self.cfg['model']['network'])
        if args.DataParallel:
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        # metric
        self.mIoU_metric = mIoUMetric()
        self.nIoU_metric = nIoUMetric()
        self.PdFa_metric = PdFaMetric()
        self.bins = 200
        self.ROC_metric = ROCMetric(bins=self.bins)

        # checkpoint
        checkpoint = torch.load(self.cfg['test']['checkpoint'], map_location='cpu')
        self.model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Successfully load the checkpoint from {self.cfg['test']['checkpoint']}.\n")

        # visualize
        self.pred_result_visualize_dir = os.path.join(self.cfg['test']['exp_root'], 'pred_result_visualize')
        self.feature_map_visualize_dir = os.path.join(self.cfg['test']['exp_root'], 'feature_map_visualize')
        make_dir(self.pred_result_visualize_dir)
        make_dir(self.feature_map_visualize_dir)

    def test(self):
        print(f"[testset]\n")
        self.model.eval()
        with torch.no_grad():
            for iter_idx, data in enumerate(tqdm(self.val_loader)):
                frames, mask, name = data
                frames, mask = frames.to(self.device), mask.to(self.device)
                preds, feature_maps = run_model(self.model, self.model_name, False, frames)
                pred = preds[0]

                # update metrics
                self.mIoU_metric.update(pred, mask)
                self.nIoU_metric.update(pred, mask)
                self.PdFa_metric.update(pred, mask)
                self.ROC_metric.update(pred, mask)

                # # prediction result visualize
                # pred_sigmoid = torch.sigmoid(pred)
                # final_pred = pred_sigmoid.data.cpu().numpy()[0, 0, :, :]
                # mask_pred = Image.fromarray(np.uint8(final_pred * 255))
                # save_dir = os.path.join(self.pred_result_visualize_dir, name[0].split('/')[0])
                # make_dir(save_dir)
                # mask_pred.save(os.path.join(self.pred_result_visualize_dir, name[0]))

                # feature map visualize
                img_path = os.path.join(self.cfg['dataset']['root'], 'image', name[0])
                save_dir = os.path.join(self.feature_map_visualize_dir, name[0].split('/')[0])
                make_dir(save_dir)
                draw_feature_map(feature_maps, img_path=img_path, save_dir=save_dir,
                                 name=name[0].split('/')[1].split('.')[0])

        # get metrics
        # print('FPS=%.3f' % ((iter_idx + 1) / (time.time() - start_time)))
        _, mIoU = self.mIoU_metric.get()
        nIoU, _ = self.nIoU_metric.get()
        FA, PD = self.PdFa_metric.get()
        tp_rates, fp_rates, recall, precision, f_score = self.ROC_metric.get()
        AUC = auc(fp_rates, tp_rates)
        precision_05 = precision[(self.bins + 1) // 2 + 1]
        recall_05 = recall[(self.bins + 1) // 2 + 1]
        f_score_05 = f_score[(self.bins + 1) // 2 + 1]

        # log the test info
        message = f'\n[mIoU:{mIoU:.5f}][nIoU:{nIoU:.5f}][PD:{PD:.5f}][FA:{FA:.5e}][AUC:{AUC:.5f}]' \
                  f'[precision:{precision_05:.5f}][recall:{recall_05:.5f}][f_score:{f_score_05:.5f}]'
        print(message)


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    trainer = Trainer(args)
    trainer.test()


def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Deep-MIST Test')
    parser.add_argument('--config', type=str, default='/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/configs/multiframe/LS_STDNet/test_LS_STDNet_MIST.yaml',
                        help='path to config file')
    parser.add_argument('--device', type=str, default='cuda', help='device (use cuda or cpu)')
    parser.add_argument('--DataParallel', default=True, help='use DataParallel or not')
    parser.add_argument('--gpu_ids', type=str, default='2', help='the ids of gpus')

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    main(args)
