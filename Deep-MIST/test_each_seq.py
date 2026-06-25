import argparse
import os
import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import auc
from tqdm import tqdm

from deepmist.datasets import DataLoaderX
from deepmist.datasets.MISTDataset_each_seq import MISTDataset
from deepmist.metrics import mIoUMetric, nIoUMetric, PdFaMetric, ROCMetric
from deepmist.models import build_model, run_model
from deepmist.utils import ordered_yaml


def set_seed(seed, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        # slower, more reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # faster
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


class Tester(object):
    def __init__(self, args, seq):
        # seed
        set_seed(args.seed)

        # config
        with open(args.config, mode='r') as f:
            self.cfg = yaml.load(f, Loader=ordered_yaml()[0])

        # device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # dataloader
        dataset_cfg = deepcopy(self.cfg['dataset'])
        dataset_name = dataset_cfg.pop('name')
        if dataset_name == 'MIST':
            self.val_dataset = MISTDataset(**dataset_cfg, seq=seq)
        self.seq = seq
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
        self.bins = 10  # 10, 200
        self.ROC_metric = ROCMetric(bins=self.bins)

        # checkpoint
        checkpoint = torch.load(self.cfg['test']['checkpoint'], map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        print(f"Successfully load the checkpoint from {self.cfg['test']['checkpoint']}.\n")

    def test(self):
        print(f"[testset]\n")
        self.model.eval()
        # reset metrics
        self.mIoU_metric.reset()
        self.nIoU_metric.reset()
        self.PdFa_metric.reset()
        self.ROC_metric.reset()

        # initial inference time
        total_infer_time = 0

        val_loader = self.val_loader

        with torch.no_grad():
            for iter_idx, data in enumerate(tqdm(val_loader)):
                frames, mask, h, w, name = data
                # frames, mask, name = data
                frames, mask = frames.to(self.device), mask.to(self.device)
                if hasattr(self.model.module, 'use_sufficiency_loss') and self.model.module.use_sufficiency_loss:
                    iter_start_time = time.time()
                    preds, _, _ = run_model(self.model, self.model_name, True, False, frames)
                    iter_end_time = time.time()
                elif hasattr(self.model.module, 'use_edge_loss') and self.model.module.use_edge_loss:
                    iter_start_time = time.time()
                    preds, _, = run_model(self.model, self.model_name, False, True, frames)
                    iter_end_time = time.time()
                else:
                    iter_start_time = time.time()
                    preds = run_model(self.model, self.model_name, False, False, frames)
                    iter_end_time = time.time()

                # update inference time
                total_infer_time += (iter_end_time - iter_start_time)

                if not isinstance(preds, (list, tuple)):
                    preds = [preds]
                pred = preds[0]  # Note: distinguish between 0 and -1 when using deep supervision
                # Restore the original image size
                pred = pred[:, :, :h, :w]
                mask = mask[:, :, :h, :w]

                # update metrics
                self.mIoU_metric.update(pred, mask)
                self.nIoU_metric.update(pred, mask)
                self.PdFa_metric.update(pred, mask)
                self.ROC_metric.update(pred, mask)

        # get metrics
        _, mIoU = self.mIoU_metric.get()
        nIoU, _ = self.nIoU_metric.get()
        FA, PD = self.PdFa_metric.get()
        tp_rates, fp_rates, _, _, f_score = self.ROC_metric.get()
        AUC = auc(fp_rates, tp_rates)
        f_score_05 = f_score[(self.bins + 1) // 2]

        # get FPS
        fps = (iter_idx + 1) / total_infer_time

        # print the test info
        message = f'[seq{self.seq}]'
        message += f'[mIoU:{mIoU:.5f}][nIoU:{nIoU:.5f}]'
        message += f'[PD:{PD:.5f}][FA:{FA:.5e}]'
        message += f'[AUC:{AUC:.5f}][f_score:{f_score_05:.5f}]\n'
        print(message)

        try:
            with open(os.path.join(self.cfg['test']['exp_root'], f'metrics_each_seq.txt'), 'a', encoding='utf-8') as f:
                f.write(message)
        except Exception as e:
            print(f"写入文件时出错: {e}")


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    seq_list = [3, 7, 12, 13, 15, 16, 20, 24, 35, 37, 38, 40, 43, 46, 49, 55, 66, 75, 76, 77, 80, 83, 84, 86,
                90, 91, 93, 95, 96, 104, 105, 108, 110, 116, 117, 121, 122, 124, 125, 133, 134, 138]
    print(f'number of test sequences: {len(seq_list)}')
    for seq in seq_list:
        tester = Tester(args, seq)
        tester.test()


def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Deep-MIST Test')
    parser.add_argument('--config', type=str,
                        default='/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/configs/multiframe/MISTNet/test_MISTNet_MIST.yaml',
                        help='path to config file')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device (use cuda or cpu)')
    parser.add_argument('--DataParallel', default=True, help='use DataParallel or not')
    parser.add_argument('--gpu_ids', type=str, default='5', help='the ids of gpus')

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    main(args)
