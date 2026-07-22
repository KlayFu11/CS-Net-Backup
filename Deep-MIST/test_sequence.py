import argparse
import os
import random
import time
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from sklearn.metrics import auc
from tqdm import tqdm
from scipy.io import savemat
import json

from deepmist.datasets import build_dataset, DataLoaderX
from deepmist.metrics import mIoUMetric, nIoUMetric, PdFaMetric, ROCMetric
from deepmist.models import build_model, run_model
from deepmist.utils import ordered_yaml, make_dir

# 自定义的数据集类，用于测试特定的序列列表
class CustomMISTDataset:
    def __init__(self, root, seq_list, num_inputs=5, img_size=None, frame_padding=False):
        if img_size is None:
            img_size = [384, 384]
        self.img_size = img_size
        self.num_inputs = num_inputs
        self.grouped_frame_paths = []
        self.grouped_mask_paths = []
        self.sequence_names = []

        # grouping
        for seq in seq_list:
            frame_paths = sorted(glob(os.path.join(root, 'image', str(seq), '*.png')))
            mask_paths = sorted(glob(os.path.join(root, 'mask', str(seq), '*.png')))
            num_frames = len(frame_paths)
            assert num_inputs <= num_frames, f"number of input frames '{num_inputs}' exceeds the total number."
            
            # 存储序列名称
            seq_name = str(seq)
            
            if frame_padding:
                for i in range(num_frames):
                    frame_list = []
                    for j in range(num_inputs - 1, -1, -1):
                        if i - j < 0:
                            frame_list.append(frame_paths[0])
                        else:
                            frame_list.append(frame_paths[i - j])
                    self.grouped_frame_paths.append(frame_list)
                    self.grouped_mask_paths.append(mask_paths[i])
                    self.sequence_names.append(seq_name)
            else:
                for i in range(num_inputs - 1, num_frames):
                    frame_list = []
                    for j in range(num_inputs - 1, -1, -1):
                        frame_list.append(frame_paths[i - j])
                    self.grouped_frame_paths.append(frame_list)
                    self.grouped_mask_paths.append(mask_paths[i])
                    self.sequence_names.append(seq_name)

        # transforms
        self.frame_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.359, 0.359, 0.359], [0.15, 0.15, 0.15])])
        self.mask_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()])

    def __getitem__(self, index):
        frames = []
        for i in range(self.num_inputs):
            frames.append(rgb_loader(self.grouped_frame_paths[index][i]))
        mask = binary_loader(self.grouped_mask_paths[index])

        # transforms
        for i in range(len(frames)):
            frames[i] = self.frame_transform(frames[i])
        frames = torch.stack(frames, dim=1)
        mask = self.mask_transform(mask)

        # get name
        mask_path_split = self.grouped_mask_paths[index].split('/')
        name = mask_path_split[-2] + '/' + mask_path_split[-1]

        # return frames, mask, name, sequence name
        return frames, mask, self.img_size[0], self.img_size[1], name, self.sequence_names[index]

    def __len__(self):
        return len(self.grouped_frame_paths)

# 导入所需的辅助函数
def rgb_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def binary_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

def set_seed(seed, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

from glob import glob
import torchvision.transforms as transforms

class SequenceTester(object):
    def __init__(self, args):
        # seed
        set_seed(args.seed)

        # config
        with open(args.config, mode='r') as f:
            self.cfg = yaml.load(f, Loader=ordered_yaml()[0])

        # device
        self.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

        # 自定义序列列表
        # self.seq_list = [3, 7, 12, 13, 15, 16, 20, 24, 35, 37, 38, 40, 43, 46, 49, 55, 66, 75, 76, 77, 80, 83, 84, 86,
        #                 90, 91, 93, 95, 96, 104, 105, 108, 110, 116, 117, 121, 122, 124, 125, 133, 134, 138]
        self.seq_list = [39]

        # 数据集
        self.dataset = CustomMISTDataset(
            root=self.cfg['dataset']['root'],
            seq_list=self.seq_list,
            num_inputs=self.cfg['dataset']['num_inputs'],
            img_size=self.cfg['dataset']['img_size'],
            frame_padding=self.cfg['dataset']['frame_padding']
        )
        
        self.data_loader = DataLoaderX(self.dataset, batch_size=1,
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
        self.bins = 200  # 10, 200
        self.ROC_metric = ROCMetric(bins=self.bins)

        # checkpoint
        checkpoint_path = args.checkpoint or self.cfg['test']['checkpoint']
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        print(f"Successfully load the checkpoint from {checkpoint_path}.\n")

        # 结果保存目录
        self.results_dir = os.path.join(self.cfg['test']['exp_root'], 'sequence_results')
        make_dir(self.results_dir)
        
        # 可视化目录
        self.pred_vis_dir = os.path.join(self.cfg['test']['exp_root'], 'pred_vis')
        make_dir(self.pred_vis_dir)
        
        # 存储每个序列的指标
        self.sequence_metrics = {}
        for seq in self.seq_list:
            self.sequence_metrics[str(seq)] = {
                'mIoU': 0,
                'nIoU': 0,
                'PD': 0,
                'FA': 0,
                'AUC': 0,
                'precision': 0,
                'recall': 0,
                'f_score': 0,
                'frame_count': 0
            }

    def test(self):
        print("[测试指定序列]\n")
        self.model.eval()

        # 记录每个序列的预测结果
        sequence_preds = {}
        sequence_masks = {}
        for seq in self.seq_list:
            sequence_preds[str(seq)] = []
            sequence_masks[str(seq)] = []

        with torch.no_grad():
            for iter_idx, data in enumerate(tqdm(self.data_loader)):
                frames, mask, h, w, name, seq_name = data
                frames, mask = frames.to(self.device), mask.to(self.device)
                
                iter_start_time = time.time()
                if hasattr(self.model.module, 'use_sufficiency_loss') and self.model.module.use_sufficiency_loss:
                    preds, _, _ = run_model(self.model, self.model_name, True, False, frames)
                elif hasattr(self.model.module, 'use_edge_loss') and self.model.module.use_edge_loss:
                    preds, _, = run_model(self.model, self.model_name, False, True, frames)
                else:
                    preds = run_model(self.model, self.model_name, False, False, frames)
                iter_end_time = time.time()

                if not isinstance(preds, (list, tuple)):
                    preds = [preds]
                pred = preds[0]  # Note: distinguish between 0 and -1 when using deep supervision
                # Restore the original image size
                pred = pred[:, :, :h, :w]
                mask = mask[:, :, :h, :w]

                # 保存预测结果用于后续按序列计算指标
                sequence_preds[seq_name[0]].append(pred.detach().cpu())
                sequence_masks[seq_name[0]].append(mask.detach().cpu())

                # 可视化预测结果
                pred_sigmoid = torch.sigmoid(pred)
                final_pred = pred_sigmoid.data.cpu().numpy()[0, 0, :, :]
                mask_pred = Image.fromarray(np.uint8(final_pred * 255))
                save_dir = os.path.join(self.pred_vis_dir, name[0].split('/')[0])
                make_dir(save_dir)
                mask_pred.save(os.path.join(self.pred_vis_dir, name[0]))

        # 按序列计算指标
        overall_results = {
            'mIoU': 0,
            'nIoU': 0,
            'PD': 0,
            'FA': 0,
            'AUC': 0,
            'precision': 0,
            'recall': 0,
            'f_score': 0
        }
        
        results_file = os.path.join(self.results_dir, 'sequence_results_summary.txt')
        with open(results_file, 'w', encoding='utf-8') as f:
            f.write("序列ID\tmIoU\tnIoU\tPD\tFA\tAUC\tprecision\trecall\tf_score\n")
            
            for seq_id in self.seq_list:
                seq_str = str(seq_id)
                if len(sequence_preds[seq_str]) == 0:
                    continue
                
                # 重置指标
                self.mIoU_metric.reset()
                self.nIoU_metric.reset()
                self.PdFa_metric.reset()
                self.ROC_metric.reset()
                
                # 计算当前序列的指标
                for pred, mask in zip(sequence_preds[seq_str], sequence_masks[seq_str]):
                    pred = pred.to(self.device)
                    mask = mask.to(self.device)
                    self.mIoU_metric.update(pred, mask)
                    self.nIoU_metric.update(pred, mask)
                    self.PdFa_metric.update(pred, mask)
                    self.ROC_metric.update(pred, mask)
                
                # 获取指标值
                _, mIoU = self.mIoU_metric.get()
                nIoU, _ = self.nIoU_metric.get()
                FA, PD = self.PdFa_metric.get()
                tp_rates, fp_rates, recall, precision, f_score = self.ROC_metric.get()
                AUC = auc(fp_rates, tp_rates)
                precision_05 = precision[(self.bins + 1) // 2]
                recall_05 = recall[(self.bins + 1) // 2]
                f_score_05 = f_score[(self.bins + 1) // 2]
                
                # 保存结果
                self.sequence_metrics[seq_str] = {
                    'mIoU': mIoU,
                    'nIoU': nIoU,
                    'PD': PD,
                    'FA': FA,
                    'AUC': AUC,
                    'precision': precision_05,
                    'recall': recall_05,
                    'f_score': f_score_05,
                    'frame_count': len(sequence_preds[seq_str])
                }
                
                # 累加总体结果
                overall_results['mIoU'] += mIoU
                overall_results['nIoU'] += nIoU
                overall_results['PD'] += PD
                overall_results['FA'] += FA
                overall_results['AUC'] += AUC
                overall_results['precision'] += precision_05
                overall_results['recall'] += recall_05
                overall_results['f_score'] += f_score_05
                
                # 写入结果到文件
                f.write(f"{seq_str}\t{mIoU:.5f}\t{nIoU:.5f}\t{PD:.5f}\t{FA:.5e}\t{AUC:.5f}\t{precision_05:.5f}\t{recall_05:.5f}\t{f_score_05:.5f}\n")
                
                # 打印当前序列的结果
                print(f"[序列 {seq_str}]")
                print(f"mIoU: {mIoU:.5f}\tnIoU: {nIoU:.5f}")
                print(f"PD: {PD:.5f}\tFA: {FA:.5e}")
                print(f"AUC: {AUC:.5f}\tprecision: {precision_05:.5f}\trecall: {recall_05:.5f}\tf_score: {f_score_05:.5f}\n")
            
            # 计算并写入总体结果
            num_valid_sequences = len([s for s in self.seq_list if len(sequence_preds[str(s)]) > 0])
            if num_valid_sequences > 0:
                for key in overall_results:
                    overall_results[key] /= num_valid_sequences
                
                f.write("\n[总体结果]\n")
                f.write(f"mIoU: {overall_results['mIoU']:.5f}\tnIoU: {overall_results['nIoU']:.5f}\n")
                f.write(f"PD: {overall_results['PD']:.5f}\tFA: {overall_results['FA']:.5e}\n")
                f.write(f"AUC: {overall_results['AUC']:.5f}\tprecision: {overall_results['precision']:.5f}\trecall: {overall_results['recall']:.5f}\tf_score: {overall_results['f_score']:.5f}\n")
        
        # 保存详细结果为JSON文件
        json_results_file = os.path.join(self.results_dir, 'sequence_results_detail.json')
        with open(json_results_file, 'w', encoding='utf-8') as f:
            json.dump(self.sequence_metrics, f, indent=2, ensure_ascii=False)
        
        print(f"\n测试完成！结果已保存至：{results_file}")
        print(f"详细结果已保存至：{json_results_file}")


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    tester = SequenceTester(args)
    tester.test()


def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Deep-MIST Test')
    parser.add_argument('--config', type=str,
                        default='/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/configs/multiframe/MISTNet/test_MISTNet_MIST.yaml',
                        help='path to config file')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--device', type=str, default='cuda', help='device (use cuda or cpu)')
    parser.add_argument('--DataParallel', default=True, help='use DataParallel or not')
    parser.add_argument('--gpu_ids', type=str, default='3', help='the ids of gpus')
    parser.add_argument('--checkpoint', type=str, 
                        default='/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/results/LS_STDNet_MIST_0408_PANLSAGGDecoder_LKP7SKA3ParaLoss/20260412_000735/checkpoints/model_epoch_83.pth',
                        help='path to model checkpoint')

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    main(args)