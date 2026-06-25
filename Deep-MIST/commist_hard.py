#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
from PIL import Image
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt


class MIST0912Analyzer:
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        self.sequence_ids = []
        self.results = {}
        
    def load_dataset_structure(self):
        """加载数据集结构，获取所有序列ID"""
        image_root = os.path.join(self.dataset_root, 'image')
        sequence_folders = glob.glob(os.path.join(image_root, '*'))
        self.sequence_ids = sorted([os.path.basename(f) for f in sequence_folders if os.path.isdir(f)])
        print(f"找到 {len(self.sequence_ids)} 个序列")
        
    def calculate_snr(self, image, mask):
        """计算信噪比(SNR)"""
        # 分离目标和背景区域
        target_region = image[mask > 0]
        background_region = image[mask == 0]
        
        if len(target_region) == 0 or len(background_region) == 0:
            return 0.0
        
        # 计算目标均值和背景标准差
        target_mean = np.mean(target_region)
        background_std = np.std(background_region)
        
        if background_std == 0 or target_mean <= 0:
            return 0.0
        
        # SNR = 20 * log10(目标均值 / 背景标准差)
        snr = 20 * np.log10(target_mean / background_std)
        return snr
    
    def calculate_strict_snr(self, image, mask):
        """严格公式计算点目标信噪比 SNR"""
        target_region = image[mask > 0]
        background_region = image[mask == 0]
        
        if len(target_region) == 0 or len(background_region) == 0:
            return 0.0

        # 背景噪声标准差
        DN_noise = np.std(background_region)

        # 背景均值
        DN_bkg = np.mean(background_region)

        numerator = np.sum(target_region - DN_bkg)
        
        if DN_noise == 0:
            return 0.0
        return numerator / DN_noise
    
    def calculate_psnr(self, image, mask):
        """严格公式计算点目标峰值信噪比 PSNR"""
        target_region = image[mask > 0]
        background_region = image[mask == 0]

        if len(target_region) == 0 or len(background_region) == 0:
            return 0.0

        # 背景噪声标准差
        DN_noise = np.std(background_region)

        # 背景均值
        DN_bkg = np.mean(background_region)

        # 目标最大值（减背景）
        DN_obj_max = np.max(target_region) - DN_bkg

        if DN_noise == 0:
            return 0.0
        return DN_obj_max / DN_noise
    
    def calculate_scr(self, image, mask):
        """严格公式计算信杂比(SCR)，无单位"""
        # 分离目标和背景区域
        target_region = image[mask > 0]
        background_region = image[mask == 0]
        
        if len(target_region) == 0 or len(background_region) == 0:
            return 0.0
        
        # 计算目标均值和背景均值
        target_mean = np.mean(target_region)
        background_mean = np.mean(background_region)
        background_std = np.std(background_region)
        
        if background_std == 0:
            return 0.0
        
        # SCR = |目标均值 - 背景均值| / 背景标准差
        scr = np.abs(target_mean - background_mean) / background_std
        return scr
    
    def calculate_target_size(self, mask):
        """计算目标大小(像素数)"""
        return np.sum(mask > 0)
    
    def calculate_target_intensity(self, image, mask):
        """计算目标区域平均强度"""
        target_region = image[mask > 0]
        if len(target_region) == 0:
            return 0.0
        return np.mean(target_region)
    
    def get_target_centroid(self, mask):
        """计算目标质心"""
        if np.sum(mask) == 0:
            return None
        
        # 使用cv2计算质心
        moments = cv2.moments(mask)
        if moments['m00'] == 0:
            return None
        
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        return np.array([cx, cy])
    
    def calculate_target_velocity(self, centroids):
        """计算目标速度，单位：像素/帧"""
        if len(centroids) < 2:
            return []
        
        velocities = []
        for i in range(1, len(centroids)):
            # 计算相邻两帧质心的欧几里得距离 (像素)
            dist = np.sqrt(np.sum((centroids[i] - centroids[i-1]) ** 2))
            velocities.append(dist)
        
        return velocities
    
    def analyze_sequence(self, sequence_id):
        """分析单个序列的所有指标"""
        print(f"分析序列: {sequence_id}")
        
        # 获取所有帧文件
        image_folder = os.path.join(self.dataset_root, 'image', sequence_id)
        mask_folder = os.path.join(self.dataset_root, 'mask', sequence_id)
        
        image_files = sorted(glob.glob(os.path.join(image_folder, '*.png')))
        mask_files = sorted(glob.glob(os.path.join(mask_folder, '*.png')))
        
        if len(image_files) == 0 or len(mask_files) == 0:
            print(f"警告: 序列 {sequence_id} 缺少数据文件")
            return
        
        # 初始化结果存储
        sequence_results = {
            'snr': [],
            'scr': [],
            'target_sizes': [],
            'target_intensities': [],
            'centroids': [],
            'velocities': [],
            'psnr': []
        }
        
        # 分析每一帧
        for image_file, mask_file in tqdm(zip(image_files, mask_files), total=len(image_files)):
            try:
                # 读取图像和掩码
                image = np.array(Image.open(image_file).convert('L')).astype(np.float32)
                mask = np.array(Image.open(mask_file).convert('L')).astype(np.uint8)
                mask = (mask > 128).astype(np.uint8)  # 转换为二值图像
                
                # 确保image和mask的空间维度匹配
                if image.shape != mask.shape:
                    # 调整mask大小以匹配image
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
                
                # 计算指标
                snr = self.calculate_snr(image, mask)
                scr = self.calculate_scr(image, mask)
                target_size = self.calculate_target_size(mask)
                target_intensity = self.calculate_target_intensity(image, mask)
                centroid = self.get_target_centroid(mask)
                psnr = self.calculate_psnr(image, mask)
                
                # 存储结果
                sequence_results['snr'].append(snr)
                sequence_results['scr'].append(scr)
                sequence_results['target_sizes'].append(target_size)
                sequence_results['target_intensities'].append(target_intensity)
                sequence_results['psnr'].append(psnr)
                if centroid is not None:
                    sequence_results['centroids'].append(centroid)
                    
            except Exception as e:
                print(f"处理文件时出错 {image_file}: {str(e)}")
                continue
        
        # 计算速度
        if len(sequence_results['centroids']) > 1:
            sequence_results['velocities'] = self.calculate_target_velocity(
                sequence_results['centroids'])
        
        # 保存该序列结果
        self.results[sequence_id] = sequence_results
    
    def analyze_all_sequences(self):
        """分析所有序列"""
        for sequence_id in self.sequence_ids:
            self.analyze_sequence(sequence_id)
    
    def generate_sequence_individual_summary(self, output_dir):
        """生成每个序列单独的指标摘要，输出为CSV格式"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 创建CSV格式的汇总报告
        report_path = os.path.join(output_dir, 'MIST_0912_individual_sequences_report.csv')
        
        # 首先创建详细日志来调试强度值问题
        debug_log_path = os.path.join(output_dir, 'debug_intensity_values.log')
        debug_log = open(debug_log_path, 'w', encoding='utf-8')
        debug_log.write("调试目标强度值\n")
        debug_log.write("="*50 + "\n")
        
        with open(report_path, 'w', encoding='utf-8-sig') as f:
            # 写入CSV表头，使用逗号分隔
            headers = ["序列ID", "平均SNR", "平均SCR", "平均PSNR", "平均目标大小", "平均目标速度", "目标强度范围"]
            f.write(",".join(headers) + "\n")
            
            # 按序列ID数字大小排序
            sorted_sequence_ids = sorted(self.results.keys(), key=lambda x: int(x))
            
            # 遍历每个序列，输出单独的指标
            for seq_id in sorted_sequence_ids:
                seq_results = self.results[seq_id]
                
                # 调试：记录原始强度值
                debug_log.write(f"序列 {seq_id} 的原始强度值: {seq_results['target_intensities'][:10]}...\n")
                
                # 计算每个序列的平均指标，过滤无效值
                avg_snr = np.mean([val for val in seq_results['snr'] if not (np.isinf(val) or np.isnan(val))]) if seq_results['snr'] else 0.0
                avg_scr = np.mean([val for val in seq_results['scr'] if not (np.isinf(val) or np.isnan(val))]) if seq_results['scr'] else 0.0
                avg_psnr = np.mean([val for val in seq_results['psnr'] if not (np.isinf(val) or np.isnan(val))]) if seq_results['psnr'] else 0.0
                avg_target_size = np.mean(seq_results['target_sizes']) if seq_results['target_sizes'] else 0.0
                avg_velocity = np.mean(seq_results['velocities']) if seq_results['velocities'] else 0.0
                
                # 计算目标强度范围
                valid_intensities = [val for val in seq_results['target_intensities'] if not (np.isinf(val) or np.isnan(val)) and val > 0]
                
                # 调试：记录有效强度值
                debug_log.write(f"序列 {seq_id} 的有效强度值: {valid_intensities[:10]}...\n")
                debug_log.write(f"有效强度值数量: {len(valid_intensities)}\n")
                
                # 确保有有效强度值再计算范围
                if valid_intensities:
                    min_intensity = np.min(valid_intensities)
                    max_intensity = np.max(valid_intensities)
                    intensity_range = f"{min_intensity:.2f}-{max_intensity:.2f}"
                    debug_log.write(f"序列 {seq_id} 的强度范围: {intensity_range}\n")
                else:
                    intensity_range = "N/A"
                    debug_log.write(f"序列 {seq_id} 没有有效强度值\n")
                
                debug_log.write("-"*50 + "\n")
                
                # 写入CSV行数据，使用逗号分隔
                row_data = [
                    seq_id,
                    f"{avg_snr:.2f}",
                    f"{avg_scr:.2f}",
                    f"{avg_psnr:.2f}",
                    f"{avg_target_size:.2f}",
                    f"{avg_velocity:.2f}",
                    intensity_range
                ]
                f.write(",".join(row_data) + "\n")
        
        debug_log.close()
        print(f"序列单独分析报告(CSV格式)已保存至: {report_path}")
        print(f"调试日志已保存至: {debug_log_path}")
    

if __name__ == '__main__':
    # 设置数据集路径和输出目录
    dataset_root = '/home/ubuntu/data/zhengqinxu/object_detection/DATA/MIST_0912/MIST_0912'
    output_dir = '/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/analysis_results_mist_individual'
    # 创建分析器实例
    analyzer = MIST0912Analyzer(dataset_root)
    
    # 加载数据集结构
    analyzer.load_dataset_structure()
    
    # 分析所有序列
    analyzer.analyze_all_sequences()
    
    # 生成每个序列单独的分析报告
    analyzer.generate_sequence_individual_summary(output_dir)

    print("数据集单独序列分析完成!")