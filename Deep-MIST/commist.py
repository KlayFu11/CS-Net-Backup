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
    
    def generate_summary(self, output_dir):
        """生成分析摘要"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 创建汇总报告
        report_path = os.path.join(output_dir, 'MIST_0912_analysis_report.txt')
        
        # 初始化全局统计数据收集器
        global_stats = {
            'snr': [],
            'scr': [],
            'target_sizes': [],
            'target_intensities': [],
            'velocities': [],
            'psnr': []
        }
        
        # 收集所有序列的数据，并过滤无效值
        for seq_id, seq_results in self.results.items():
            for metric in global_stats.keys():
                if seq_results.get(metric) and len(seq_results[metric]) > 0:
                    # 过滤掉无效值（-inf, inf, nan）
                    valid_values = [val for val in seq_results[metric] if not (np.isinf(val) or np.isnan(val))]
                    global_stats[metric].extend(valid_values)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("MIST_0912 数据集分析报告\n")
            f.write("="*50 + "\n\n")
            
            # 添加数据集级别的统计信息
            f.write("数据集全局统计\n")
            f.write("="*30 + "\n")
            
            # 计算并输出用户要求的所有指标
            if global_stats['snr']:
                f.write(f"平均SNR: {np.mean(global_stats['snr']):.2f} dB\n")
                
            if global_stats['scr']:
                f.write(f"平均SCR: {np.mean(global_stats['scr']):.2f}\n")
                
            if global_stats['psnr']:
                f.write(f"平均PSNR: {np.mean(global_stats['psnr']):.2f}\n")
                
            if global_stats['target_sizes']:
                f.write(f"平均目标大小: {np.mean(global_stats['target_sizes']):.2f} 像素\n")
                
            if global_stats['velocities']:
                f.write(f"平均目标速度: {np.mean(global_stats['velocities']):.2f} 像素/帧\n")
                f.write(f"目标速度范围: {np.min(global_stats['velocities']):.2f} - {np.max(global_stats['velocities']):.2f} 像素/帧\n")
                
            if global_stats['target_intensities']:
                f.write(f"目标强度范围: {np.min(global_stats['target_intensities']):.2f} - {np.max(global_stats['target_intensities']):.2f}\n")
            
            f.write("\n\n")
            
        print(f"分析报告已保存至: {report_path}")
        
        # 生成可视化图表
        self.generate_visualizations(global_stats, output_dir)
        
    def generate_visualizations(self, global_stats, output_dir):
        """生成可视化图表"""
        # 生成目标强度分布图
        if global_stats['target_intensities'] and len(global_stats['target_intensities']) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(global_stats['target_intensities'], bins=50, alpha=0.7, color='blue')
            plt.title('Distribution of Target Intensities')
            plt.xlabel('Intensity Value')
            plt.ylabel('Frequency')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # 添加平均值、最小值和最大值标记
            mean_intensity = np.mean(global_stats['target_intensities'])
            min_intensity = np.min(global_stats['target_intensities'])
            max_intensity = np.max(global_stats['target_intensities'])
            
            plt.axvline(mean_intensity, color='red', linestyle='dashed', linewidth=2, label=f'Average: {mean_intensity:.2f}')
            plt.axvline(min_intensity, color='green', linestyle='dashed', linewidth=1, label=f'Minimum: {min_intensity:.2f}')
            plt.axvline(max_intensity, color='orange', linestyle='dashed', linewidth=1, label=f'Maximum: {max_intensity:.2f}')
            
            plt.legend()
            
            intensity_plot_path = os.path.join(output_dir, 'target_intensity_distribution.png')
            plt.savefig(intensity_plot_path)
            plt.close()
            print(f"目标强度分布图已保存至: {intensity_plot_path}")
            
        # 生成目标速度分布图
        if global_stats['velocities'] and len(global_stats['velocities']) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(global_stats['velocities'], bins=50, alpha=0.7, color='green')
            plt.title('Distribution of Target Velocities')
            plt.xlabel('Velocity (pixels/frame)')
            plt.ylabel('Frequency')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            mean_velocity = np.mean(global_stats['velocities'])
            plt.axvline(mean_velocity, color='red', linestyle='dashed', linewidth=2, label=f'Average speed: {mean_velocity:.2f}')
            plt.legend()
            
            velocity_plot_path = os.path.join(output_dir, 'target_velocity_distribution.png')
            plt.savefig(velocity_plot_path)
            plt.close()
            print(f"目标速度分布图已保存至: {velocity_plot_path}")
            
        print("数据集分析完成!")


if __name__ == '__main__':
    # 设置数据集路径和输出目录
    dataset_root = '/home/ubuntu/data/zhengqinxu/object_detection/DATA/MIST_0912/MIST_0912'
    output_dir = '/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/analysis_results_mist'
    
    # 创建分析器实例
    analyzer = MIST0912Analyzer(dataset_root)
    
    # 加载数据集结构
    analyzer.load_dataset_structure()
    
    # 分析所有序列
    analyzer.analyze_all_sequences()
    
    # 生成分析报告和可视化结果
    analyzer.generate_summary(output_dir)