import os
import numpy as np
import cv2
from PIL import Image
import glob
from tqdm import tqdm


class MIST0912TargetBackgroundAnalyzer:
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        self.sequence_ids = []
        self.results = {}
        # 创建膨胀核（3x3的矩形核）
        self.dilate_kernel = np.ones((3, 3), np.uint8)  # 用于膨胀一个像素
        
    def load_dataset_structure(self):
        """加载数据集结构，获取所有序列ID"""
        image_root = os.path.join(self.dataset_root, 'image')
        sequence_folders = glob.glob(os.path.join(image_root, '*'))
        self.sequence_ids = sorted([os.path.basename(f) for f in sequence_folders if os.path.isdir(f)])
        print(f"找到 {len(self.sequence_ids)} 个序列")
        
    def extract_independent_targets(self, mask):
        """使用连通组件分析提取掩码中的独立目标实例"""
        # 确保掩码是8位单通道图像
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        
        # 使用连通组件分析
        num_labels, labels_im = cv2.connectedComponents(mask)
        
        # 提取每个独立目标的掩码
        target_masks = []
        for label in range(1, num_labels):  # 跳过背景(label=0)
            target_mask = (labels_im == label).astype(np.uint8)
            target_masks.append(target_mask)
        
        return target_masks
        
    def dilate_target(self, target_mask):
        """将目标向四周膨胀一个像素"""
        # 使用3x3的核进行膨胀，相当于向四周扩展一个像素
        dilated_mask = cv2.dilate(target_mask, self.dilate_kernel, iterations=1)
        
        # 计算膨胀区域（只包含膨胀出来的部分，不包含原始目标）
        # 膨胀区域 = 膨胀后的掩码 - 原始目标掩码
        background_ring = cv2.subtract(dilated_mask, target_mask)
        
        return dilated_mask, background_ring
        
    def calculate_region_intensity(self, image, region_mask):
        """计算指定区域的平均强度"""
        region_pixels = image[region_mask > 0]
        if len(region_pixels) == 0:
            return 0.0
        return np.mean(region_pixels)
        
    def analyze_sequence(self, sequence_id):
        """分析单个序列中目标强度与膨胀背景强度的差值范围"""
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
            'target_background_diffs': [],  # 存储每个目标的强度差值
            'total_independent_targets': 0,  # 整个序列中的独立目标实例数
            'max_diff': 0.0  # 新增：存储该序列中的最大差值
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
                
                # 提取独立目标实例
                target_masks = self.extract_independent_targets(mask)
                sequence_results['total_independent_targets'] += len(target_masks)
                
                # 处理每个独立目标
                for target_mask in target_masks:
                    # 计算目标区域强度
                    target_intensity = self.calculate_region_intensity(image, target_mask)
                    
                    # 膨胀目标并获取膨胀背景环
                    _, background_ring = self.dilate_target(target_mask)
                    
                    # 计算膨胀背景环的强度
                    background_intensity = self.calculate_region_intensity(image, background_ring)
                    
                    # 计算目标强度与背景强度的差值
                    diff = target_intensity - background_intensity
                    sequence_results['target_background_diffs'].append(diff)
                    
                    # 更新最大差值
                    if diff > sequence_results['max_diff']:
                        sequence_results['max_diff'] = diff
                        
            except Exception as e:
                print(f"处理文件时出错 {image_file}: {str(e)}")
                continue
        
        # 保存该序列结果
        self.results[sequence_id] = sequence_results
    
    def analyze_all_sequences(self):
        """分析所有序列"""
        for sequence_id in self.sequence_ids:
            self.analyze_sequence(sequence_id)
    
    def generate_target_background_diff_summary(self, output_dir):
        """生成每个序列的目标强度与膨胀背景强度差值范围摘要，增加最大差值输出"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 创建汇总报告
        report_path = os.path.join(output_dir, 'MIST_0912_target_background_diff_summary.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("MIST_0912 数据集目标强度与膨胀背景强度差值范围统计报告\n")
            f.write("="*110 + "\n")
            # 增加最大差值列
            f.write(f"{'序列ID':<10}{'独立目标实例数':<18}{'差值最小值':<15}{'差值最大值':<15}{'最大差值':<15}{'差值范围':<20}\n")
            f.write("="*110 + "\n")
            
            # 按序列ID数字大小排序
            sorted_sequence_ids = sorted(self.results.keys(), key=lambda x: int(x))
            
            # 遍历每个序列，输出差值范围统计信息
            for seq_id in sorted_sequence_ids:
                seq_results = self.results[seq_id]
                
                # 获取独立目标实例总数
                total_independent_targets = seq_results['total_independent_targets']
                
                # 获取差值数据
                diffs = seq_results['target_background_diffs']
                
                # 获取预先计算的最大差值
                max_diff = seq_results.get('max_diff', 0.0)
                
                # 计算差值范围
                if diffs:
                    min_diff = np.min(diffs)
                    # 虽然之前已经计算了max_diff，但为了确保数据一致性，这里再次计算
                    max_diff_data = np.max(diffs)
                    diff_range = f"{min_diff:.2f}-{max_diff_data:.2f}"
                else:
                    min_diff = 0.0
                    max_diff_data = 0.0
                    diff_range = "N/A"
                
                # 写入每个序列的结果，每行一个序列，增加最大差值列
                f.write(f"{seq_id:<10}{total_independent_targets:<18}{min_diff:<15.2f}{max_diff_data:<15.2f}{max_diff:<15.2f}{diff_range:<20}\n")
        
        print(f"目标强度与膨胀背景强度差值范围统计报告已保存至: {report_path}")


if __name__ == '__main__':
    # 设置数据集路径和输出目录
    dataset_root = '/home/ubuntu/data/zhengqinxu/object_detection/DATA/MIST_0912/MIST_0912'
    output_dir = '/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/analysis_results_mist_individual'
    
    # 创建分析器实例
    analyzer = MIST0912TargetBackgroundAnalyzer(dataset_root)
    
    # 加载数据集结构
    analyzer.load_dataset_structure()
    
    # 分析所有序列
    analyzer.analyze_all_sequences()
    
    # 生成差值范围统计报告
    analyzer.generate_target_background_diff_summary(output_dir)
    
    print("数据集目标强度与膨胀背景强度差值范围统计分析完成!")