import os
import cv2
import numpy as np
from scipy.spatial.distance import cdist
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MISTTargetTracker:
    def __init__(self, dataset_path, output_dir, dist_thresh=10):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.dist_thresh = dist_thresh
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    def compute_num_targets_in_sequence(self, mask_folder):
        """
        计算一个序列中的目标实例数量（Number of targets per sequence）
        参数：
            mask_folder: 这一序列所有mask文件的路径（按帧排序）
        返回：
            total_tracks: 序列中不同目标的数量
        """
        try:
            # 获取并排序mask文件
            mask_files = sorted([os.path.join(mask_folder, f) for f in os.listdir(mask_folder) 
                                if f.endswith('.png')])
            
            if not mask_files:
                logger.warning(f"在{mask_folder}中未找到mask文件")
                return 0
            
            all_tracks = []  # list of lists, each track = list of centroids (x, y)
            
            for i, mask_path in enumerate(mask_files):
                # 读取并预处理mask
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    logger.warning(f"无法读取mask文件: {mask_path}")
                    continue
                
                mask = (mask > 127).astype(np.uint8)
                
                # 查找连通组件
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
                # 跳过背景（标签0）
                valid_centroids = centroids[1:] if num_labels > 1 else []
                
                if i == 0:
                    # 第一帧，为每个检测到的目标初始化一个轨迹
                    for c in valid_centroids:
                        all_tracks.append([c])
                else:
                    # 将本帧目标与上一帧轨迹进行匹配
                    if len(all_tracks) == 0 or len(valid_centroids) == 0:
                        # 如果上一帧没有轨迹或本帧没有目标，无法匹配
                        if len(valid_centroids) > 0:
                            # 为新出现的目标创建轨迹
                            for c in valid_centroids:
                                all_tracks.append([c])
                        continue
                    
                    # 获取上一帧所有轨迹的最后一个质心
                    prev_centroids = np.array([t[-1] for t in all_tracks])
                    
                    # 计算距离矩阵
                    dist_mat = cdist(prev_centroids, valid_centroids)
                    matched = set()  # 记录本帧中已匹配的目标索引
                    
                    # 匹配过程
                    for t_idx, track in enumerate(all_tracks):
                        if t_idx >= len(dist_mat):
                            continue
                        min_idx = np.argmin(dist_mat[t_idx])
                        if dist_mat[t_idx, min_idx] < self.dist_thresh:
                            # 认为是同一个目标，更新轨迹
                            track.append(valid_centroids[min_idx])
                            matched.add(min_idx)
                    
                    # 对于未匹配的目标，创建新的轨迹
                    for j, c in enumerate(valid_centroids):
                        if j not in matched:
                            all_tracks.append([c])
            
            # 过滤掉长度过短的轨迹（可能是噪声）
            min_track_length = 2  # 轨迹至少需要2帧
            filtered_tracks = [track for track in all_tracks if len(track) >= min_track_length]
            
            total_tracks = len(filtered_tracks)
            logger.debug(f"序列{mask_folder}中的目标实例数量: {total_tracks}")
            return total_tracks
        except Exception as e:
            logger.error(f"处理序列{mask_folder}时出错: {str(e)}")
            return 0
    
    def analyze_dataset(self):
        """
        分析整个数据集，计算每个序列的目标实例数量
        """
        logger.info(f"开始分析数据集: {self.dataset_path}")
        
        # 获取所有序列目录
        sequence_dirs = []
        for root, dirs, files in os.walk(self.dataset_path):
            # 假设mask文件直接存储在序列目录中
            if any(f.endswith('.png') for f in files):
                sequence_dirs.append(root)
        
        if not sequence_dirs:
            logger.warning("未找到包含mask文件的序列目录")
            return
        
        # 按序列ID排序
        sequence_dirs.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
        
        results = []
        
        # 分析每个序列
        for seq_dir in tqdm(sequence_dirs, desc="分析序列"):
            seq_id = os.path.basename(seq_dir)
            num_targets = self.compute_num_targets_in_sequence(seq_dir)
            results.append((seq_id, num_targets))
        
        # 生成结果报告
        self.generate_summary(results)
        
        logger.info(f"数据分析完成，结果已保存到: {self.output_dir}")
    
    def generate_summary(self, results):
        """
        生成结果摘要文件
        """
        output_path = os.path.join(self.output_dir, "MIST_0912_target_instances_summary.txt")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("MIST_0912 数据集目标实例数量统计报告\n")
            f.write("=" * 50 + "\n")
            f.write("序列ID\t目标实例数量\n")
            f.write("=" * 50 + "\n")
            
            total_targets = 0
            for seq_id, num_targets in results:
                f.write(f"{seq_id}\t{num_targets}\n")
                total_targets += num_targets
            
            f.write("=" * 50 + "\n")
            f.write(f"总序列数: {len(results)}\n")
            f.write(f"总目标实例数: {total_targets}\n")
            f.write(f"平均每个序列的目标实例数: {total_targets/len(results):.2f}\n")

if __name__ == "__main__":
    # 配置参数
    DATASET_PATH = "/home/ubuntu/data/zhengqinxu/object_detection/DATA/MIST_0912/MIST_0912"
    OUTPUT_DIR = "/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/analysis_results_mist_individual"
    DIST_THRESH = 10  # 判定相邻帧中同一目标的最大距离（像素）
    
    # 创建分析器并运行
    tracker = MISTTargetTracker(DATASET_PATH, OUTPUT_DIR, DIST_THRESH)
    tracker.analyze_dataset()