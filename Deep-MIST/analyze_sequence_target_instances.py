import os
import numpy as np
import cv2
from PIL import Image
import glob
from tqdm import tqdm

class TargetTracker:
    """单个目标的跟踪器"""
    def __init__(self, centroid, bbox, area, target_id):
        self.centroid = np.array(centroid, dtype=np.float32)
        self.bbox = bbox  # (x, y, w, h)
        self.area = area
        self.id = target_id
        self.velocity = np.zeros(2, dtype=np.float32)
        self.lost_frames = 0
        self.lifetime = 1

    def predict_next_position(self):
        """预测下一帧的位置（用于短时丢失的补偿）"""
        self.centroid = self.centroid + self.velocity

    def update(self, centroid, bbox, area):
        """更新目标状态"""
        new_centroid = np.array(centroid, dtype=np.float32)
        self.velocity = new_centroid - self.centroid
        self.centroid = new_centroid
        self.bbox = bbox
        self.area = area
        self.lost_frames = 0
        self.lifetime += 1


def compute_iou(boxA, boxB):
    """计算IoU"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = boxAArea + boxBArea - interArea
    return interArea / unionArea if unionArea > 0 else 0


class MIST0912TargetInstanceAnalyzer:
    def __init__(self, dataset_root,
                 dist_threshold=15,
                 iou_threshold=0.2,
                 max_lost=3,
                 min_area=3,
                 min_lifetime=3):
        self.dataset_root = dataset_root
        self.sequence_ids = []
        self.results = {}
        self.dist_threshold = dist_threshold
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self.min_area = min_area
        self.min_lifetime = min_lifetime  # 最短轨迹长度过滤

        # 匹配权重
        self.w_dist = 1.0
        self.w_iou = 1.5
        self.w_area = 0.5

    def load_dataset_structure(self):
        mask_root = os.path.join(self.dataset_root, 'mask')
        sequence_folders = glob.glob(os.path.join(mask_root, '*'))
        self.sequence_ids = sorted([os.path.basename(f) for f in sequence_folders if os.path.isdir(f)])
        print(f"找到 {len(self.sequence_ids)} 个序列")

    def extract_targets(self, mask):
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask)
        targets = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area < self.min_area:
                continue
            targets.append((centroids[i], (x, y, w, h), area))
        return targets

    def match_targets(self, trackers, detections):
        """匹配现有轨迹和检测结果"""
        matches = [-1] * len(detections)

        for tracker in trackers:
            best_idx = -1
            best_score = 1e9
            for i, (centroid, bbox, area) in enumerate(detections):
                dist = np.linalg.norm(tracker.centroid - centroid)
                iou = compute_iou(tracker.bbox, bbox)
                area_diff = abs(tracker.area - area) / (tracker.area + area + 1e-6)
                score = self.w_dist * (dist / self.dist_threshold) - self.w_iou * iou + self.w_area * area_diff
                if score < best_score and (dist < self.dist_threshold or iou > self.iou_threshold):
                    best_score = score
                    best_idx = i
            if best_idx != -1:
                tracker.update(*detections[best_idx])
                matches[best_idx] = tracker.id
            else:
                tracker.lost_frames += 1
                tracker.predict_next_position()
        return matches

    def analyze_sequence(self, seq_id):
        print(f"分析序列: {seq_id}")
        mask_folder = os.path.join(self.dataset_root, 'mask', seq_id)
        mask_files = sorted(glob.glob(os.path.join(mask_folder, '*.png')))
        if len(mask_files) == 0:
            return

        trackers = []
        next_id = 0
        all_ids = set()

        for mask_file in tqdm(mask_files, desc=f"Seq {seq_id}", ncols=80):
            mask = np.array(Image.open(mask_file).convert('L'))
            mask = (mask > 128).astype(np.uint8)
            detections = self.extract_targets(mask)
            matches = self.match_targets(trackers, detections)

            # 新增目标
            for i, (centroid, bbox, area) in enumerate(detections):
                if matches[i] == -1:
                    new_tracker = TargetTracker(centroid, bbox, area, next_id)
                    trackers.append(new_tracker)
                    all_ids.add(next_id)
                    next_id += 1

            # 删除长时间丢失的目标
            trackers = [t for t in trackers if t.lost_frames <= self.max_lost]

        # 过滤掉只出现过极短时间的目标
        long_lived_ids = [t.id for t in trackers if t.lifetime >= self.min_lifetime]
        self.results[seq_id] = {'unique_targets': len(long_lived_ids)}

    def analyze_all_sequences(self):
        for seq_id in self.sequence_ids:
            self.analyze_sequence(seq_id)

    def generate_summary(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, 'MIST_0912_unique_targets_refined1.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("MIST_0912 序列独立目标实例数报告（高精度版）\n")
            f.write("=" * 60 + "\n")
            f.write(f"{'序列ID':<10}{'独立目标数':<10}\n")
            f.write("=" * 60 + "\n")
            for seq_id in sorted(self.results.keys(), key=lambda x: int(x)):
                f.write(f"{seq_id:<10}{self.results[seq_id]['unique_targets']:<10}\n")
        print(f"报告已保存至: {report_path}")


if __name__ == '__main__':
    dataset_root = '/home/ubuntu/data/zhengqinxu/object_detection/DATA/MIST_0912/MIST_0912'
    output_dir = '/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/analysis_results_mist_refined'

    analyzer = MIST0912TargetInstanceAnalyzer(dataset_root)
    analyzer.load_dataset_structure()
    analyzer.analyze_all_sequences()
    analyzer.generate_summary(output_dir)
    print("✅ 高精度独立目标实例统计完成！")


