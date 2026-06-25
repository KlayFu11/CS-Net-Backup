import torch
import torch.nn as nn
import os
import yaml
from deepmist.models import build_model
from deepmist.utils import ordered_yaml

def export_onnx():
    # ---------------- 配置部分 ----------------
    config_path = '/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/configs/multiframe/LS_STDNet/train_LS_STDNet_MSIT_PANDecoder.yaml'
    checkpoint_path = '/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/results/LS_STDNet55_MIST_0107_PANLSAGGDecoder/20260107_104655/checkpoints/model_epoch_55.pth'
    output_onnx = "DeepMIST_LS_STDNet_epoch55_5D.onnx"
    # ----------------------------------------

    # 1. 加载配置
    with open(config_path, mode='r') as f:
        cfg = yaml.load(f, Loader=ordered_yaml()[0])
    
    # 2. 构建模型
    print("Building model...")
    model, model_name = build_model(cfg['model']['network'])
    
    # 3. 加载权重 (处理 DataParallel 的 module. 前缀)
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model']
    
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()

    # 4. 构造正确的 Dummy Input
    # 依据你的 Dataset 代码：
    # MixData_out shape 是 (3, 5, H, W)
    # DataLoader 会增加 Batch 维度 -> (Batch, 3, 5, H, W)
    # 假设默认输入尺寸为 384x384
    batch_size = 1
    channels = 3   # 重复了3次
    time_steps = 5 # 取了最后5帧
    height = 384
    width = 384
    
    dummy_input = torch.randn(batch_size, channels, time_steps, height, width)
    print(f"Dummy Input Shape: {dummy_input.shape}")

    # 5. 导出 ONNX
    print("Exporting to ONNX...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_onnx,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input_sequence'],
        output_names=['output_mask'],
        # 动态维度设置：
        # 输入是 5 维 (B, C, T, H, W) -> 索引分别是 0, 1, 2, 3, 4
        # 我们允许 Batch(0), Height(3), Width(4) 动态变化
        # Channel(1) 和 Time(2) 通常是固定的
        dynamic_axes={
            'input_sequence': {0: 'batch_size', 3: 'height', 4: 'width'},
            'output_mask': {0: 'batch_size', 2: 'height', 3: 'width'} 
            # 注意：输出通常是 4 维 (B, C, H, W) 或者是 (B, 1, H, W)，所以输出的 H,W 索引可能是 2,3
        }
    )
    print(f"Success! Model saved to: {os.path.abspath(output_onnx)}")

if __name__ == '__main__':
    export_onnx()