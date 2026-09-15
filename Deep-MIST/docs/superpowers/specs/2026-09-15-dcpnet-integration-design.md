# DCPNet 接入 Deep-MIST 训练体系 — 设计文档

日期：2026-09-15
状态：已确认

## 背景

`deepmist/models/multiframe/DCPNet/` 目前装的是 DCPNet 独立项目（自带嵌套 `deepmist/`、`configs/`、`train.py`、`NATTEN-0.14.6/`、`.git`）。目标是把 DCPNet 模型接入主仓库训练体系：

1. 注册到 `deepmist/models/__init__.py`（`build_model` + `run_model`）
2. 编写 `configs/multiframe/DCPNet/train_DCPNet_MIST.yaml`（MIST 数据集）
3. 训练超参参考独立项目 NUDT-MIRSDT 配置
4. 通过 `train_loss1.py` 在 RFR_ADSU conda 环境训练

## 关键事实（已核实）

- `model_DCPNet.py` 完全自包含：仅依赖 torch、`natten.functional`、thop（无其他本地模块依赖）
- DCPNet forward 输入 `[B, C, T, H, W]`（关键帧 = 最后一帧），输出单个分割 logits；默认 `num_inputs=5`
- 主仓库 MISTDataset 返回 `frames` 形状 `[B, C, 5, H, W]`（torch.stack dim=1）— 与 DCPNet 匹配
- 主仓库兼容项已核实：`SoftIoULoss(smooth, reduction)`、`AdamW` + `weight_decay`、`MultiStepLR` + `step_interval: epoch` + `milestones` + `gamma` 全部支持
- 环境 RFR_ADSU：Python 3.8.20，torch 1.10.1+cu113（L20 上 CUDA 可用），thop 已装，**natten 未安装**
- 硬件：6× NVIDIA L20（sm_89）空闲；系统 nvcc 11.5（最高支持 sm_86）
- MIST 数据路径已验证存在：`/home/ubuntu/data/zhengqinxu/object_detection/DATA/MIST_0912/MIST_0912`
- `configs/multiframe/DCPNet/train_DCPNet_MIST.yaml` 已存在但为空

## 设计

### 1. 目录重构

- 将独立项目整体移到仓库根目录 `Deep-MIST/DCPNet_standalone/`（保留作参考，不纳入主仓库跟踪；嵌套 `.git` 原样保留）
- 从主仓库 index 移除已暂存的 gitlink 条目（`git rm --cached`，不动磁盘文件）
- 新建模型包 `deepmist/models/multiframe/DCPNet/`：
  - `__init__.py`（空）
  - `model_DCPNet.py`（从独立项目拷贝）

### 2. NATTEN 编译安装（RFR_ADSU）

- `pip install ninja`（构建工具，env 中缺失）
- 从 `DCPNet_standalone/NATTEN-0.14.6` 编译安装：`TORCH_CUDA_ARCH_LIST="8.6+PTX" pip install .`
- 原因：nvcc 11.5 不支持 sm_89（L20），8.6+PTX 生成的 PTX 由驱动 JIT 适配 sm_89（与现有 torch 1.10.1+cu113 在 L20 上运行的机制一致）
- 验证：`import natten` 成功

### 3. 模型注册（`deepmist/models/__init__.py`）

- import：`from deepmist.models.multiframe.DCPNet.model_DCPNet import DCPNet`
- `build_model` 加分支：`elif model_name == 'DCPNet': model = DCPNet(**model_cfg)`
  - config 的 network 段只传 `num_inputs: 5`、`in_channels: 3`，其余用构造器默认值
- `run_model` 加分支：`elif model_name in ['DCPNet']: preds = model(frames)`
  - frames `[B,C,T,H,W]` 直通；bim=None（不启用 BiM gating，与独立项目训练一致）

### 4. 训练配置 `train_DCPNet_MIST.yaml`

超参照搬独立项目 NUDT-MIRSDT 配置：

- network: `name: DCPNet, num_inputs: 5, in_channels: 3`
- optimizer: AdamW, lr 5e-4, betas [0.9, 0.999], weight_decay 1e-4
- lr_scheduler: MultiStepLR, step_interval: epoch, milestones [32, 44], gamma 0.1, warmup_iters -1
- loss: num_preds 1, SoftIoULoss, weight 1, smooth 1, reduction mean
- dataset: MIST, root `/home/ubuntu/data/zhengqinxu/object_detection/DATA/MIST_0912/MIST_0912`, num_inputs 5, img_size [384, 384], frame_padding False, 增强 random_flip/crop/rotation True
- train: exp_name `model_DCPNet_MIST`, batch_size 16, num_workers 16, total_epochs 50, log/save/val interval 1

### 5. 冒烟测试 + 启动训练

- 冒烟测试：随机输入 `[2, 3, 5, 384, 384]` 前向 + 反向，验证 natten kernel 与输出形状 `[2, 1, 384, 384]`
- 启动：`python train_loss1.py --config configs/multiframe/DCPNet/train_DCPNet_MIST.yaml --gpu_ids '0,1'`
  - 2 卡 DataParallel，batch 16（每卡 8）
  - 后台运行，监控前几个 epoch loss 正常下降、无 NaN

## 验证标准

- `import natten` 成功
- 冒烟测试前向/反向通过
- 训练启动无报错，前几个 epoch SoftIoULoss 正常下降
- epoch ≥ 20 起 val 正常输出 mIoU/nIoU/Pd/Fa
