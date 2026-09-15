# DCPNet 接入 Deep-MIST 训练体系 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 DCPNet 模型接入 Deep-MIST 训练体系（注册 + MIST 配置 + NATTEN 环境），并用 train_loss1.py 启动训练。

**Architecture:** 独立项目移出模型目录后，把自包含的 model_DCPNet.py 放入主仓库模型包；natten 从独立项目自带的 NATTEN-0.14.6 源码编译进 RFR_ADSU 环境；按主仓库惯例在 models/__init__.py 注册 build_model/run_model 分支；训练超参照搬 NUDT-MIRSDT 配置。

**Tech Stack:** PyTorch 1.10.1+cu113 (RFR_ADSU conda env), NATTEN 0.14.6 (源码编译), YAML 配置, L20×2 DataParallel

**约定：**
- 仓库根目录是 `/home/ubuntu/data/zhengqinxu/object_detection`（Deep-MIST 是其子目录），所有 git 命令用 `git -C` 并给出相对仓库根的路径
- 环境 python 直接用 `/home/ubuntu/anaconda3/envs/RFR_ADSU/bin/python`
- 主仓库根（代码运行目录）= `/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST`，简写为 `<ROOT>`

---

### Task 1: 目录重构（移除 gitlink + 移动独立项目）

**Files:**
- 移动: `<ROOT>/deepmist/models/multiframe/DCPNet/` → `<ROOT>/DCPNet_standalone/`
- 索引: 移除 gitlink 条目 `Deep-MIST/deepmist/models/multiframe/DCPNet`

- [ ] **Step 1: 从 git 索引移除已暂存的 gitlink（不动磁盘文件）**

```bash
git -C /home/ubuntu/data/zhengqinxu/object_detection rm --cached Deep-MIST/deepmist/models/multiframe/DCPNet
```
Expected: `rm 'Deep-MIST/deepmist/models/multiframe/DCPNet'`

- [ ] **Step 2: 确认磁盘文件未受影响**

```bash
ls /home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/deepmist/models/multiframe/DCPNet/train.py
```
Expected: 文件存在

- [ ] **Step 3: 移动独立项目到仓库根**

```bash
mv /home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/deepmist/models/multiframe/DCPNet /home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/DCPNet_standalone
```

- [ ] **Step 4: 验证移动结果**

```bash
ls /home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/DCPNet_standalone/
ls /home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/deepmist/models/multiframe/ | grep DCPNet
```
Expected: 第一个 ls 显示 `configs deepmist train.py NATTEN-0.14.6 README.md ...`；第二个 grep 无输出（DCPNet 已移走）

- [ ] **Step 5: 提交**

```bash
git -C /home/ubuntu/data/zhengqinxu/object_detection commit -m "refactor: move standalone DCPNet project out of multiframe model dir"
```
Expected: 提交成功，只有 gitlink 删除这一条变更（DCPNet_standalone 保持 untracked，不加入索引）

---

### Task 2: 创建主仓库 DCPNet 模型包

**Files:**
- Create: `<ROOT>/deepmist/models/multiframe/DCPNet/__init__.py`
- Create: `<ROOT>/deepmist/models/multiframe/DCPNet/model_DCPNet.py`（从独立项目拷贝）

- [ ] **Step 1: 建目录并拷贝模型文件**

```bash
mkdir -p /home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/deepmist/models/multiframe/DCPNet
cp /home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/DCPNet_standalone/deepmist/models/multiframe/DCPNet/model_DCPNet.py /home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/deepmist/models/multiframe/DCPNet/
```

- [ ] **Step 2: 创建空 `__init__.py`**

```bash
touch /home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/deepmist/models/multiframe/DCPNet/__init__.py
```

- [ ] **Step 3: 语法检查**

```bash
/home/ubuntu/anaconda3/envs/RFR_ADSU/bin/python -m py_compile /home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/deepmist/models/multiframe/DCPNet/model_DCPNet.py
```
Expected: 无输出（编译通过）

- [ ] **Step 4: 提交**

```bash
git -C /home/ubuntu/data/zhengqinxu/object_detection add Deep-MIST/deepmist/models/multiframe/DCPNet
git -C /home/ubuntu/data/zhengqinxu/object_detection commit -m "feat: add DCPNet model package to multiframe models"
```
Expected: 提交成功，包含 `__init__.py` 与 `model_DCPNet.py`

---

### Task 3: NATTEN 编译安装（RFR_ADSU 环境）

**Files:**
- 环境: `/home/ubuntu/anaconda3/envs/RFR_ADSU/`（安装 ninja + natten）
- 源码: `<ROOT>/DCPNet_standalone/NATTEN-0.14.6/`

背景：PyPI 网络不可达（已实测 pip download 失败），因此 ninja 从其他 conda 环境拷贝二进制（ninja 为静态可执行文件），natten 用 `--no-build-isolation` 本地编译（env 已有 setuptools 75.1.0、packaging 25.0）。nvcc 11.5 最高支持 sm_86，L20 是 sm_89，故 `TORCH_CUDA_ARCH_LIST="8.6+PTX"` 让驱动 JIT 适配（与 torch 1.10.1+cu113 在 L20 上的现有机制一致）。

- [ ] **Step 1: 拷贝 ninja 二进制进 RFR_ADSU**

```bash
cp /home/ubuntu/anaconda3/envs/Deep-MIST-1/bin/ninja /home/ubuntu/anaconda3/envs/RFR_ADSU/bin/ninja
/home/ubuntu/anaconda3/envs/RFR_ADSU/bin/ninja --version
```
Expected: `1.12.1`

- [ ] **Step 2: 编译安装 NATTEN（后台运行，约 10-20 分钟）**

```bash
cd /home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/DCPNet_standalone/NATTEN-0.14.6 && TORCH_CUDA_ARCH_LIST="8.6+PTX" /home/ubuntu/anaconda3/envs/RFR_ADSU/bin/pip install . --no-build-isolation
```
用 Bash 工具的 run_in_background 执行，输出重定向到 `/tmp/natten_build.log`。等待期间不进行后续任务（Task 4 依赖 natten）。

- [ ] **Step 3: 检查构建结果**

```bash
tail -5 /tmp/natten_build.log
grep -c "error" /tmp/natten_build.log || true
```
Expected: 结尾出现 `Successfully installed natten-0.14.6+torch110cu113`（版本号后缀由 setup.py 按 torch 1.10.1+cu113 生成）；grep error 计数为 0（或仅有无关 false positive，以 Successfully installed 为准）

- [ ] **Step 4: 验证 natten 可导入**

```bash
/home/ubuntu/anaconda3/envs/RFR_ADSU/bin/python -c "import natten; print(natten.__version__)"
```
Expected: `0.14.6+torch110cu113`

（无 git 变更，不提交）

---

### Task 4: 注册模型 + 冒烟测试

**Files:**
- Modify: `<ROOT>/deepmist/models/__init__.py`（import 区 + build_model + run_model）
- Test: 临时脚本 `/tmp/dcpnet_smoke.py`（用后删除）

- [ ] **Step 1: 加 import（在 DeepPro 的 import 行之后）**

文件 `<ROOT>/deepmist/models/__init__.py`，在

```python
from deepmist.models.multiframe.DeepPro.model_DeepPro import DeepPro
```

之后新增一行：

```python
from deepmist.models.multiframe.DCPNet.model_DCPNet import DCPNet
```

- [ ] **Step 2: build_model 加分支**

在

```python
    elif model_name == 'DeepPro':
        model = DeepPro(**model_cfg)
```

之后新增：

```python
    elif model_name == 'DCPNet':
        model = DCPNet(**model_cfg)
```

- [ ] **Step 3: run_model 加分支（并入直通列表）**

把

```python
    elif model_name in ['PSTFNet', 'LVNet', 'MISTNet_wo_MFB_Ls', 'TSINet']:
        preds = model(frames)
```

改为：

```python
    elif model_name in ['PSTFNet', 'LVNet', 'MISTNet_wo_MFB_Ls', 'TSINet', 'DCPNet']:
        preds = model(frames)
```

（DCPNet forward 输入 `[B,C,T,H,W]`，关键帧=最后一帧，返回单个分割 logits，与 MIST 数据集格式一致）

- [ ] **Step 4: 写冒烟测试脚本**

`/tmp/dcpnet_smoke.py`：

```python
import torch
from deepmist.models import build_model, run_model

model, name = build_model({'name': 'DCPNet', 'num_inputs': 5, 'in_channels': 3})
model = model.cuda()
frames = torch.randn(2, 3, 5, 384, 384).cuda()
preds = run_model(model, name, False, False, frames)
assert preds.shape == (2, 1, 384, 384), f'unexpected shape: {preds.shape}'
target = torch.rand(2, 1, 384, 384).cuda()
loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, target)
loss.backward()
print('SMOKE TEST PASSED', preds.shape)
```

- [ ] **Step 5: 运行冒烟测试**

```bash
cd /home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST && /home/ubuntu/anaconda3/envs/RFR_ADSU/bin/python /tmp/dcpnet_smoke.py
```
Expected: `SMOKE TEST PASSED torch.Size([2, 1, 384, 384])`
（该测试同时验证：natten CUDA kernel 在 L20 上可运行、模型注册正确、run_model 直通正确、反向传播正常）

- [ ] **Step 6: 清理临时脚本并提交**

```bash
rm /tmp/dcpnet_smoke.py
git -C /home/ubuntu/data/zhengqinxu/object_detection add Deep-MIST/deepmist/models/__init__.py
git -C /home/ubuntu/data/zhengqinxu/object_detection commit -m "feat: register DCPNet in model registry"
```

---

### Task 5: 编写训练配置 train_DCPNet_MIST.yaml

**Files:**
- Modify: `<ROOT>/configs/multiframe/DCPNet/train_DCPNet_MIST.yaml`（已存在的空文件，整体写入）

- [ ] **Step 1: 写入配置内容**

```yaml
model:
  network:
    name: DCPNet
    num_inputs: 5
    in_channels: 3
  optimizer:
    type: AdamW
    init_lr: !!float 5e-4
    betas: [ 0.9, 0.999 ]
    weight_decay: !!float 1e-4
  lr_scheduler:
    warmup_iters: -1
    scheduler:
      type: MultiStepLR
      step_interval: epoch
      milestones: [ 32, 44 ]
      gamma: 0.1

  loss:
    num_preds: 1
    loss_1:
      type: SoftIoULoss
      weight: 1
      smooth: 1
      reduction: mean

dataset:
  name: MIST
  root: /home/ubuntu/data/zhengqinxu/object_detection/DATA/MIST_0912/MIST_0912
  num_inputs: 5
  img_size: [ 384, 384 ]
  frame_padding: False
  data_aug:
    random_flip: True
    random_crop: True
    random_rotation: True
    color_enhance: False
    random_peper: False

train:
  exp_name: model_DCPNet_MIST
  batch_size: 16
  num_workers: 16
  total_epochs: 50
  log_interval: 50
  save_interval: 1
  val_interval: 1
  resume:
```

（超参 1:1 照搬独立项目 NUDT-MIRSDT 配置：AdamW 5e-4 / wd 1e-4 / MultiStepLR [32,44] gamma 0.1 / SoftIoULoss / batch 16 / 50 epochs；dataset 段与主仓库 TSINet MIST 配置一致）

- [ ] **Step 2: 配置解析 + 组件构建验证**

```bash
cd /home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST && /home/ubuntu/anaconda3/envs/RFR_ADSU/bin/python - <<'EOF'
import yaml, torch
from deepmist.utils.yaml_configs import ordered_yaml
from deepmist.utils.train_and_eval import set_optimizer, set_lr_scheduler
from deepmist.losses import build_loss

cfg = yaml.load(open('configs/multiframe/DCPNet/train_DCPNet_MIST.yaml'), Loader=ordered_yaml()[0])
assert cfg['model']['network']['name'] == 'DCPNet'
opt, lr = set_optimizer([torch.nn.Parameter(torch.zeros(1))], cfg['model']['optimizer'])
print('optimizer:', type(opt).__name__, 'lr:', lr, 'wd:', opt.param_groups[0]['weight_decay'])
sched, interval, warmup = set_lr_scheduler(opt, 50, 1000, cfg['model']['lr_scheduler'])
print('scheduler:', type(sched).__name__, 'interval:', interval, 'milestones:', sched.milestones, 'gamma:', sched.gamma, 'warmup:', warmup)
loss_fn, loss_w, *_ = build_loss(cfg['model']['loss'])
print('loss:', list(loss_fn.keys()), 'weight:', loss_w)
EOF
```
Expected: 输出 `optimizer: AdamW lr: 0.0005 wd: 0.0001`、`scheduler: MultiStepLR interval: epoch milestones: [32, 44] gamma: 0.1 warmup: -1`、`loss: ['SoftIoULoss'] weight: {'SoftIoULoss': [1]}`

- [ ] **Step 3: 提交**

```bash
git -C /home/ubuntu/data/zhengqinxu/object_detection add Deep-MIST/configs/multiframe/DCPNet/train_DCPNet_MIST.yaml
git -C /home/ubuntu/data/zhengqinxu/object_detection commit -m "feat: add DCPNet MIST training config"
```

---

### Task 6: 启动训练并监控

**Files:**
- 运行: `<ROOT>/train_loss1.py`
- 日志: `<ROOT>/results/model_DCPNet_MIST/<时间戳>/train_model_DCPNet_MIST_<时间戳>.log`

- [ ] **Step 1: 后台启动训练（2× L20，DataParallel，batch 16）**

```bash
cd /home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST && nohup /home/ubuntu/anaconda3/envs/RFR_ADSU/bin/python train_loss1.py --config configs/multiframe/DCPNet/train_DCPNet_MIST.yaml --gpu_ids '0,1' > /tmp/dcpnet_train_launch.log 2>&1 &
```
用 Bash 工具的 run_in_background 启动，记录 PID。

- [ ] **Step 2: 等待 2 分钟后检查启动日志**

```bash
tail -20 /tmp/dcpnet_train_launch.log; ls -t /home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/results/model_DCPNet_MIST/*/train_*.log 2>/dev/null | head -1
```
Expected: 启动日志无 traceback；训练日志文件已生成，头部包含 `Version Information` 与配置 dump（含 `PyTorch: 1.10.1+cu113`）

- [ ] **Step 3: 检查 GPU 占用与首个 epoch 迭代**

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
tail -5 "$(ls -t /home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/results/model_DCPNet_MIST/*/train_*.log | head -1)"
```
Expected: GPU 0、1 显存占用 >15GB；训练日志出现 `[epoch:1, iter:50/..., lr:5.000e-04]` 且 `[SoftIoULoss_0:...]` 数值为有限数（非 NaN）

- [ ] **Step 4: 持续监控前几个 epoch（每 10-20 分钟看一次）**

```bash
tail -3 "$(ls -t /home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/results/model_DCPNet_MIST/*/train_*.log | head -1)"
```
Expected: epoch 递增，`SoftIoULoss_0` 逐 epoch 总体下降；epoch ≥ 20 起每个 epoch 出现 `[testset]` 段并输出 mIoU/nIoU/PD/FA 指标。若出现 NaN 或 loss 发散，立即用 TaskStop 停止训练并排查（优先检查学习率与数据）。

- [ ] **Step 5: 确认训练稳定后汇报状态**

向用户汇报：日志路径、当前 epoch、loss 趋势、预计总时长。训练为 50 epochs，持续数小时以上，进程用 nohup 启动不随会话退出而终止。

---

## 自检记录

- Spec 覆盖：设计文档 5 个章节（目录重构/NATTEN/注册/配置/冒烟+启动）分别对应 Task 1/3/4/5/6 ✓
- 无占位符：所有步骤含具体命令、代码与期望输出 ✓
- 一致性：`DCPNet` 模型名、`num_inputs: 5`、`in_channels: 3`、`run_model(model, name, False, False, frames)` 签名在各任务间一致 ✓
