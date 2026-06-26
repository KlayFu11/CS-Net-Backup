# Deep-MIST 外部模型接入接口规则

## 1. 文档目的

本文档基于当前 Deep-MIST 仓库中已被 git 跟踪的文件，审计外部模型如何接入现有训练、测试、数据集、loss、metric 和模型加载体系。

后续接入其他论文模型时，原则上只能做模型侧接入和必要 wrapper / adapter，不能重写 Deep-MIST 的实验体系。所有模型必须继续使用当前仓库统一的数据集输出、训练入口、测试入口、loss、metric 和阈值逻辑。

本次审计未接入任何具体新模型，未运行训练或测试。

## 2. 当前 Deep-MIST 实验体系概览

当前训练入口是 `train_loss1.py`，测试入口是 `test.py`。

训练入口执行流程：

1. 通过 `--config` 读取 YAML：`yaml.load(f, Loader=ordered_yaml()[0])`。
2. 用 `build_dataset(self.cfg['dataset'], mode='train')` 构建 train / val / val_hard 数据集。
3. 用 `build_model(self.cfg['model']['network'])` 实例化模型。
4. 用 `build_loss(self.cfg['model']['loss'])` 构建 loss。
5. 每个 batch 从 dataloader 取出 `frames, mask, h, w, name`，通过 `run_model(...)` 统一调用模型。
6. loss 直接接收模型输出 `pred` 或 `preds` 与 `mask`。
7. validation 阶段将 `pred` 和 `mask` 裁剪到原图尺寸 `:h, :w` 后送入 metric。

测试入口执行流程：

1. 通过 `--config` 读取 YAML。
2. 用 `build_dataset(self.cfg['dataset'], mode='val')` 构建 val / val_hard 数据集。
3. 用 `build_model(self.cfg['model']['network'])` 实例化模型。
4. 从 `self.cfg['test']['checkpoint']` 读取 checkpoint，并加载 `checkpoint['model']`。
5. 每个样本通过 `run_model(...)` 得到输出。
6. 使用 `preds[0]` 作为最终 metric 和可视化结果。
7. `pred` 和 `mask` 裁剪到 `:h, :w` 后更新 mIoU、nIoU、PdFa、ROC。

## 3. 当前模型注册与加载机制

模型注册入口是 `deepmist/models/__init__.py`。

当前没有发现自动 registry、字典映射、装饰器注册或动态 import。模型接入依赖三处硬编码：

1. 顶部显式 import 模型 class，例如：
   `from deepmist.models.multiframe.DeepPro.model_DeepPro import DeepPro`
2. `build_model(model_cfg)` 中读取 `model_cfg['name']`，用 `if / elif` 选择 class 并执行 `ModelClass(**model_cfg)`。
3. `run_model(model, model_name, use_sufficiency_loss, use_edge_loss, frames)` 中按 `model_name` 分支处理输入和输出。

新模型真正接入时必须同时检查：

- 是否需要在 `deepmist/models/__init__.py` 顶部添加 import。
- 是否需要在 `build_model()` 添加 `elif model_name == '<NewModel>'`。
- 是否需要在 `run_model()` 添加该模型的输入 / 输出适配分支。
- config 中 `model.network.name` 必须与 `build_model()` 中判断的字符串一致。

单帧模型和多帧模型没有单独 registry，但在文件组织和 `run_model()` 分支中有明确区分：

- 单帧模型位于 `deepmist/models/singleframe/<ModelName>/`。
- 多帧模型位于 `deepmist/models/multiframe/<ModelName>/`。
- `run_model()` 对单帧模型统一执行 `frames = frames[:, :, -1, :, :]`，即取最后一帧作为 `[B, C, H, W]` 输入。
- 多帧模型通常直接接收 `[B, C, T, H, W]`，少数模型有特殊变换，例如 `RFR` 执行 `frames.permute(0, 2, 1, 3, 4)`。

## 4. DeepPro 接入流程审计

DeepPro 的当前 Deep-MIST 接入文件是：

- `deepmist/models/multiframe/DeepPro/model_DeepPro.py`

该文件定义了当前体系直接 import 的 class：

```python
class DeepPro(nn.Module):
    def __init__(self, num_classes, seqlen=100, out_len=100):
```

`deepmist/models/__init__.py` 通过以下方式接入：

```python
from deepmist.models.multiframe.DeepPro.model_DeepPro import DeepPro
...
elif model_name == 'DeepPro':
    model = DeepPro(**model_cfg)
```

DeepPro 是多帧模型。其 `forward(seq_imgs)` 读取输入尺寸：

```python
_, _, _, h, w = seq_imgs.size()
```

结合文件底部独立测试可确认，DeepPro 当前接收：

```python
inputs = torch.randn((1, 3, 5, 384, 384)).cuda()
```

维度含义是 `[B, C, T, H, W]`，即 batch、channel、sequence length、height、width。

DeepPro `forward()` 返回：

```python
return seq_feats, seq_midseg
```

其中 `seq_midseg` 注释为 `b, t, h, w`。Deep-MIST 的 `run_model()` 对 DeepPro 做了最终适配：

```python
seq_feats, preds = model(frames)
preds = preds[:, -1, :, :].unsqueeze(1)
```

因此 DeepPro 在当前统一训练 / 测试体系中的最终预测是最后一个时间步，shape 为 `[B, 1, H, W]`，用于对齐当前 `mask`。

DeepPro 目录中还包含其原始训练、测试、data_utils、networks、losses、README、日志和工具文件。后续接入其他模型时应优先模仿 `model_DeepPro.py` 这种“把网络 class 接入 Deep-MIST 统一入口”的方式，而不是迁移或复用 DeepPro 自带的 `train.py`、`test.py`、`data_utils` 或专用 loss。

后续接入其他模型时，可以参考 DeepPro 的接入流程，但绝对不能修改 DeepPro 既有文件。

## 5. 单帧 / 多帧模型判断规则

后续用户要求接入外部论文模型时，必须先判断该模型属于：

- 单帧模型；
- 多帧模型；
- 不确定，需要进一步审计。

判断依据必须来自代码或资料证据，不能只根据模型名称猜测。至少检查：

- 外部模型 `forward()` 的输入维度。
- 外部仓库最小测试代码或 README 中的 dummy input。
- 论文 / README 是否明确描述 temporal、multi-frame、sequence、video 等信息。
- 模型内部是否使用 3D conv、temporal block、frame alignment、sequence aggregation 等结构。
- 模型是否只处理单张 `[B, C, H, W]` 图像。

接入 Deep-MIST 后的分类标准以当前 pipeline 为准：

- 单帧模型应能接收 `run_model()` 从统一 `frames` 中取出的最后一帧：`frames[:, :, -1, :, :]`，shape 为 `[B, C, H, W]`。
- 多帧模型优先接收 dataloader 原始输出 `frames`，shape 为 `[B, C, T, H, W]`。
- 如果外部模型需要 `[B, T, C, H, W]`，应在模型侧 wrapper 或 `run_model()` 对该模型的分支中做 `permute`，不能修改 dataset。
- 如果无法确定输入维度，必须停止并标记为“需要人工确认”。

## 6. 外部模型独立 forward / FLOPs / Params 测试规则

任何外部模型接入 Deep-MIST 前，都必须先在当前 conda 环境中独立跑通最小 forward / FLOPs / Params 测试。

该测试只用于确认模型能否独立 import、instantiate、forward，并估算参数量和计算量。它不是训练成功，不是正式测试，不是论文指标。

### 多帧模型测试模板

若目标模型确认接收 `[B, C, T, H, W]`，可参考 DeepPro / LS_STDNet 的形式：

```python
if __name__ == '__main__':
    model = NewModel(...).cuda()
    inputs = torch.randn((1, 3, 5, 384, 384)).cuda()
    flops, params = profile(model, (inputs,))
    print('Params = ' + str(round(params / 1000 ** 2, 2)) + 'M')
    print('FLOPs = ' + str(round(flops / 1000 ** 3, 2)) + 'G')
    flops, params = clever_format([flops, params], '%.6f')
    print('Params = ' + params)
    print('FLOPs = ' + flops)
```

注意：

- `(1, 3, 5, 384, 384)` 只能在目标模型真实接口也是 `[B, C, T, H, W]` 且当前 config 使用 5 帧、384 尺寸时使用。
- 若目标模型输入是 `[B, T, C, H, W]`，必须调整 dummy input 和 adapter，不能强行套用。
- 若目标模型需要其他帧数、通道数、尺寸或额外参数，必须按其真实接口测试。
- 若 forward 输出是 tuple / list / dict，必须记录最终用于 Deep-MIST loss / metric 的是哪一个输出。

### 单帧模型测试模板

当前已有单帧模型示例中，FC3Net 和 UIUNet 的独立测试使用：

```python
if __name__ == '__main__':
    model = NewSingleFrameModel(...).cuda()
    inputs = torch.randn((1, 3, 384, 384)).cuda()
    flops, params = profile(model, (inputs,))
    print('Params = ' + str(round(params / 1000 ** 2, 2)) + 'M')
    print('FLOPs = ' + str(round(flops / 1000 ** 3, 2)) + 'G')
    flops, params = clever_format([flops, params], '%.6f')
    print('Params = ' + params)
    print('FLOPs = ' + flops)
```

单帧模型接入 Deep-MIST 后不会直接接收 dataloader 的 `[B, C, T, H, W]`，而是由 `run_model()` 取最后一帧得到 `[B, C, H, W]`。

## 7. 新模型接入 deepmist/models/__init__.py 的规则

真正接入新模型时，按当前仓库风格执行最小改动：

1. 在对应目录新增模型实现文件，例如：
   - 单帧：`deepmist/models/singleframe/<NewModel>/model_<NewModel>.py`
   - 多帧：`deepmist/models/multiframe/<NewModel>/model_<NewModel>.py`
2. 在 `deepmist/models/__init__.py` 顶部添加显式 import。
3. 在 `build_model()` 中添加与 config 一致的 `model_name` 分支。
4. 在 `run_model()` 中添加或复用输入输出适配分支。

当前文件中没有 `__all__`，无需更新 `__all__`。当前也没有 registry / dict，无需创建新的 registry。

不得破坏已有 import。不要为了接入一个模型重构 `build_model()` 或 `run_model()`。

命名规则：

- class 名应与当前模型名清晰对应，例如 `DeepPro`、`LS_STDNet`、`STDQNet`。
- config 中 `model.network.name` 必须与 `build_model()` 的字符串完全一致。
- 目录名、文件名、class 名可以参考已有风格，但最终以 `deepmist/models/__init__.py` 中 import 和 `build_model()` 分支为准。

## 8. config 生成与修改规则

训练和测试代码确认依赖 YAML config，但本次审计中 `configs/` 下的 YAML 文件没有出现在 `git ls-files` 结果中；上层 `.gitignore` 采用先忽略全部再选择性放行的规则，当前 `configs/` 内容未被 git 跟踪。因此，本次不能基于具体 DeepPro YAML 内容写确定模板。

从已跟踪源码可确认 config 至少需要以下字段：

训练入口 `train_loss1.py`：

- `dataset`：传给 `build_dataset()`。
- `model.network`：传给 `build_model()`，其中必须包含 `name`。
- `model.loss`：传给 `build_loss()`，其中必须包含 `num_preds` 和各 loss 子项。
- `model.optimizer`：传给 `set_optimizer()`。
- `model.lr_scheduler`：传给 `set_lr_scheduler()`。
- `train.batch_size`
- `train.num_workers`
- `train.total_epochs`
- `train.resume`，可选
- `train.log_interval`
- `train.save_interval`
- `train.val_interval`
- `train.exp_name`

测试入口 `test.py`：

- `dataset`：传给 `build_dataset()`。
- `model.network`：传给 `build_model()`。
- `test.num_workers`
- `test.checkpoint`
- `test.exp_root`

后续生成新模型 config 时必须遵守：

- 只能修改模型选择相关的必要字段，例如 `model.network.name` 和该模型构造函数真正需要的参数。
- 除非用户明确要求，不得改变 dataset、loss、metric、optimizer、scheduler、epoch、batch size、input size、threshold、train / val / test split 等公平对比相关设置。
- 若复制现有 config，只能复制同类型、同数据集、同训练协议的最小模板。
- 单帧模型优先参考当前单帧配置目录约定；多帧模型优先参考当前多帧配置目录约定。
- 由于具体 YAML 未被 git 跟踪，DeepPro config 的具体字段和值需要人工确认。

## 9. 数据集与输入输出 shape 合约

当前 `build_dataset()` 支持：

- `MIST`
- `SatVideoIRSTD`
- `NUDT-MIRSDT`

`Synthetic` 相关分支在 `deepmist/datasets/__init__.py` 中被注释，不能视为当前训练 / 测试入口默认可选数据集。

### 统一 dataloader 返回值

训练和测试入口都按以下格式解包数据：

```python
frames, mask, h, w, name = data
```

其中：

- `frames`：模型输入序列。
- `mask`：目标帧标签。
- `h, w`：原始或有效图像尺寸，用于测试 / validation 时裁剪预测和 mask。
- `name`：保存可视化结果时使用的路径名。

### 多帧输入格式

MIST、SatVideoIRSTD、Synthetic 都用：

```python
frames = torch.stack(frames, dim=1)
```

单帧图像经 `ToTensor()` 后是 `[C, H, W]`，沿 `dim=1` stack 后样本级 shape 为 `[C, T, H, W]`。经过 DataLoader batch 后是：

```text
[B, C, T, H, W]
```

NUDT-MIRSDT 从 `.mat` 中取最后五帧：

```python
MixData_out = torch.unsqueeze(MixData[-5:, :, :], 0)
MixData_out = MixData_out.repeat(3, 1, 1, 1)
```

样本级 shape 同样是 `[3, 5, H, W]`，batch 后是 `[B, 3, 5, H, W]`。

因此，当前 Deep-MIST 统一多帧输入格式是：

```text
[B, C, T, H, W]
```

常见示例是：

```python
inputs = torch.randn((1, 3, 5, 384, 384)).cuda()
```

维度含义为 batch、channel、time、height、width。

### target frame

MIST 和 SatVideoIRSTD 在构造 frame_list 时使用从 `i - (num_inputs - 1)` 到 `i` 的连续帧，并将 `mask_paths[i]` 作为标签。因此 target frame 是序列最后一帧。

NUDT-MIRSDT 使用 `MixData[-5:, :, :]` 作为输入，并读取同一样本路径对应的 mask；从代码可确认使用最后五帧作为输入，但 `.mat` 样本与 mask 对应的具体时间语义需要结合数据格式人工确认。

Synthetic 文件中注释明确：

```text
输入 [frame[0], frame[1], frame[2], frame[3], frame[4]]，期望输出是 mask[4]
```

### 单帧模型输入

当前 `run_model()` 对单帧模型执行：

```python
frames = frames[:, :, -1, :, :]
```

因此单帧模型接收 target frame，也就是序列最后一帧，shape 为：

```text
[B, C, H, W]
```

### 输出合约

普通 loss 和 metric 期望最终 prediction 与 mask 对齐：

```text
pred: [B, 1, H, W]
mask: [B, 1, H, W]
```

`mIoUMetric` 中有显式断言：

```python
assert pred.shape == mask.shape
```

测试和 validation 会执行：

```python
pred = pred[:, :, :h, :w]
mask = mask[:, :, :h, :w]
```

因此模型输出空间尺寸应至少覆盖有效 `h, w`，最终 metric 前必须能与 mask 对齐。

多输出模型允许返回 list / tuple，但训练和测试默认使用：

- 训练：若 `preds` 不是 list / tuple，则包装成 `[preds]`，然后按 config 中 `num_preds` 和 loss weight 遍历。
- 测试：使用 `preds[0]` 作为最终 metric 和可视化输出。

## 10. wrapper / adapter 使用规则

如果外部模型接口与 Deep-MIST 当前 pipeline 不一致，优先写模型侧 wrapper / adapter。

wrapper 允许做：

- 从 tuple / list / dict 中选择最终预测。
- 将输出 resize 到 label 大小。
- 在模型侧把输入从 `[B, C, T, H, W]` 转成目标模型需要的维度顺序。
- 对单帧模型选择 target frame。
- 将返回值调整为当前 loss / metric 可接受的 `[B, 1, H, W]` 或预测列表。

wrapper 不允许做：

- 额外后处理。
- 修改 threshold。
- 替换 loss。
- 替换 metric。
- 加入外部论文专属训练 trick。
- 改变数据分布。
- 修改 dataset loader。
- 改变公平对比协议。

如果 wrapper 无法解决，必须停止并向用户说明原因，不能擅自修改训练 / 测试 / dataset 主流程。

## 11. 外部论文代码迁移边界

从外部论文仓库接入模型时，必须先找到真正的网络模型文件。

通常只允许迁移：

- 模型主网络 class。
- forward 必需的 block。
- forward 必需的 layer。
- 必要初始化函数。
- 极少量 forward 必需 helper。

禁止迁移：

- 外部仓库 dataset。
- 外部仓库 train。
- 外部仓库 test。
- 外部仓库 metric。
- 外部仓库 loss。
- 外部仓库 logger。
- 外部仓库 config。
- 外部仓库完整工程结构。
- 外部仓库无关 demo / visualization / evaluation 代码。

不要修改外部论文仓库中的原始文件。需要适配时，应在 Deep-MIST 中新增模型文件或 wrapper。

DeepPro 当前目录里保留了原始 `train.py`、`test.py`、`data_utils`、`networks/losses` 等文件，但当前 Deep-MIST 训练 / 测试入口并不应该调用这些外部工程入口。后续新模型接入应更克制，只迁移最小网络代码。

## 12. 已有模型保护规则

绝对不允许修改已经存在的模型文件，包括：

- `deepmist/models/multiframe/DeepPro` 下已有文件。
- 当前 singleframe baseline 模型文件。
- 当前 multiframe baseline 模型文件。
- 与本次新模型无关的任何模型目录。

如果为了接入新模型需要复用已有工具函数，优先 import。若不能 import，必须说明原因，不要复制粘贴大段已有代码。

本次规则文档生成不修改任何已有模型文件。

## 13. 公平对比保护规则

后续外部模型必须使用 Deep-MIST 当前统一训练 / 测试 / 评价体系：

- 使用用户指定的当前训练入口训练，例如 `train_loss1.py`；该入口通过 config 中的 `model.optimizer`、`model.loss`、`model.lr_scheduler` 接入 optimizer、loss 和 scheduler。
- 使用 `test.py` 测试。
- 使用 `deepmist/datasets` 当前数据输出。
- 使用 `deepmist/losses` 当前 loss 构建机制。
- 使用 `deepmist/metrics` 当前 metric。
- 使用 `run_model()` 做模型接口适配。

禁止为了某个外部模型单独改变：

- dataset。
- train / val / test split。
- loss。
- metric。
- optimizer。
- scheduler。
- epoch。
- batch size。
- input size。
- threshold。
- sigmoid / probability 协议。
- 后处理。

当前 metric 细节：

- `mIoUMetric`、`nIoUMetric`、`PdFaMetric` 默认 threshold 为 `0`，直接对 logits 做 `pred > 0`，等价于 `sigmoid(pred) > 0.5`。
- `ROCMetric` 内部先执行 `torch.sigmoid(pred)`，再遍历 threshold。
- 测试可视化也对 `pred` 执行 `torch.sigmoid(pred)` 后保存灰度图。

因此新模型最终输出应是 logits，而不是已经 threshold 后的 binary mask。不要在 wrapper 中随意添加 sigmoid / threshold。

## 14. 最小验证清单

每次真正接入一个新模型后，至少完成以下验证，并如实记录：

```text
Import check: passed / failed / not run
Instantiation check: passed / failed / not run
Standalone forward check: passed / failed / not run
FLOPs / Params check: passed / failed / not run
Deep-MIST model registry check: passed / failed / not run
Config loading check: passed / failed / not run
Train one-batch smoke check: passed / failed / not run
Test smoke check: passed / failed / not run
```

说明：

- smoke test 不是正式实验结果。
- FLOPs / Params 不是检测指标。
- 训练完整指标必须来自正式训练和正式测试。
- 不允许编造任何 IoU、nIoU、Pd、Fa、ROC、AUC、FPS 或 loss 结果。

## 15. 后续接入新模型的标准流程

后续接入任何新模型时，必须按下面顺序实现和验证。不能跳过前置步骤直接改训练或测试入口。

1. 审计外部模型，判断单帧 / 多帧 / 不确定。
2. 找到外部仓库真正的网络主 class 和 forward 必需依赖，只迁移最小网络代码。
3. 先解决模型自身的库包依赖，优先复用当前 conda 环境已有依赖；确需新增依赖时，先说明原因、影响范围和替代方案，不得为了 dataset、logger、config、metric 或训练脚本引入外部工程依赖。
4. 在当前 conda 环境中先让模型文件自身可以 import 和 instantiate。
5. 在模型文件自身的最小测试中跑通 forward / FLOPs / Params。多帧模型可参考以下模板，但 `NewModel(...)` 和 input shape 必须按目标模型真实接口调整：

```python
if __name__ == '__main__':
    model = NewModel(...).cuda()
    inputs = torch.randn((1, 3, 5, 384, 384)).cuda()
    flops, params = profile(model, (inputs,))
    print('Params = ' + str(round(params / 1000 ** 2, 2)) + 'M')
    print('FLOPs = ' + str(round(flops / 1000 ** 3, 2)) + 'G')
    flops, params = clever_format([flops, params], '%.6f')
    print('Params = ' + params)
    print('FLOPs = ' + flops)
```

6. 记录模型自身测试的输入 shape、输出类型、输出 shape，以及 tuple / list / dict 中最终用于训练的 prediction。
7. 在 Deep-MIST 内新增最小模型文件或 wrapper，使模型最终适配当前 pipeline 的输入输出合约。
8. 注册到 `deepmist/models/__init__.py`：添加显式 import，在 `build_model()` 添加模型名分支，在 `run_model()` 添加该模型的输入输出适配分支。
9. 生成或复制对应数据集的最小 config，只改 `model.network.name` 和目标模型构造函数真正需要的参数；optimizer、loss、scheduler、dataset、epoch、batch size、input size 等公平对比字段默认沿用对应数据集的现有配置。
10. 先确认模型能够被 `train_loss1.py` 通过对应 config 加载，并使用 config 中的 `model.optimizer`、`model.loss`、`model.lr_scheduler` 开始训练入口的 smoke check。该阶段只验证训练入口能接上模型，不代表正式训练完成。
11. 训练入口接通后，最后再考虑 `test.py` 接入：确认 checkpoint 加载、`run_model()` 输出、`preds[0]`、metric 和可视化路径都与当前测试合约一致。
12. 汇报所有改动、验证结果、未确认问题，不声称未运行的正式指标。

## 16. DQAligner 接入 train_loss1.py 的复用节点

本节记录一次实际把 DQAligner 接入 `train_loss1.py` 时暴露出的关键节点和取舍。后续接入新模型时，应优先按这些节点检查，不要直接改训练循环。

### 16.1 主训练入口优先走 registry

`train_loss1.py` 已经通过以下入口调用模型和 loss：

```python
from deepmist.losses import build_loss
from deepmist.models import build_model, run_model
```

因此，新增模型或 loss 时优先修改 registry：

- 模型接入 `deepmist/models/__init__.py` 的 `build_model()` 和 `run_model()`。
- loss 接入 `deepmist/losses/__init__.py` 的 `choose_loss()`。
- config 通过 `model.network.name` 和 `model.loss.loss_x.type` 选择模型与 loss。

除非 registry 机制无法覆盖，否则不要在 `train_loss1.py` 中直接 import 新模型或新 loss。这样可以避免所有训练任务启动时都提前加载某个模型的重依赖、CUDA 扩展或外部包。

### 16.2 模型注册不是只加 build_model 分支

多帧模型接入时必须同时处理两层接口：

1. `build_model()`：根据 `model.network.name` 实例化模型。
2. `run_model()`：把该模型真实 `forward()` 输出转换成当前训练循环可消费的预测。

以 DQAligner 为例，外部模型 `forward()` 返回四元组：

```python
return [], output, feat_prop, track_loss
```

原训练代码实际用于 loss 的是第二个输出，并去掉时间维：

```python
outputs = model(frames, None, 0, False)
preds = outputs[1].squeeze(2)
```

接入 Deep-MIST 时，应在 `run_model()` 的 `DQAligner` 分支中完成这个适配，使最终输出保持：

```text
[B, 1, H, W]
```

不能为了适配一个模型修改 `train_loss1.py` 的 batch 解包、loss 计算或 metric 逻辑。

### 16.3 原始模型代码要清理 import 副作用

外部论文代码常见独立工程写法包括：

- `from model.xxx import ...`
- `sys.path.append(...)`
- import 阶段执行 `os.environ['CUDA_VISIBLE_DEVICES'] = '0'`
- 顶层 import `thop`、可视化包或只在 `__main__` 测 FLOPs 时才需要的工具

接入当前仓库时，应改成 Deep-MIST package 路径，并去掉 import 阶段副作用。只在独立 FLOPs / Params 测试中使用的依赖，应下沉到：

```python
if __name__ == '__main__':
    from thop import clever_format, profile
```

如果模型依赖编译扩展，例如 DQAligner 的 `DCN2`，接入代码只修 package import，不应顺手编译、安装或改写扩展。验证阶段如实报告扩展是否可用。

### 16.4 外部 loss 接入要最小复制

如果用户明确要求使用外部论文的训练 loss，优先把该 loss 作为 Deep-MIST loss registry 的一个普通 loss 接入，而不是调用外部训练脚本。

以 DQAligner 的 `AdaFocalLoss` 为例：

- 只复制 `AdaFocalLoss` 本体和必要 import。
- 不复制同一文件中未使用的 `scipy`、`matplotlib`、绘图函数、其他 loss 或调试代码。
- 新文件放在 `deepmist/losses/loss_AdaFocal.py`。
- 在 `deepmist/losses/__init__.py` 中 import 并在 `choose_loss()` 添加 `loss_type == 'AdaFocalLoss'` 分支。
- config 中使用：

```yaml
model:
  loss:
    num_preds: 1
    loss_1:
      type: AdaFocalLoss
      weight: 1
```

如果外部 loss 的 `__init__()` 不接收 `smooth`、`reduction` 等参数，config 中必须移除这些字段，否则 `choose_loss()` 会把多余参数传给 loss 构造函数导致报错。

### 16.5 config 字段必须按当前 dataloader 与接口修正

论文或外部训练脚本的参数不能机械照抄，需要先对齐当前 Deep-MIST 的真实数据接口。

DQAligner 原训练脚本中曾使用 `input_channels=1`，但当前 `MISTDataset` 给 `train_loss1.py` 返回 RGB 多帧输入：

```text
[B, 3, T, H, W]
```

因此接入 MIST 配置时应把 DQAligner 的 `input_channels` 设为 `3`。这属于模型构造参数对当前 pipeline 的必要适配，不是修改 dataset。

如果论文描述“输入帧 zero-padded 到 512 x 512”，而当前 dataset 只暴露 `img_size` resize 接口，没有显式 zero padding 实现，则 config 中只能通过：

```yaml
dataset:
  img_size: [ 512, 512 ]
```

表达尺寸要求。不要为了复现论文一句预处理描述擅自修改 dataset loader。

### 16.6 验证要区分静态通过和运行时通过

接入完成后至少做两层验证：

静态 / 配置验证：

- YAML 能解析。
- `model.network.name` 与 `build_model()` 分支一致。
- `model.loss.loss_x.type` 与 `choose_loss()` 分支一致。
- `run_model()` 分支中最终输出选择规则明确。
- `train_loss1.py` 无非必要 diff。

运行时验证：

- `build_loss()` 能构造新 loss。
- `build_model()` 能构造新模型。
- dummy 或 one-batch forward 输出 `[B, 1, H, W]`。
- loss 能接收 `pred, mask` 并返回标量。

如果当前环境缺少 `torch`、`DCN2` 或 CUDA 扩展，只能报告 runtime check 被阻塞，不能把静态检查通过描述成训练可运行。

### 16.7 本次 DQAligner 的关键取舍

本次接入做过的关键取舍如下，后续遇到类似情况优先复用：

- 选择 registry 接入，不在 `train_loss1.py` 直接 import `DQAligner` 或 `AdaFocalLoss`。
- 选择 lazy import `DQAligner`，避免其他模型训练时也加载 DQAligner 的 DCN 依赖。
- 选择在 `run_model()` 中取 `outputs[1].squeeze(2)`，保持训练循环只接收最终 logits。
- 选择只复制 `AdaFocalLoss`，不迁移外部 loss 文件中的其他无关代码。
- 选择保留用户已经设置的 epoch，不因外部论文参数或历史配置顺手覆盖。
- 选择把外部模型的 `CUDA_VISIBLE_DEVICES` 顶层设置移除，GPU 选择继续由当前训练入口参数控制。
- 选择把 `thop` import 下沉到独立测试入口，避免训练 import 阶段依赖 FLOPs 工具。
- 选择如实记录 runtime 阻塞，例如当前环境缺 `torch` 或模型依赖的 `DCN2` 未编译，而不是伪造 forward 结果。

## 17. 仍需人工确认的问题

以下问题无法仅从本次允许审计的 git 跟踪文件中完全确认：

- `configs/` 下具体 YAML 内容未被 git 跟踪；DeepPro config 的准确字段、loss、optimizer、scheduler、epoch、batch size、input size、checkpoint 路径等需要人工确认。
- DeepPro 原始目录下存在 `train.py`、`test.py`、`data_utils` 和 `networks/losses`，但当前 Deep-MIST 统一训练 / 测试入口不应调用这些文件；它们与当前公平对比配置之间的关系需要用户确认。
- NUDT-MIRSDT 的 `.mat` 样本与 mask 的时间对应关系只能从代码确认“输入取最后五帧并返回对应 mask”，更细的数据语义需要人工确认。
- `test.py` 中使用 `self.model.module` 判断 sufficiency / edge loss，这默认依赖 `DataParallel=True`；若后续关闭 DataParallel，测试分支是否仍能覆盖这些特殊模型需要人工确认。
- `deepmist/models/__init__.py` 中存在若干注释掉或未 import 的模型分支，例如 `RFR`、`MISTNet`、`ISNet`、若干 vis / ablation 模型；后续接入新模型时不应顺手恢复或修改这些历史分支，除非用户明确要求。
