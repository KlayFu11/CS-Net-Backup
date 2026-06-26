# AGENTS.md

本仓库是红外小目标检测 / 分割论文实验代码仓库，核心用途是完成不同网络模型在同一实验体系下的公平对比。

Codex 在本仓库中的主要任务是：

> 将其他论文提出的网络模型接入当前已有实验体系，使其可以在相同数据集、相同训练流程、相同测试流程、相同评价指标下运行，并与本仓库已有模型进行公平对比。

当前仓库已经具备训练代码、测试代码、数据集加载、模型加载、损失函数和评价指标。
Codex 不应重写实验体系，只应完成模型接入所需的最小改动。

---

## 1. 总体原则

本仓库的目标是论文实验对比，不是通用深度学习框架开发。

所有修改必须服务于一个明确目标：

> 让某个外部论文模型能够在当前仓库既有训练 / 测试 / 评价体系中运行。

除非用户明确要求，否则不得修改：

* 数据集加载逻辑；
* 训练循环；
* 测试循环；
* 损失函数；
* 评价指标；
* baseline 模型；
* 默认训练超参数；
* 输入尺寸；
* batch size；
* learning rate；
* optimizer；
* scheduler；
* epoch 数；
* 数据增强；
* 阈值策略；
* checkpoint 保存和加载逻辑。

默认规则是：

> 适配外部模型到本仓库现有实验协议，而不是为了外部模型改造本仓库实验协议。
> 项目严格采用git进行版本管理，除非特别指定，禁止你修改git没有跟踪的文件。
> 当我要求开始实施把模型接入训练测试体系的时候，必须参考接口规则文档Port.md。

---

## 2. 沟通风格

Codex 在分析和实现时必须保持客观、严谨、审慎。

要求：

* 不要为了迎合用户而过快同意。
* 不要夸大实现效果。
* 不要把猜测说成事实。
* 不要隐藏不确定性。
* 不要在没有证据的情况下断言某个模型结构、指标或论文结论。
* 如果存在多种解释，必须说明差异和取舍。
* 如果发现用户任务中存在实验公平性问题，必须直接指出。

可以说：

```text
当前外部模型的输入格式与本仓库现有 pipeline 不一致，如果直接修改 dataset loader，会破坏公平对比。更合理的做法是优先写一个模型侧 adapter。
```

不应说：

```text
这个模型我已经完全接好了，应该可以跑。
```

除非已经完成必要验证。

---

## 3. 外部证据与资料使用

当任务涉及其他论文、外部代码仓库或相关工作实现时，不要只依赖内部知识。

需要时应主动查看：

* 目标论文代码仓库；
* 目标模型的 `README`；
* 目标模型的核心网络定义；
* 目标模型的输入输出格式；
* 目标模型是否依赖特殊预处理、特殊标签格式或特殊 loss；
* 本仓库当前训练 / 测试 / 模型注册方式。

如果无法访问外部资料，必须明确说明：

```text
我无法确认原仓库实现细节，因此下面只基于当前可见代码做最小接入，不能声称完全复现原论文实现。
```

不得编造论文实现细节、实验指标或训练结果。

---

## 4. 编码前必须先审计

在写任何实质性代码之前，必须先进行最小审计，并输出简短计划。

至少确认：

1. 当前已有模型放在哪里。
2. 当前模型如何被导入、注册或选择。
3. 训练脚本如何选择模型。
4. 测试脚本如何选择模型。
5. 当前 pipeline 传入模型的输入 tensor shape。
6. 当前 loss 和 metric 期望的输出 tensor shape。
7. 外部模型的 `forward()` 是否与当前 pipeline 兼容。
8. 是否需要 wrapper / adapter。

不要猜测这些信息。

如果不清楚，先停止并说明不清楚的位置。

---

## 5. 最小实现计划格式

每次接入新模型时，Codex 应先输出类似计划：

```text
假设：
- 本次任务是将 <MODEL_NAME> 接入当前实验体系，用于公平对比。
- 数据集、训练循环、测试循环、loss、metrics 默认不修改。

计划：
1. 审计当前模型接口和模型选择机制。
   验证：找到模型定义位置、注册位置、输入输出格式。

2. 审计外部模型代码。
   验证：确认必要网络文件、构造参数、forward 返回值。

3. 添加最小模型文件或 wrapper。
   验证：模型可以 import 和 instantiate。

4. 接入当前模型选择机制。
   验证：训练 / 测试脚本可以通过已有方式选择该模型。

5. 运行最小 smoke test。
   验证：dummy forward 或最小训练 / 测试命令可以运行。

6. 汇报改动文件、验证结果和未解决问题。
```

计划必须短，不要写成大段设计文档。

---

## 6. 最小可运行切片优先

优先完成最小可运行切片，而不是一次性做完整大改。

正确顺序：

1. 让模型文件能被 import。
2. 让模型能 instantiate。
3. 让 dummy input 能 forward。
4. 让 output shape 对齐当前 loss / metric。
5. 让训练脚本能选择该模型。
6. 让测试脚本能选择该模型。
7. 再考虑是否需要补充说明文档。

不要一上来做：

* 大规模重构；
* 新增完整 model zoo；
* 新增复杂配置系统；
* 新增通用 registry 框架；
* 新增实验管理系统；
* 重写训练入口；
* 重写测试入口。

---

## 7. 局部修改原则

只修改完成任务必须修改的文件。

通常允许：

* 新增一个模型定义文件；
* 新增少量模型辅助层；
* 新增一个 wrapper / adapter；
* 在已有模型选择机制中添加一个模型名；
* 在必要位置添加 import；
* 添加最小 smoke test，前提是仓库已有类似测试位置或用户要求。

通常禁止：

* 重构训练代码；
* 重构测试代码；
* 重构数据集代码；
* 重构指标代码；
* 修改 baseline 模型；
* 修改已有模型行为；
* 重命名已有文件；
* 删除无关代码；
* 格式化无关文件；
* 清理与本任务无关的历史问题。

判断标准：

> 每一行改动都必须能解释为“为了让这个外部模型接入当前实验体系”。

如果发现无关 bug、死代码或风格问题，只在最终报告中说明，不主动修改。

---

## 8. 公平对比规则

论文实验最重要的是公平对比。

除非用户明确要求复现外部论文的原始训练设置，否则默认所有模型使用本仓库当前设置。

不得为某个对比模型单独改变：

* loss；
* optimizer；
* learning rate；
* scheduler；
* epoch；
* augmentation；
* input size；
* normalization；
* threshold；
* post-processing；
* metric；
* train / val / test split。

如果外部模型原论文依赖特殊设置，应报告：

```text
该模型原论文使用了不同的训练协议。为了保证与本仓库模型公平对比，本次接入只保留网络结构，不迁移其训练策略。
```

如果某个模型不修改实验协议就无法运行，必须先说明原因，再提出最小修改方案。

---

## 9. 模型接口约束

新增模型必须适配本仓库已有模型接口。

在实现前必须确认：

* 输入是单帧还是多帧；
* 输入 tensor 是 `[B, C, H, W]` 还是 `[B, T, C, H, W]`；
* 多帧是否通过 channel concat；
* 输出是 logits 还是 probability；
* 输出是 tensor、tuple、list 还是 dict；
* 当前 loss 是否接受多输出；
* 当前 metric 是否接受多输出；
* 输出是否需要与 label 保持相同空间尺寸。

如果外部模型返回多个输出，而当前 pipeline 只接受一个 tensor，优先在模型侧或 wrapper 中返回最终预测结果。

优先做：

```python
class ExternalModelWrapper(nn.Module):
    def __init__(self, ...):
        super().__init__()
        self.model = ExternalModel(...)

    def forward(self, x):
        out = self.model(x)
        return normalize_to_existing_output_contract(out)
```

不优先做：

```text
为了适配一个模型，修改整个 train loop，使其支持各种复杂返回格式。
```

只有 wrapper 无法解决时，才考虑修改训练 / 测试逻辑。

---

## 10. Shape 处理规则

涉及 tensor shape 时必须谨慎。

不要随意添加：

* `sigmoid`；
* `softmax`；
* `squeeze`；
* `unsqueeze`；
* `permute`；
* `view`；
* `reshape`；
* `interpolate`。

只有在确认当前 loss / metric / label 格式需要时才允许。

如果必须 resize 输出，应优先遵循仓库已有实现风格，例如已有模型如何对齐输出尺寸，就按已有方式处理。

如果外部模型是单帧模型，而本仓库是多帧输入，不要发明复杂时序融合策略。

可接受的最低限度策略包括：

* 使用当前 pipeline 中定义的 target frame；
* 使用已有代码中已经采用的帧选择方式；
* 写一个明确的 adapter。

如果不清楚 target frame 是哪一帧，必须先审计代码，不得猜。

---

## 11. 外部模型代码接入规则

从其他论文代码仓库迁移模型时，只迁移必要网络代码。

允许迁移：

* 核心网络结构；
* 必要 block；
* 必要 layer；
* 必要初始化函数；
* 极少量与 forward 强相关的工具函数。

禁止迁移：

* 外部仓库的数据集加载；
* 外部仓库的训练脚本；
* 外部仓库的测试脚本；
* 外部仓库的评价指标；
* 外部仓库的日志系统；
* 外部仓库的配置系统；
* 外部仓库的完整工程结构。

必要时可在模型文件顶部添加简短来源说明：

```python
"""
Model implementation adapted from <paper/repository name>.
Only the network definition is integrated.
Training, testing, dataset loading, loss, and metrics are provided by this repository.
"""
```

不要在代码中写长篇论文介绍。

---

## 12. 依赖管理

不要随意添加新依赖。

添加新依赖前必须判断：

1. 当前仓库是否已经有等价依赖。
2. 是否可以用 PyTorch 原生实现。
3. 是否只是为了少量 layer 引入大型库。
4. 是否会影响实验环境稳定性。

通常不应为了以下目的添加依赖：

* config；
* logging；
* visualization；
* metric；
* dataset；
* training helper；
* model summary；
* FLOPs 统计。

如果外部模型只需要几个自定义层，优先直接迁移必要层，而不是引入整个外部项目。

---

## 13. 结果与指标诚信

绝对禁止伪造实验结果。

不得编造：

* IoU；
* nIoU；
* Pd；
* Fa；
* ROC；
* FPS；
* FLOPs；
* Params；
* train loss；
* test result；
* checkpoint 性能；
* 消融实验结果。

如果没有运行训练，就说没有运行训练。

如果只运行了 smoke test，就说只运行了 smoke test。

如果测试脚本报错，报告真实错误。

不得把“代码能 forward”描述成“实验完成”。

---

## 14. 验证要求

实现后至少进行最小验证。

优先验证：

1. Import check
   新模型文件可以正常 import。

2. Instantiation check
   新模型可以通过当前仓库已有模型选择机制实例化。

3. Forward check
   使用与当前训练输入一致的 dummy tensor 完成 forward。

4. Output check
   输出 shape 与当前 label / loss / metric 期望一致。

5. Pipeline smoke check
   如果可行，运行最小训练或测试命令。

不能运行的检查必须说明原因。

最终报告中必须如实写：

```text
Import check: passed / failed / not run
Instantiation check: passed / failed / not run
Forward check: passed / failed / not run
Train/test smoke check: passed / failed / not run
```

---

## 15. 子代理使用规则

只有任务明显可拆分且并行探索能提高质量时，才使用 subagents。

适合使用 subagents 的情况：

* 一个 agent 审计本仓库模型接口；
* 一个 agent 审计外部论文模型代码；
* 一个 agent 检查训练 / 测试入口；
* 一个 agent 做最小 smoke test。

不适合使用 subagents 的情况：

* 小范围代码修改；
* 简单 import 修复；
* 单文件 wrapper 编写；
* casual brainstorming；
* 没有明确边界的探索。

即使使用 subagents，也必须保持少量、明确、可汇总。

不要因为任务复杂就盲目创建大量 subagents。

---

## 16. 什么时候必须询问用户

遇到以下情况时，不要擅自大改，必须先向用户说明问题并请求决策：

* 外部模型要求改变数据集格式；
* 外部模型要求改变标签格式；
* 外部模型要求改变 loss；
* 外部模型要求改变 metric；
* 外部模型要求不同输入帧数；
* 外部模型要求不同输入尺寸；
* 外部模型需要特殊预处理；
* 外部模型需要特殊后处理；
* 当前仓库存在多个可能模型接口，无法判断该接哪一个；
* 外部模型代码不完整；
* 外部模型 license / 权重 / 依赖导致无法直接运行；
* 要完成任务必须修改训练 / 测试主流程。

不要询问可以通过阅读代码确认的小问题。

---

## 17. 最终汇报格式

完成任务后必须输出：

```text
Summary:
- 接入了哪个模型。
- 新增或修改了哪些接入点。
- 是否保持了当前训练 / 测试 / 指标体系不变。

Changed files:
- <file path>: <修改原因>

Interface:
- 输入格式:
- 输出格式:
- 是否使用 wrapper:
- wrapper 做了什么:

Verification:
- Import check:
- Instantiation check:
- Forward check:
- Train/test smoke check:

Notes:
- 尚未解决的问题。
- 与外部原论文实现可能存在的差异。
- 后续用户需要运行的命令。
```

不得隐藏失败项。

不得把未验证的内容写成已完成。

---

## 18. 禁止行为清单

禁止：

* 重写整个项目；
* 替换训练 pipeline；
* 替换测试 pipeline；
* 替换 dataset loader；
* 替换 metric；
* 修改 baseline 模型；
* 静默改变实验设置；
* 为单个模型引入复杂架构；
* 添加无关功能；
* 编造实验结果；
* 编造论文结论；
* 删除无关文件；
* 清理无关代码；
* 大范围格式化；
* 未验证就声称完成；
* 不看代码直接凭经验修改。

---

## 19. 完成标准

一次模型接入任务只有在满足以下条件时才算完成：

1. 新模型可以通过当前仓库已有模型选择机制被选择。
2. 新模型可以正常 import。
3. 新模型可以正常 instantiate。
4. 新模型可以接受当前 pipeline 的输入。
5. 新模型输出兼容当前 loss / metric。
6. 当前 baseline 行为没有被修改。
7. 数据集、训练循环、测试循环、评价指标没有被无故修改。
8. 最小验证结果已经如实汇报。

如果其中任意一项不满足，任务必须标记为 incomplete，并说明原因。

---

## 20. 推荐任务启动提示词

当用户要求 Codex 接入一个新模型时，可以按以下方式启动：

```text
You are working in this repository under AGENTS.md.

Task:
Integrate <MODEL_NAME> from <PAPER_OR_REPOSITORY> into the existing experiment pipeline for fair comparison.

Constraints:
- Do not change dataset loading.
- Do not change train/test loops unless absolutely necessary.
- Do not change loss or metrics.
- Do not change the existing baseline model.
- Use the existing model selection/import mechanism.
- Add only the minimal model file, wrapper, and registration needed.
- Do not fabricate experiment results.

Before coding:
1. Inspect the current model interface.
2. Inspect how train/test scripts select models.
3. Identify expected input/output tensor shapes.
4. Inspect the external model implementation if available.
5. Propose a minimal implementation plan.

After coding:
1. Run import / instantiation / forward smoke checks.
2. Run a minimal train/test smoke check if feasible.
3. Report changed files and verification results honestly.
```
