import matplotlib.pyplot as plt
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据：方法名称，参数量(M)，FLOPS(G)，nIoU(%)
methods = [
    {'name': 'DNANet', 'params': 4.70, 'flops': 32.13, 'niou': 50.25},
    {'name': 'UIUNet', 'params': 50.54, 'flops': 122.63, 'niou': 48.66},
    {'name': 'DNANet-DTUM', 'params': 1.21, 'flops': 99.98, 'niou': 51.32},
    {'name': 'ResUNet-DTUM', 'params': 0.30, 'flops': 23.29, 'niou': 51.33},
    {'name': 'LMAFormer', 'params': 390.05, 'flops': 279.63, 'niou': 32.12},
    {'name': 'DNANet-RFR', 'params': 4.80, 'flops': 63.22, 'niou': 63.46},
    {'name': 'ResUNet-RFR', 'params': 1.01, 'flops': 39.85, 'niou': 64.45},
    {'name': 'DeepPro', 'params': 0.02, 'flops': 8.47, 'niou': 50.75},
    {'name': 'MISTNet', 'params': 0.85, 'flops': 19.31, 'niou': 70.48},
    {'name': 'MS$^2$DNet', 'params': 0.57, 'flops': 10.14, 'niou': 72.32},
]

# 提取数据
names = [m['name'] for m in methods]
params = [m['params'] for m in methods]
flops = [m['flops'] for m in methods]
niou = [m['niou'] for m in methods]

# 颜色映射
colors = ['#FF8C00', '#F898CB', '#4DAF4A', '#731A73', '#FFE29E', '#0098B2','#3D505A',
        '#699ECA','#66C1A4','#EC3E31']

# 创建气泡大小的缩放因子（根据FLOPS）
min_flops, max_flops = min(flops), max(flops)
bubble_sizes = [(f - min_flops) / (max_flops - min_flops) * 4000 + 100 for f in flops]

# 定义断轴映射函数 - 只放大间距，保留原始值
# 区域1: 0-5 -> 显示为 0-15 (放大间距)
# 区域2: 47-52 -> 显示为 18-23
# 区域3: 388-395 -> 显示为 25-32
def map_x(x):
    if x <= 5:
        return x * 3  # 0-5 -> 0-15 (间距放大3倍)
    elif x <= 52:
        return 18 + (x - 47)  # 47-52 -> 18-23
    else:
        return 25 + (x - 388) * (7/8)  # 388-395 -> 25-31

# 映射后的x坐标
mapped_x = [map_x(p) for p in params]

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))

# 绘制气泡
for i, name in enumerate(names):
    ax.scatter(mapped_x[i], niou[i],
               s=bubble_sizes[i],
               c=colors[i],
               alpha=0.6,
               edgecolors=colors[i],
               linewidth=2)

# 设置y轴
ax.set_ylabel('nIoU (%)', fontsize=14, fontweight='bold')
ax.set_ylim(28, 73)
ax.set_yticks([30, 40, 50, 60, 70])
ax.set_yticklabels([30, 40, 50, 60, 70], fontsize=12)

# 设置x轴 - 自定义刻度，保留原始值但放大间距
# 区域1: 0-5
tick_pos_1 = [v * 3 for v in [0, 1, 2, 3, 4, 5]]
tick_labels_1 = ['0', '1', '2', '3', '4', '5']

# 区域2: 47-52
tick_pos_2 = [18 + (v - 47) for v in [47, 48, 49, 50, 51, 52]]
tick_labels_2 = ['47', '48', '49', '50', '51', '52']

# 区域3: 388-395
tick_pos_3 = [25 + (v - 388) * (7/8) for v in [388, 390, 392, 394, 395]]
tick_labels_3 = ['388', '390', '392', '394', '395']

# 合并刻度
all_ticks = tick_pos_1 + tick_pos_2 + tick_pos_3
all_labels = tick_labels_1 + tick_labels_2 + tick_labels_3

ax.set_xticks(all_ticks)
ax.set_xticklabels(all_labels, fontsize=12)
ax.set_xlim(-0.2, 29)

# 设置x轴标签
ax.set_xlabel('Parameters (M)', fontsize=14, fontweight='bold')

# 添加网格
ax.grid(True, linestyle='--', alpha=0.3)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/blob_plot.png', format='png', 
            dpi=1000, bbox_inches='tight')
print("Bubble plot saved to: /home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/blob_plot.png")

plt.show()
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.patches as mpatches
# import matplotlib.lines as mlines

# # ====== 关键：SVG & 论文设置 ======
# plt.rcParams['svg.fonttype'] = 'none'  # 保持文字为文本（可编辑）
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['lines.antialiased'] = True

# # ====== 数据 ======
# methods = [
#     {'name': 'DNANet', 'params': 4.70, 'flops': 32.13, 'niou': 50.25},
#     {'name': 'UIUNet', 'params': 50.54, 'flops': 122.63, 'niou': 48.66},
#     {'name': 'DNANet-DTUM', 'params': 1.21, 'flops': 99.98, 'niou': 51.32},
#     {'name': 'ResUNet-DTUM', 'params': 0.30, 'flops': 23.29, 'niou': 51.33},
#     {'name': 'LMAFormer', 'params': 390.05, 'flops': 279.63, 'niou': 32.12},
#     {'name': 'DNANet-RFR', 'params': 4.80, 'flops': 63.22, 'niou': 63.46},
#     {'name': 'ResUNet-RFR', 'params': 1.01, 'flops': 39.85, 'niou': 64.45},
#     {'name': 'DeepPro', 'params': 0.02, 'flops': 8.47, 'niou': 50.75},
#     {'name': 'MISTNet', 'params': 0.85, 'flops': 19.31, 'niou': 70.48},
#     {'name': 'MS$^2$DNet', 'params': 0.57, 'flops': 10.14, 'niou': 72.32},
# ]

# names = [m['name'] for m in methods]
# params = [m['params'] for m in methods]
# flops = [m['flops'] for m in methods]
# niou = [m['niou'] for m in methods]

# colors = ['#FF8C00', '#F898CB', '#4DAF4A', '#731A73', '#FFE29E',
#           '#0098B2', '#3D505A', '#699ECA', '#66C1A4', '#EC3E31']

# # ====== 气泡大小 ======
# min_flops, max_flops = min(flops), max(flops)
# bubble_sizes = [(f - min_flops) / (max_flops - min_flops) * 4000 + 100 for f in flops]

# # ====== 断轴映射 ======
# def map_x(x):
#     if x <= 5:
#         return x * 3
#     elif x <= 52:
#         return 18 + (x - 47)
#     else:
#         return 25 + (x - 388) * (7/8)

# mapped_x = [map_x(p) for p in params]

# # ====== 画图 ======
# fig, ax = plt.subplots(figsize=(8, 6))

# for i in range(len(names)):
#     ax.scatter(mapped_x[i], niou[i],
#                s=bubble_sizes[i],
#                c=colors[i],
#                alpha=0.7,
#                edgecolors=colors[i],
#                linewidth=1.8)

# # ====== 坐标轴 ======
# ax.set_ylabel('nIoU (%)', fontsize=12, fontweight='bold')
# ax.set_ylim(28, 73)
# ax.set_yticks([30, 40, 50, 60, 70])

# # X轴刻度
# tick_pos_1 = [v * 3 for v in [0, 1, 2, 3, 4, 5]]
# tick_labels_1 = ['0', '1', '2', '3', '4', '5']

# tick_pos_2 = [18 + (v - 47) for v in [47, 48, 49, 50, 51, 52]]
# tick_labels_2 = ['47', '48', '49', '50', '51', '52']

# tick_pos_3 = [25 + (v - 388) * (7/8) for v in [388, 390, 392]]
# tick_labels_3 = ['388', '390', '392']

# ax.set_xticks(tick_pos_1 + tick_pos_2 + tick_pos_3)
# ax.set_xticklabels(tick_labels_1 + tick_labels_2 + tick_labels_3, fontsize=9)
# ax.set_xlim(-0.5, 29)

# ax.set_xlabel('Parameters (M)', fontsize=12, fontweight='bold')

# # ====== 网格 ======
# ax.grid(True, linestyle='--', alpha=0.3)

# # ====== 断轴符号（关键）=====
# ax.text(16.5, 29.5, '//', fontsize=14, ha='center')
# ax.text(24, 29.5, '//', fontsize=14, ha='center')

# # ====== 图例 ======
# legend_handles = []
# for i, name in enumerate(names):
#     handle = mlines.Line2D([], [], color=colors[i], marker='o',
#                            linestyle='None', markersize=9, label=name)
#     legend_handles.append(handle)

# rect = mpatches.FancyBboxPatch(
#     (-1.0, 29.0), 23.0, 21.0,
#     boxstyle="round,pad=0.5",
#     facecolor="#E8E8E8",
#     alpha=0.4,
#     edgecolor="#CCCCCC",
#     linewidth=0.5,
#     zorder=1
# )
# ax.add_patch(rect)

# legend = ax.legend(
#     handles=legend_handles,
#     labels=names,
#     ncol=2,
#     loc='lower left',
#     bbox_to_anchor=(-0.5, 29.5, 22, 20),
#     mode="expand",
#     frameon=False,
#     fontsize=9.5,
#     labelspacing=0.9,
#     borderaxespad=0
# )
# legend.set_zorder(10)

# # ====== 保存SVG ======
# filename = 'bubble_plot_paper.svg'
# plt.savefig(filename, format='svg', bbox_inches='tight')
# print(f"Saved to {filename}")

# plt.show()
# import matplotlib.pyplot as plt
# import numpy as np

# # ===== 关键：SVG + PPT 兼容 =====
# plt.rcParams['svg.fonttype'] = 'none'   # 保持文字为文本
# plt.rcParams['font.sans-serif'] = ['Arial']  # PPT默认字体（关键！）
# plt.rcParams['axes.unicode_minus'] = False

# # ===== 数据 =====
# methods = [
#     {'name': 'DNANet', 'params': 4.70, 'flops': 32.13, 'niou': 50.25},
#     {'name': 'UIUNet', 'params': 50.54, 'flops': 122.63, 'niou': 48.66},
#     {'name': 'DNANet-DTUM', 'params': 1.21, 'flops': 99.98, 'niou': 51.32},
#     {'name': 'ResUNet-DTUM', 'params': 0.30, 'flops': 23.29, 'niou': 51.33},
#     {'name': 'LMAFormer', 'params': 390.05, 'flops': 279.63, 'niou': 32.12},
#     {'name': 'DNANet-RFR', 'params': 4.80, 'flops': 63.22, 'niou': 63.46},
#     {'name': 'ResUNet-RFR', 'params': 1.01, 'flops': 39.85, 'niou': 64.45},
#     {'name': 'DeepPro', 'params': 0.02, 'flops': 8.47, 'niou': 50.75},
#     {'name': 'MISTNet', 'params': 0.85, 'flops': 19.31, 'niou': 70.48},
#     {'name': 'MS$^2$DNet', 'params': 0.57, 'flops': 10.14, 'niou': 72.32},
# ]

# names = [m['name'] for m in methods]
# params = [m['params'] for m in methods]
# flops = [m['flops'] for m in methods]
# niou = [m['niou'] for m in methods]

# colors = ['#FF8C00', '#F898CB', '#4DAF4A', '#731A73', '#FFE29E',
#           '#0098B2','#3D505A', '#699ECA','#66C1A4','#EC3E31']

# # ===== 气泡大小 =====
# min_flops, max_flops = min(flops), max(flops)
# bubble_sizes = [(f - min_flops) / (max_flops - min_flops) * 4000 + 100 for f in flops]

# # ===== 断轴映射 =====
# def map_x(x):
#     if x <= 5:
#         return x * 3
#     elif x <= 52:
#         return 18 + (x - 47)
#     else:
#         return 25 + (x - 388) * (7/8)

# mapped_x = [map_x(p) for p in params]

# # ===== 图尺寸（非常关键：匹配PPT比例）=====
# fig, ax = plt.subplots(figsize=(10, 7))  # 比8x6更适合PPT

# # ===== 绘制 =====
# for i in range(len(names)):
#     ax.scatter(mapped_x[i], niou[i],
#                s=bubble_sizes[i],
#                c=colors[i],
#                alpha=0.7,
#                edgecolors=colors[i],
#                linewidth=1.5)

# # ===== 坐标轴 =====
# ax.set_ylabel('nIoU (%)', fontsize=12, fontweight='bold')
# ax.set_ylim(28, 73)
# ax.set_yticks([30, 40, 50, 60, 70])

# tick_pos_1 = [v * 3 for v in [0, 1, 2, 3, 4, 5]]
# tick_labels_1 = ['0', '1', '2', '3', '4', '5']

# tick_pos_2 = [18 + (v - 47) for v in [47, 48, 49, 50, 51, 52]]
# tick_labels_2 = ['47', '48', '49', '50', '51', '52']

# tick_pos_3 = [25 + (v - 388) * (7/8) for v in [388, 390, 392]]
# tick_labels_3 = ['388', '390', '392']

# ax.set_xticks(tick_pos_1 + tick_pos_2 + tick_pos_3)
# ax.set_xticklabels(tick_labels_1 + tick_labels_2 + tick_labels_3, fontsize=11)
# ax.set_xlim(-0.2, 29)

# ax.set_xlabel('Parameters (M)', fontsize=12, fontweight='bold')

# # 网格
# ax.grid(True, linestyle='--', alpha=0.3)

# # ===== 去掉多余边框（PPT更干净）=====
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# import matplotlib.lines as mlines
# import matplotlib.patches as mpatches

# # ===== 创建图例句柄 =====
# # legend_handles = []
# # for i, name in enumerate(names):
# #     handle = mlines.Line2D([], [], 
# #                            color=colors[i], 
# #                            marker='o', 
# #                            linestyle='None',
# #                            markersize=8,
# #                            label=name)
# #     legend_handles.append(handle)

# # # ===== 背景框（灰色半透明）=====
# # rect = mpatches.FancyBboxPatch(
# #     (-1.0, 29.0),   # 左下角位置（根据你图调过的）
# #     23.0,           # 宽
# #     21.0,           # 高
# #     boxstyle="round,pad=0.5",
# #     facecolor="#E8E8E8",
# #     alpha=0.4,
# #     edgecolor="#CCCCCC",
# #     linewidth=0.5,
# #     zorder=1
# # )
# # ax.add_patch(rect)

# # # ===== 图例 =====
# # legend = ax.legend(
# #     handles=legend_handles,
# #     labels=names,
# #     ncol=2,
# #     loc='lower left',
# #     bbox_to_anchor=(-0.5, 29.5, 22, 20),
# #     mode="expand",
# #     frameon=False,
# #     fontsize=10,
# #     labelspacing=0.8,
# #     borderaxespad=0
# # )

# # legend.set_zorder(10)

# # ===== 保存 SVG（核心）=====
# save_path = 'bubble_plot_ppt.svg'
# plt.savefig(save_path, format='svg', bbox_inches='tight')

# print(f"SVG saved to: {save_path}")

# plt.show()