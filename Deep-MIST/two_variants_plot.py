# import matplotlib.pyplot as plt
# import numpy as np

# # 设置中文字体支持
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
# plt.rcParams['axes.unicode_minus'] = False

# # 创建图表和双Y轴
# fig, ax1 = plt.subplots(figsize=(10, 7))

# # 创建第二个Y轴
# ax2 = ax1.twinx()

# # ============================================
# # 数据填充区域 - 请在这里填入你的实际数据
# # ============================================

# # X轴刻度标签
# x_labels = ['3~5', '5~7', '7~9', '9~11', '11~13']
# x_positions = np.arange(len(x_labels))  # [0, 1, 2, 3, 4]

# # 左Y轴数据 (折线图) - PD (%)
# # 请在这里填入5个数值，范围建议在60-100之间

# left_y_data = [
#     95.01,  # 3~5 对应的PD值
#     87.65,  # 5~7 对应的PD值
#     97.38,  # 7~9 对应的PD值
#     93.21,  # 9~11 对应的PD值
#     77.7   # 11~13 对应的PD值
# ]

# # 右Y轴数据 (柱状图) - Fa (10^-6)
# # 请在这里填入5个数值，范围建议在0-6之间
# right_y_data = [
#     1.13,  # 3~5 对应的Fa值
#     4.15,  # 5~7 对应的Fa值
#     0.84,  # 7~9 对应的Fa值
#     5.26,  # 9~11 对应的Fa值
#     0.14   # 11~13 对应的Fa值
# ]

# # ============================================
# # 绘图设置
# # ============================================

# # 绘制柱状图 (左Y轴数据 - PD)
# bar_color = '#C4D7C1'
# #bar_color = '#B4C3DD'
# bars = ax1.bar(x_positions, left_y_data, width=0.6,
#                color=bar_color, alpha=0.7, label='Pd (%)')

# # 绘制折线图 (右Y轴数据 - FA)
# #line_color = '#F2B382'

# #line_color = '#E89995'
# #line_color = '#DEAC80'
# line_color = '#F7B971'
# line, = ax2.plot(x_positions, right_y_data, color=line_color,
#                  marker='o', markersize=8, linewidth=2.5,
#                  label='Fa (10⁻⁶)')

# # ============================================
# # 坐标轴设置
# # ============================================

# # 设置X轴
# ax1.set_xlabel('Average Speed of target (pixels/frame)', fontsize=12, fontweight='bold')
# ax1.set_xticks(x_positions)
# ax1.set_xticklabels(x_labels, fontsize=11)

# # 设置左Y轴 (柱状图 - PD)
# ax1.set_ylabel('Pd (%)', fontsize=12, fontweight='bold', color='black')
# ax1.set_ylim(60, 100)
# ax1.set_yticks([60, 70, 80, 90, 100])
# ax1.tick_params(axis='y', labelcolor='black', labelsize=11)
# ax1.spines['left'].set_color('black')
# ax1.spines['left'].set_linewidth(1.5)

# # 设置右Y轴 (折线图 - FA)
# ax2.set_ylabel('Fa (10⁻⁶)', fontsize=12, fontweight='bold', color='black')
# ax2.set_ylim(0, 15)
# ax2.set_yticks([0, 3, 6, 9, 12, 15])
# ax2.tick_params(axis='y', labelcolor='black', labelsize=11)
# ax2.spines['right'].set_color('black')
# ax2.spines['right'].set_linewidth(1.5)

# # 设置X轴颜色
# ax1.spines['bottom'].set_color('black')
# ax1.spines['bottom'].set_linewidth(1.5)
# ax1.tick_params(axis='x', labelsize=11)

# # ============================================
# # 图例和网格
# # ============================================

# # 添加图例
# lines = [bars, line]
# labels = [l.get_label() for l in lines]
# ax1.legend(lines, labels, loc='upper left', fontsize=11, framealpha=0.9)

# # 添加网格线 (基于左Y轴)
# ax1.grid(True, linestyle='--', alpha=0.3, color='gray')

# # 设置标题
# plt.title('Performance vs Target Speed', fontsize=14, fontweight='bold', pad=15)

# # 调整布局
# plt.tight_layout()

# # 保存图片
# output_path = '/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/Statistical_chart/dual_axis_plot_MSMSDNet.png'
# plt.savefig(output_path, dpi=300, bbox_inches='tight')
# print(f"图表已保存至: {output_path}")

# # 显示图表
# plt.show()
# import matplotlib.pyplot as plt
# import numpy as np

# # ===== 新增：SVG可编辑 =====
# plt.rcParams['svg.fonttype'] = 'none'

# # 设置中文字体支持
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
# plt.rcParams['axes.unicode_minus'] = False

# # 创建图表和双Y轴
# fig, ax1 = plt.subplots(figsize=(10, 7))

# # 创建第二个Y轴
# ax2 = ax1.twinx()

# # ============================================
# # 数据填充区域
# # ============================================

# x_labels = ['3~5', '5~7', '7~9', '9~11', '11~13']
# x_positions = np.arange(len(x_labels))

# left_y_data = [95.01, 87.65, 97.38, 93.21, 77.7]
# #left_y_data = [94.45, 83.89, 96.25, 92.07, 74.07]

# right_y_data = [1.13, 4.15, 0.84, 5.26, 0.14]
# #right_y_data = [1.92, 8.28, 1.57, 5.86,13.4]
# # ============================================
# # 绘图
# # ============================================

# bar_color = '#C4D7C1'
# #bar_color = '#B4C3DD'
# bars = ax1.bar(x_positions, left_y_data, width=0.6,
#                color=bar_color, alpha=0.7, label='Pd (%)')

# line_color = '#F7B971'
# # #line_color = '#E89995'
# # #line_color = '#DEAC80'
# #line_color = '#E89995'
# line, = ax2.plot(x_positions, right_y_data, color=line_color,
#                  marker='o', markersize=8, linewidth=2.5,
#                  label='Fa (10⁻⁶)')

# # ============================================
# # 坐标轴设置
# # ============================================

# ax1.set_xlabel('Average Speed of target (pixels/frame)', fontsize=14, fontweight='bold')
# ax1.set_xticks(x_positions)
# ax1.set_xticklabels(x_labels, fontsize=11)

# ax1.set_ylabel('Pd (%)', fontsize=14, fontweight='bold', color='black')
# ax1.set_ylim(60, 100)
# ax1.set_yticks([60, 70, 80, 90, 100])
# ax1.tick_params(axis='y', labelcolor='black', labelsize=11)
# ax1.spines['left'].set_color('black')
# ax1.spines['left'].set_linewidth(1.5)

# ax2.set_ylabel('Fa (10⁻⁶)', fontsize=14, fontweight='bold', color='black')
# ax2.set_ylim(0, 15)
# ax2.set_yticks([0, 3, 6, 9, 12, 15])
# ax2.tick_params(axis='y', labelcolor='black', labelsize=11)
# ax2.spines['right'].set_color('black')
# ax2.spines['right'].set_linewidth(1.5)

# ax1.spines['bottom'].set_color('black')
# ax1.spines['bottom'].set_linewidth(1.5)
# ax1.tick_params(axis='x', labelsize=11)

# # ============================================
# # 图例和网格
# # ============================================

# lines = [bars, line]
# labels = [l.get_label() for l in lines]
# ax1.legend(lines, labels, loc='upper left', fontsize=11, framealpha=0.9)

# ax1.grid(True, linestyle='--', alpha=0.3, color='gray')

# # ===== 删除标题（直接删掉这一行即可）=====
# # plt.title(...)

# # 调整布局
# plt.tight_layout()

# # ===== 修改：保存为SVG =====
# output_path = '/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/Statistical_chart/dual_axis_plot_MSMSDNet.svg'
# plt.savefig(output_path, format='svg', bbox_inches='tight')

# print(f"图表已保存至: {output_path}")

# plt.show()
# import matplotlib.pyplot as plt
# import numpy as np

# # ===== 新增：SVG可编辑 =====
# plt.rcParams['svg.fonttype'] = 'none'

# # 设置中文字体支持
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
# plt.rcParams['axes.unicode_minus'] = False

# # 创建图表和双Y轴
# fig, ax1 = plt.subplots(figsize=(10, 7))

# # 创建第二个Y轴
# ax2 = ax1.twinx()

# # ============================================
# # 数据填充区域
# # ============================================

# x_labels = ['3~5', '5~7', '7~9', '9~11', '11~13']
# x_positions = np.arange(len(x_labels))

# left_y_data = [95.01, 87.65, 97.38, 93.21, 77.7]
# #left_y_data = [94.45, 83.89, 96.25, 92.07, 74.07]

# right_y_data = [1.13, 4.15, 0.84, 5.26, 0.14]
# #right_y_data = [1.92, 8.28, 1.57, 5.86,13.4]
# # ============================================
# # 绘图
# # ============================================

# bar_color = '#C4D7C1'
# #bar_color = '#B4C3DD'
# bars = ax1.bar(x_positions, left_y_data, width=0.6,
#                color=bar_color, alpha=0.7, label='Pd (%)')

# line_color = '#F7B971'
# # #line_color = '#E89995'
# # #line_color = '#DEAC80'
# #line_color = '#E89995'
# line, = ax2.plot(x_positions, right_y_data, color=line_color,
#                  marker='o', markersize=8, linewidth=2.5,
#                  label='Fa (10⁻⁶)')

# # ============================================
# # 坐标轴设置
# # ============================================

# ax1.set_xlabel('Average Speed of target (pixels/frame)', fontsize=14, fontweight='bold')
# ax1.set_xticks(x_positions)
# ax1.set_xticklabels(x_labels, fontsize=11)

# ax1.set_ylabel('Pd (%)', fontsize=14, fontweight='bold', color='black')
# ax1.set_ylim(60, 100)
# ax1.set_yticks([60, 70, 80, 90, 100])
# ax1.tick_params(axis='y', labelcolor='black', labelsize=11)
# ax1.spines['left'].set_color('black')
# ax1.spines['left'].set_linewidth(1.5)

# ax2.set_ylabel('Fa (10⁻⁶)', fontsize=14, fontweight='bold', color='black')
# ax2.set_ylim(0, 15)
# ax2.set_yticks([0, 3, 6, 9, 12, 15])
# ax2.tick_params(axis='y', labelcolor='black', labelsize=11)
# ax2.spines['right'].set_color('black')
# ax2.spines['right'].set_linewidth(1.5)

# ax1.spines['bottom'].set_color('black')
# ax1.spines['bottom'].set_linewidth(1.5)
# ax1.tick_params(axis='x', labelsize=11)

# # ============================================
# # 图例和网格
# # ============================================

# lines = [bars, line]
# labels = [l.get_label() for l in lines]
# ax1.legend(lines, labels, loc='upper left', fontsize=11, framealpha=0.9)

# ax1.grid(True, linestyle='--', alpha=0.3, color='gray')

# # ===== 删除标题（直接删掉这一行即可）=====
# # plt.title(...)

# # 调整布局
# plt.tight_layout()

# # ===== 修改：保存为SVG =====
# # 这里按照你的要求保留了原来的注释文字，仅修改了扩展名和保存参数
# output_path = '/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/Statistical_chart/dual_axis_plot_MSMSDNet.png'
# plt.savefig(output_path, dpi=600, bbox_inches='tight')

# print(f"图表已保存至: {output_path}")

# plt.show()
import matplotlib.pyplot as plt
import numpy as np

# ===== 新增：SVG可编辑 =====
plt.rcParams['svg.fonttype'] = 'none'

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 创建图表和双Y轴 
# ===== 修改：整体图形变窄，将 figsize 宽度由 10 改为了 8 =====
fig, ax1 = plt.subplots(figsize=(8, 7))

# 创建第二个Y轴
ax2 = ax1.twinx()

# ============================================
# 数据填充区域
# ============================================

x_labels = ['3~5', '5~7', '7~9', '9~11', '11~13']
x_positions = np.arange(len(x_labels))

#left_y_data = [95.01, 87.65, 97.38, 93.21, 77.7]
left_y_data = [94.45, 83.89, 96.25, 92.07, 74.07]

#right_y_data = [1.13, 4.15, 0.84, 5.26, 0.14]
right_y_data = [1.92, 8.28, 1.57, 5.86,13.4]
# ============================================
# 绘图
# ============================================

#bar_color = '#C4D7C1'
bar_color = '#B4C3DD'
# ===== 修改：柱状图变窄，将 width 由 0.6 改为了 0.4 =====
bars = ax1.bar(x_positions, left_y_data, width=0.4,
               color=bar_color, alpha=0.7, label='Pd')
# ===== 在柱状图顶部添加数值 =====
for bar in bars:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2,  # 水平居中
        height + 0.1,                       # 稍微高于柱子
        f'{height:.2f}',                    # 保留两位小数
        ha='center', va='bottom',
        fontsize=18, color='black'
    )

#line_color = '#F7B971'
# #line_color = '#E89995'
# #line_color = '#DEAC80'
line_color = '#E89995'
line, = ax2.plot(x_positions, right_y_data, color=line_color,
                 marker='o', markersize=12, linewidth=2.5,
                 label='Fa')
# ===== 在折线点添加数值 =====
for x, y in zip(x_positions, right_y_data):
    ax2.text(
        x,
        y + 0.2,              # 稍微高于点（避免重叠）
        f'{y:.2f}',
        ha='center', va='bottom',
        fontsize=18, color='black'
    )
# ============================================
# 坐标轴设置
# ============================================

ax1.set_xlabel('Average Speed of target (pixels/frame)', fontsize=22, fontweight='bold')
ax1.set_xticks(x_positions)
ax1.set_xticklabels(x_labels, fontsize=16)

ax1.set_ylabel('Pd (%)', fontsize=22, fontweight='bold', color='black')
ax1.set_ylim(60, 100)
ax1.set_yticks([60, 70, 80, 90, 100])
ax1.tick_params(axis='y', labelcolor='black', labelsize=18)
ax1.spines['left'].set_color('black')
ax1.spines['left'].set_linewidth(1.5)

ax2.set_ylabel('Fa (10⁻⁶)', fontsize=22, fontweight='bold', color='black')
ax2.set_ylim(0, 15)
ax2.set_yticks([0, 3, 6, 9, 12, 15])
ax2.tick_params(axis='y', labelcolor='black', labelsize=18)
ax2.spines['right'].set_color('black')
ax2.spines['right'].set_linewidth(1.5)

ax1.spines['bottom'].set_color('black')
ax1.spines['bottom'].set_linewidth(1.5)
ax1.tick_params(axis='x', labelsize=18)

# ============================================
# 图例和网格
# ============================================

lines = [bars, line]
labels = [l.get_label() for l in lines]
ax1.legend(
    lines,
    labels,
    loc='upper left',
    bbox_to_anchor=(0.16, 1),  
    fontsize=18,
    framealpha=0.9
)

ax1.grid(True, linestyle='--', alpha=0.3, color='gray')

# ===== 删除标题（直接删掉这一行即可）=====
# plt.title(...)

# 调整布局
plt.tight_layout()

# ===== 修改：保存为PNG =====
output_path = '/home/ubuntu/data/zhengqinxu/object_detection/Deep-MIST/Statistical_chart/dual_axis_plot_MISTNet.png'
plt.savefig(output_path, dpi=1000, bbox_inches='tight')

print(f"图表已保存至: {output_path}")

plt.show()