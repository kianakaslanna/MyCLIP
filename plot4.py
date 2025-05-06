import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================== 数据准备 ==================
data = {
    'Arch': ['ViT-B/16']*6 + ['ViT-L/14']*6,
    'Method': ['ClearCLIP', 'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'MYCLIP']*2,
    'aAcc': [53.85, 45.08, 47.83, 54.14, 54.25, 55.05,
             53.85, 45.08, 47.83, 54.14, 54.25, 55.05],
    'mIoU': [39.39, 33.22, 31.88, 39.60, 38.95, 39.42,
             39.39, 33.22, 31.88, 39.60, 38.95, 39.42],
    'mAcc': [56.41, 50.18, 50.87, 56.42, 56.29, 57.82,
             56.41, 50.18, 50.87, 56.42, 56.29, 57.82]
}

df = pd.DataFrame(data)

# ================== 绘制柱状图 ==================
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold'
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
colors = ['#4E79A7', '#F28E2B', '#59A14F']  # 学术风格配色
bar_width = 0.25
methods = df['Method'].unique()
x = np.arange(len(methods))

# ViT-B/16柱状图
for i, metric in enumerate(['aAcc', 'mIoU', 'mAcc']):
    values = df[df['Arch'] == 'ViT-B/16'].groupby('Method')[metric].mean()
    ax1.bar(x + i*bar_width, values, width=bar_width, color=colors[i], label=metric)

ax1.set_title('ViT-B/16 Performance', pad=12)
ax1.set_xticks(x + bar_width)
ax1.set_xticklabels(methods, rotation=45, ha='right')
ax1.grid(axis='y', linestyle='--', alpha=0.7)
ax1.set_ylim(0, 65)

# ViT-L/14柱状图
for i, metric in enumerate(['aAcc', 'mIoU', 'mAcc']):
    values = df[df['Arch'] == 'ViT-L/14'].groupby('Method')[metric].mean()
    ax2.bar(x + i*bar_width, values, width=bar_width, color=colors[i])

ax2.set_title('ViT-L/14 Performance', pad=12)
ax2.set_xticks(x + bar_width)
ax2.set_xticklabels(methods, rotation=45, ha='right')
ax2.grid(axis='y', linestyle='--', alpha=0.7)

# 添加图例
fig.legend(['aAcc', 'mIoU', 'mAcc'], 
           loc='upper center',
           bbox_to_anchor=(0.5, 1.05),
           ncol=3,
           frameon=False)

# ================== 性能汇总表格 ==================
plt.rcParams.update({'font.size': 11})

# 计算平均结果
df_avg = df.groupby('Method').agg({
    'aAcc': 'mean',
    'mIoU': 'mean', 
    'mAcc': 'mean'
}).reset_index()

# 表格数据准备
columns = ['Method', 'aAcc', 'mIoU', 'mAcc']
cell_text = [[row['Method'], 
              f"{row['aAcc']:.2f}", 
              f"{row['mIoU']:.2f}", 
              f"{row['mAcc']:.2f}"] for _, row in df_avg.iterrows()]

# 高亮最大值
max_values = df_avg[['aAcc', 'mIoU', 'mAcc']].max()
highlight_color = '#FFF2CC'  # 浅橙色背景

# 创建表格
fig_table, ax_table = plt.subplots(figsize=(10, 4))
ax_table.axis('off')

table = ax_table.table(
    cellText=cell_text,
    colLabels=columns,
    loc='center',
    cellLoc='center',
    colColours=['#F5F5F5']*4
)

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.8)

# 应用高亮
for row in range(1, len(cell_text)+1):
    for col in [1, 2, 3]:  # 对应aAcc, mIoU, mAcc列
        cell_value = float(cell_text[row-1][col])
        if cell_value == max_values[col-1]:
            table[row, col].set_facecolor(highlight_color)

# 设置边框
for key, cell in table.get_celld().items():
    cell.set_linewidth(0.5)
    cell.set_edgecolor('#444444')

# 标题
ax_table.set_title('Average Performance Comparison Across Architectures', 
                  pad=20, fontsize=12, fontweight='bold')

# ================== 保存结果 ==================
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig('vit_comparison.png', dpi=300, bbox_inches='tight')
fig_table.savefig('performance_table.png', dpi=300, bbox_inches='tight')
plt.show()