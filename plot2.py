import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
# 读取并预处理数据
df = pd.read_excel("clip_performance.xlsx", sheet_name="Sheet1")
df['Arch'] = df['VIT']
df['Method'] = df['MODEL']

# 筛选目标方法
methods = ["vanilla", "MaskCLIP", "SCLIP", "ClearCLIP", "GEM", "MYCLIP"]
df = df[df["Method"].isin(methods)]

# 修正后的数据重塑逻辑
pivot_df = df.pivot_table(
    index=["Method", "Arch"],
    columns=["Dataset"],
    values=["aAcc", "mIoU", "mAcc"],
    aggfunc='mean'
)

# 扁平化多级列索引
pivot_df.columns = [f"{col[1]}\n{col[0]}" for col in pivot_df.columns]

# 计算平均值（仅数值列）
numeric_cols = pivot_df.select_dtypes(include=[np.number]).columns
pivot_df['Average'] = pivot_df[numeric_cols].mean(axis=1)

# 生成展示表格
display_df = pivot_df.reset_index()

# 表格样式设置
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
header_color = "#F5F5F5"
zebra_colors = ["#FFFFFF", "#F3F3F3"]
highlight_color = "#FFF2CC"

# 创建画布
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis("off")

# 生成表格数据
cell_text = []
for _, row in display_df.iterrows():
    formatted_row = [
        row["Method"],
        row["Arch"],
        *[f"{x:.1f}" if not pd.isnull(x) else "" for x in row[2:-1]],
        f"{row['Average']:.1f}"
    ]
    cell_text.append(formatted_row)

# 绘制表格
table = ax.table(
    cellText=cell_text,
    colLabels=display_df.columns,
    loc="center",
    cellLoc="center",
    colColours=[header_color]*len(display_df.columns)
)

# 斑马条纹设置
for i in range(len(display_df)):
    color = zebra_colors[i%2]
    for j in range(len(display_df.columns)):
        table[(i+1, j)].set_facecolor(color)

# 高亮最大值（排除方法/编码器列）
for col in range(2, len(display_df.columns)):
    values = [float(cell_text[i][col]) for i in range(len(cell_text)) if cell_text[i][col]]
    if values:
        max_value = max(values)
        for row in range(len(cell_text)):
            if cell_text[row][col] and float(cell_text[row][col]) == max_value:
                table[(row+1, col)].set_facecolor(highlight_color)

# 样式优化
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.8)

# 添加边框线
for key, cell in table.get_celld().items():
    cell.set_linewidth(0.5)
    cell.set_edgecolor("#CCCCCC")

# 调整列宽
col_widths = [0.15, 0.15] + [0.12]*(len(display_df.columns)-2)
for col, width in enumerate(col_widths):
    table.auto_set_column_width(col=col)

plt.title("Cross-Dataset Performance Comparison", pad=25, fontsize=14, fontweight='bold')
plt.savefig("paper_style_table.png", dpi=300, bbox_inches="tight")
plt.close()