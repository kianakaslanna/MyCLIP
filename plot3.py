import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

# 读取并预处理数据
df = pd.read_excel("clip_performance.xlsx", sheet_name="Sheet1")
df['Arch'] = df['VIT']
df['Method'] = df['MODEL']

# 筛选目标方法和数据集
target_methods = ["vanilla", "MaskCLIP", "SCLIP", "ClearCLIP", "GEM", "MYCLIP"]
target_datasets = ["VOC20", "COCOStuff", "ADE20K"]
architectures = ["ViT/B/16", "ViT/L/14"]

def generate_arch_table(arch):
    # 架构特定数据处理
    arch_df = df[(df["Arch"] == arch) & 
                (df["Method"].isin(target_methods)) &
                (df["Dataset"].isin(target_datasets))]
    
    # 数据重塑
    pivot_df = arch_df.pivot_table(
        index="Method",
        columns="Dataset",
        values=["aAcc", "mIoU", "mAcc"],
        aggfunc='mean'
    )
    
    # 列名处理
    pivot_df.columns = [f"{dataset}\n{metric}" for (metric, dataset) in pivot_df.columns]
    
    # 计算平均mIoU并排序
    mIoU_cols = [col for col in pivot_df.columns if 'mIoU' in col]
    pivot_df['Avg. mIoU'] = pivot_df[mIoU_cols].mean(axis=1)
    pivot_df = pivot_df.sort_values('Avg. mIoU', ascending=False)
    
    # 构建展示表格
    display_df = pivot_df.reset_index()[['Method'] + 
                                      [col for col in pivot_df.columns if 'VOC20' in col] +
                                      [col for col in pivot_df.columns if 'COCOStuff' in col] +
                                      [col for col in pivot_df.columns if 'ADE20K' in col] +
                                      ['Avg. mIoU']]
    
    # 生成表格
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 9
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    
    # 表格数据
    cell_text = []
    for _, row in display_df.iterrows():
        cell_text.append([
            row['Method'],
            *[f"{row[col]:.1f}" if not pd.isnull(row[col]) else "" 
             for col in display_df.columns[1:]]
        ])
    
    # 绘制表格
    table = ax.table(
        cellText=cell_text,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center",
        colColours=["#F5F5F5"]*len(display_df.columns)
    )
    
    # 斑马条纹
    for i in range(len(display_df)):
        color = "#F3F3F3" if i%2 else "#FFFFFF"
        for j in range(len(display_df.columns)):
            table[(i+1, j)].set_facecolor(color)
    
    # 高亮最大值
    for col in range(1, len(display_df.columns)):
        values = [float(cell_text[i][col]) for i in range(len(cell_text)) if cell_text[i][col]]
        if values:
            max_value = max(values)
            for row in range(len(cell_text)):
                if cell_text[row][col] and float(cell_text[row][col]) == max_value:
                    table[(row+1, col)].set_facecolor("#FFF2CC")
    
    # 样式优化
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # 添加边框
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor("#444444")
    
    # 调整列宽
    col_widths = [0.2] + [0.15]*(len(display_df.columns)-1)
    for col, width in enumerate(col_widths):
        table.auto_set_column_width(col=col)
    
    plt.title(f"Performance Comparison ({arch})", pad=5, fontsize=12, fontweight='bold')
    safe_arch = arch.replace("/", "_")
    plt.savefig(f"{safe_arch}_comparison_2.png", dpi=300, bbox_inches="tight")
    plt.close()

# 生成两个架构的表格
for arch in architectures:
    generate_arch_table(arch)