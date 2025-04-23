import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_excel("clip_performance.xlsx", sheet_name="Sheet1")

# 预处理：直接使用现有列
df['Arch'] = df['VIT']  # 提取架构信息
df['Method'] = df['MODEL']  # 提取方法名称

# 筛选需要对比的方法
methods = ["vanilla", "MaskCLIP", "SCLIP", "ClearCLIP",  "GEM", "MYCLIP"]
df = df[df["Method"].isin(methods)]

# 设置可视化参数
sns.set(style="whitegrid", font_scale=1.2)
plt.rcParams["font.family"] = "Times New Roman"
colors = sns.color_palette("husl", 3)  # 三种指标颜色
architectures = ["ViT/B/16", "ViT/L/14"]
datasets = ["VOC20", "COCOStuff", "ADE20K"]
metrics = ["aAcc", "mIoU", "mAcc"]

# 创建可视化图表
for arch in architectures:
    arch_df = df[df["Arch"] == arch]
    
    # 创建画布和子图
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f"Performance Comparison ({arch})", y=1.05, fontsize=20)
    
    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        plot_df = arch_df[arch_df["Dataset"] == dataset]
        
        # 数据重塑为长格式
        melted = plot_df.melt(id_vars=["Method"], value_vars=metrics,
                             var_name="Metric", value_name="Score")
        
        # 绘制分组柱状图
        sns.barplot(
            data=melted,
            x="Method",
            y="Score",
            hue="Metric",
            palette=colors,
            ax=ax,
            order=methods  # 保持方法顺序一致
        )
        
        # 美化图表
        ax.set_title(dataset, fontsize=16)
        ax.set_ylabel("Score (%)", fontsize=14)
        ax.set_xlabel("")
        ax.set_ylim(0, 100)
        ax.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + 0.5,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=12)
        
        # 仅在第一个子图保留图例
        if idx != 0:
            ax.get_legend().remove()

    # 添加统一图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', 
              bbox_to_anchor=(0.92, 0.95), ncol=3, title="Metrics")
    
    # 修复路径格式：替换斜杠为下划线
    safe_arch = arch.replace("/", "_")
    plt.tight_layout()
    plt.savefig(f"{safe_arch}_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()