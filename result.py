import pandas as pd
from io import StringIO

def parse_log_data(text):
    """解析日志数据并返回结构化DataFrame"""
    entries = []
    current_entry = {}
    
    # 预处理文本格式
    clean_text = text.replace("\t", " ").replace("\n\n", "\n")
    
    for line in clean_text.split("\n"):
        line = line.strip()
        if not line:
            continue
            
        # 检测新配置块
        if line.startswith("cfg_"):
            if current_entry:
                entries.append(current_entry)
                current_entry = {}
            dataset_type = line.split("_", 1)[1].upper()
            current_entry["Dataset"] = f"{dataset_type.split()[0]}Dataset"
            continue
            
        # 解析键值对
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            
            # 特殊字段处理
            if key == "VIT":
                current_entry["VIT"] = value.replace("-", "/")
            elif key == "Dataset":
                current_entry["Dataset"] = value.replace("Pascal", "")
            elif key in ["aAcc", "mIoU", "mAcc"]:
                try:
                    current_entry[key] = float(value)
                except:
                    current_entry[key] = None
            else:
                current_entry[key] = value
                
    # 添加最后一个条目
    if current_entry:
        entries.append(current_entry)
    
    # 创建DataFrame
    df = pd.DataFrame(entries)
    
    # 统一字段格式
    df["CLIP"] = "CLIP"  # 所有记录CLIP字段固定
    df["Dataset"] = df["Dataset"].str.replace("Dataset", "").str.strip()
    
    # 列排序和重命名
    column_order = ["CLIP", "VIT", "MODEL", "Dataset", "aAcc", "mIoU", "mAcc"]
    df = df[column_order]
    
    return df.dropna(subset=["MODEL"]).reset_index(drop=True)

# 示例数据（替换为实际读取文件内容）
raw_data = """[cfg_voc20
aAcc: 88.84
mIoU: 80.92
mAcc: 90.35
VIT: ViT-B/16
CLIP: CLIP
MODEL: ClearCLIP
Dataset: PascalVOC20Dataset
cfg_coco_stuff164k
aAcc: 39.74
mIoU: 23.89
mAcc: 42.63
VIT: ViT-B/16
CLIP: CLIP
MODEL: ClearCLIP
Dataset: COCOStuffDataset
cfg_ade20k
aAcc: 40.19
mIoU: 16.65
mAcc: 35.44
VIT: ViT-B/16
CLIP: CLIP
MODEL: ClearCLIP
Dataset: ADE20KDataset
cfg_ade20k
aAcc: 32.62
mIoU: 15.02
mAcc: 36.96
VIT: ViT-L-14
CLIP: CLIP
MODEL: ClearCLIP
Dataset: ADE20KDataset
cfg_coco_stuff164k
aAcc: 33.71
mIoU: 19.89
mAcc: 42.05
VIT: ViT-L-14
CLIP: CLIP
MODEL: ClearCLIP
Dataset: COCOStuffDataset
cfg_voc20
aAcc: 88.01
mIoU: 79.99
mAcc: 91.05
VIT: ViT-L-14
CLIP: CLIP
MODEL: ClearCLIP
Dataset: PascalVOC20Dataset
cfg_voc20
aAcc: 83.83
mIoU: 75.87
mAcc: 89.9
VIT: ViT-L-14
CLIP: CLIP
MODEL: vanilla
Dataset: PascalVOC20Dataset
cfg_coco_stuff164k
aAcc: 28.17
mIoU: 14.74
mAcc: 36.73
VIT: ViT-L-14
CLIP: CLIP
MODEL: vanilla
Dataset: COCOStuffDataset
cfg_ade20k
aAcc: 23.28
mIoU: 9.28
mAcc: 28.23
VIT: ViT-L-14
CLIP: CLIP
MODEL: vanilla
Dataset: ADE20KDataset
cfg_voc20
aAcc: 83.79
mIoU: 75.74
mAcc: 88.32
VIT: ViT-B/16
CLIP: CLIP
MODEL: vanilla
Dataset: PascalVOC20Dataset
cfg_coco_stuff164k
aAcc: 29.5
mIoU: 14.53
mAcc: 32.78
VIT: ViT-B/16
CLIP: CLIP
MODEL: vanilla
Dataset: COCOStuffDataset
cfg_ade20k
aAcc: 21.9
mIoU: 9.13
mAcc: 25.12
VIT: ViT-B/16
CLIP: CLIP
MODEL: vanilla
Dataset: ADE20KDataset
cfg_voc20
aAcc: 75.24
mIoU: 61.41
mAcc: 75.75
VIT: ViT-B/16
CLIP: CLIP
MODEL: MaskCLIP
Dataset: PascalVOC20Dataset
cfg_coco_stuff164k
aAcc: 30.8
mIoU: 18.38
mAcc: 36.35
VIT: ViT-B/16
CLIP: CLIP
MODEL: MaskCLIP
Dataset: COCOStuffDataset
cfg_ade20k
aAcc: 37.07
mIoU: 13.64
mAcc: 33.49
VIT: ViT-B/16
CLIP: CLIP
MODEL: MaskCLIP
Dataset: ADE20KDataset
cfg_voc20
aAcc: 79.25
mIoU: 65.09
mAcc: 80.0
VIT: ViT-L-14
CLIP: CLIP
MODEL: MaskCLIP
Dataset: PascalVOC20Dataset
cfg_coco_stuff164k
aAcc: 29.57
mIoU: 17.62
mAcc: 39.38
VIT: ViT-L-14
CLIP: CLIP
MODEL: MaskCLIP
Dataset: COCOStuffDataset
cfg_ade20k
aAcc: 35.04
mIoU: 15.11
mAcc: 40.23
VIT: ViT-L-14
CLIP: CLIP
MODEL: MaskCLIP
Dataset: ADE20KDataset
cfg_voc20
aAcc: 88.46
mIoU: 80.2
mAcc: 90.74
VIT: ViT-L-14
CLIP: CLIP
MODEL: GEM
Dataset: PascalVOC20Dataset
cfg_coco_stuff164k
aAcc: 33.97
mIoU: 20.32
mAcc: 41.72
VIT: ViT-L-14
CLIP: CLIP
MODEL: GEM
Dataset: COCOStuffDataset
cfg_ade20k
aAcc: 32.37
mIoU: 15.02
mAcc: 36.16
VIT: ViT-L-14
CLIP: CLIP
MODEL: GEM
Dataset: ADE20KDataset
cfg_voc20
aAcc: 88.47
mIoU: 80.23
mAcc: 89.7
VIT: ViT-B/16
CLIP: CLIP
MODEL: GEM
Dataset: PascalVOC20Dataset
cfg_coco_stuff164k
aAcc: 39.81
mIoU: 24.39
mAcc: 43.64
VIT: ViT-B/16
CLIP: CLIP
MODEL: GEM
Dataset: COCOStuffDataset
cfg_ade20k
aAcc: 41.73
mIoU: 17.43
mAcc: 36.53
VIT: ViT-B/16
CLIP: CLIP
MODEL: GEM
Dataset: ADE20KDataset
cfg_voc20
aAcc: 86.16
mIoU: 77.88
mAcc: 89.37
VIT: ViT-B/16
CLIP: CLIP
MODEL: SCLIP
Dataset: PascalVOC20Dataset
cfg_coco_stuff164k
aAcc: 39.39
mIoU: 23.58
mAcc: 42.27
VIT: ViT-B/16
CLIP: CLIP
MODEL: SCLIP
Dataset: COCOStuffDataset
cfg_ade20k
aAcc: 43.08
mIoU: 16.94
mAcc: 35.64
VIT: ViT-B/16
CLIP: CLIP
MODEL: SCLIP
Dataset: ADE20KDataset
cfg_voc20
aAcc: 87.65
mIoU: 79.24
mAcc: 90.55
VIT: ViT-L-14
CLIP: CLIP
MODEL: SCLIP
Dataset: PascalVOC20Dataset
cfg_coco_stuff164k
aAcc: 34.32
mIoU: 20.45
mAcc: 42.13
VIT: ViT-L-14
CLIP: CLIP
MODEL: SCLIP
Dataset: COCOStuffDataset
cfg_ade20k
aAcc: 34.91
mIoU: 15.63
mAcc: 37.75
VIT: ViT-L-14
CLIP: CLIP
MODEL: SCLIP
Dataset: ADE20KDataset
cfg_voc20
aAcc: 88.62
mIoU: 80.44
mAcc: 91.22
VIT: ViT-L-14
CLIP: CLIP
MODEL: MYCLIP
Dataset: PascalVOC20Dataset
cfg_coco_stuff164k
aAcc: 35.22
mIoU: 20.91
mAcc: 44.11
VIT: ViT-L-14
CLIP: CLIP
MODEL: MYCLIP
Dataset: COCOStuffDataset
cfg_ade20k
aAcc: 36.94
mIoU: 17.12
mAcc: 41.83
VIT: ViT-L-14
CLIP: CLIP
MODEL: MYCLIP
Dataset: ADE20KDataset
cfg_voc20
aAcc: 87.13
mIoU: 77.93
mAcc: 88.65
VIT: ViT-B/16
CLIP: CLIP
MODEL: MYCLIP
Dataset: PascalVOC20Dataset
cfg_coco_stuff164k
aAcc: 39.39
mIoU: 23.19
mAcc: 43.26
VIT: ViT-B/16
CLIP: CLIP
MODEL: MYCLIP
Dataset: COCOStuffDataset
cfg_ade20k
aAcc: 43.01
mIoU: 16.94
mAcc: 37.82
VIT: ViT-B/16
CLIP: CLIP
MODEL: MYCLIP
Dataset: ADE20KDataset
]"""

# 执行转换
df = parse_log_data(raw_data)

# 添加平均值列
df['Avg'] = df[['aAcc', 'mIoU', 'mAcc']].mean(axis=1).round(2)

# 保存到Excel
output_columns = ["CLIP", "VIT", "MODEL", "Dataset", "aAcc", "mIoU", "mAcc", "Avg"]
df[output_columns].to_excel("clip_performance.xlsx", index=False)

# 打印前5行验证
print(df.head().to_markdown(index=False))