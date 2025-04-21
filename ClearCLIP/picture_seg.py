import os
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from clearclip_segmentor import ClearCLIPSegmentation

# 配置路径
image_path = '/home/qta2szh/CLIP/pictures/school.jpeg'
output_dir = '/home/qta2szh/CLIP/ClearCLIP/images'
name_file_path = './configs/my_name.txt'
combined_output = os.path.join(output_dir, 'combined_result.png')
#name_list = ['sky', 'building', 'car', 'ground', 'tree', 'status', 'cloud']
#name_list = ['building', 'tree', 'human']
#name_list = ['playground']

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# ========== 第一部分：生成分割图像 ==========
# 加载并预处理图像
img = Image.open(image_path)
img_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
])(img)
img_tensor = img_tensor.unsqueeze(0).to('cuda')

# 写入类别文件
with open(name_file_path, 'w') as f:
    f.write('\n'.join(name_list))

# 定义模型组合
clip_types = ['CLIP']
model_types = ['vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP', 'MYCLIP']

# 生成所有分割结果
generated_files = []
for clip_type in clip_types:
    for model_type in model_types:
        try:
            # 模型初始化与预测
            model = ClearCLIPSegmentation(
                clip_type=clip_type,
                vit_type='ViT-B/16',
                model_type=model_type,
                name_path=name_file_path,
                ignore_residual=True,
                prob_thd=0.2
            )
            
            seg_pred = model.predict(img_tensor, data_samples=None).data.cpu().numpy().squeeze(0)
            
            # 保存分割结果
            output_name = f'status_{clip_type}_{model_type}.png'
            output_path = os.path.join(output_dir, output_name)
            
            plt.figure(figsize=(6, 6))
            plt.imshow(seg_pred, cmap='viridis')
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            print(f'Generated: {output_name}')
            generated_files.append(output_name)
            
        except Exception as e:
            print(f'Error: {clip_type}+{model_type} - {str(e)}')

# ========== 第二部分：合并所有图像 ==========
# 创建画布 (1行7列)
fig = plt.figure(figsize=(28, 4.5))  # 总宽度28英寸，高度4.5英寸
plt.rcParams['font.size'] = 10  # 标注字体大小

# 首先添加原始图像
ax = fig.add_subplot(1, 7, 1)  # 第一列
ax.imshow(img)
ax.set_title("Original\nImage", y=-0.18)  # 双行标题
ax.axis('off')

# 添加分割结果图像
for idx, filename in enumerate(generated_files, start=2):  # 从第二列开始
    # 解析模型信息
    model_info = filename.replace('status_CLIP_', '').replace('.png', '')
    
    # 加载图像
    img_seg = Image.open(os.path.join(output_dir, filename))
    
    # 添加子图
    ax = fig.add_subplot(1, 7, idx)
    ax.imshow(img_seg)
    ax.set_title(f"CLIP + {model_info}", y=-0.18)  # 标题下移
    ax.axis('off')

# 调整布局参数
plt.subplots_adjust(
    left=0.02,
    right=0.98,
    bottom=0.15,  # 增加底部空间
    top=0.85,
    wspace=0.3    # 列间距
)

# 保存合并结果
plt.savefig(combined_output, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n合并图像已保存至: {combined_output}")
print("结束！")