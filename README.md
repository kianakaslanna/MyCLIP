# 1.'vanilla', 'MaskCLIP', 'GEM', 'SCLIP', 'ClearCLIP'
## /home/qta2szh/CLIP/ClearCLIP/configs/base_config.py里修改模型
## /home/qta2szh/CLIP/ClearCLIP/configs里选择数据集
## cfg_DATASET.py里修改数据集路径
## /home/qta2szh/CLIP/ClearCLIP/work_logs/results.txt分析结果
'''
python eval.py     --config /home/qta2szh/CLIP/ClearCLIP/configs/cfg_voc20.py     --work-dir /home/qta2szh/CLIP/ClearCLIP/work_logs
python eval.py     --config /home/qta2szh/CLIP/ClearCLIP/configs/cfg_coco_stuff164k.py  --work-dir /home/qta2szh/CLIP/ClearCLIP/work_logs
python eval.py     --config /home/qta2szh/CLIP/ClearCLIP/configs/cfg_ade20k.py  --work-dir /home/qta2szh/CLIP/ClearCLIP/work_logs
'''

# 2.NACLIP
## /home/qta2szh/CLIP/NACLIP/configs/base_config.py里修改模型
## /home/qta2szh/CLIP/NACLIP/configs里选择数据集
## cfg_DATASET.py里修改数据集路径
## 终端输出结果
'''
python eval.py     --config /home/qta2szh/CLIP/NACLIP/configs/cfg_voc20.py
'''

