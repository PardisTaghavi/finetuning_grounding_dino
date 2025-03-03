

Download data: https://drive.google.com/file/d/1gkHVsMl0e8SaQctjcTJXSohh7SoWmYXF/view?usp=sharing

Download model: https://drive.google.com/file/d/1z_CjEKcWnhzygdtafdjzyUHn81yUC51Y/view?usp=sharing


```
pip install torch torchvision
pip install -U transformers
mim install mmengine 
pip install mmcv==2.2.0
pip install mmdet==3.3.0
```


run
```
./tools/dist_train.sh configs/mm_grounding_dino/ade20k/grounding_dino_swin_finetune_ade20k.py 2 --work-dir ADE20K_DATA
```


