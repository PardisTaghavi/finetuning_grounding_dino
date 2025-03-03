Download few-shot data: https://drive.google.com/file/d/1gkHVsMl0e8SaQctjcTJXSohh7SoWmYXF/view?usp=drive_link


Download data for self-training: https://drive.google.com/file/d/1gkHVsMl0e8SaQctjcTJXSohh7SoWmYXF/view?usp=sharing

Download model: https://drive.google.com/file/d/1Q_DEBxPzcSqOpXvzSpIX0pv68wurrG05/view?usp=drive_link


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


