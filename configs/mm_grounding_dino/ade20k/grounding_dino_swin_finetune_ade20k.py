_base_ = '../grounding_dino_swin-l_pretrain_all.py'

load_from = '/home/avalocal/Mask2Former/mmdetection/zero_shot/grounding_dino_swin-l_pretrain_all-56d69e78.pth'

data_root = '/media/avalocal/T7/ADE/ADEChallengeData2016/'
class_name = ('bed', 'windowpane', 'cabinet', 'person', 'door', 'table', 'curtain',
         'chair', 'car', 'painting', 'sofa', 'shelf', 'mirror', 'armchair',
         'seat', 'fence', 'desk', 'wardrobe', 'lamp', 'bathtub', 'railing',
         'cushion', 'box', 'column', 'signboard', 'chest of drawers',
         'counter', 'sink', 'fireplace', 'refrigerator', 'stairs', 'case',
         'pool table', 'pillow', 'screen door', 'bookcase', 'coffee table',
         'toilet', 'flower', 'book', 'bench', 'countertop', 'stove', 'palm',
         'kitchen island', 'computer', 'swivel chair', 'boat',
         'arcade machine', 'bus', 'towel', 'light', 'truck', 'chandelier',
         'awning', 'streetlight', 'booth', 'television receiver', 'airplane',
         'apparel', 'pole', 'bannister', 'ottoman', 'bottle', 'van', 'ship',
         'fountain', 'washer', 'plaything', 'stool', 'barrel', 'basket', 'bag',
         'minibike', 'oven', 'ball', 'food', 'step', 'trade name', 'microwave',
         'pot', 'animal', 'bicycle', 'dishwasher', 'screen', 'sculpture',
         'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan',
         'plate', 'monitor', 'bulletin board', 'radiator', 'glass', 'clock',
         'flag')
palette = [(204, 5, 255), (230, 230, 230), (224, 5, 255),
                    (150, 5, 61), (8, 255, 51), (255, 6, 82), (255, 51, 7),
                    (204, 70, 3), (0, 102, 200), (255, 6, 51), (11, 102, 255),
                    (255, 7, 71), (220, 220, 220), (8, 255, 214),
                    (7, 255, 224), (255, 184, 6), (10, 255, 71), (7, 255, 255),
                    (224, 255, 8), (102, 8, 255), (255, 61, 6), (255, 194, 7),
                    (0, 255, 20), (255, 8, 41), (255, 5, 153), (6, 51, 255),
                    (235, 12, 255), (0, 163, 255), (250, 10, 15), (20, 255, 0),
                    (255, 224, 0), (0, 0, 255), (255, 71, 0), (0, 235, 255),
                    (0, 173, 255), (0, 255, 245), (0, 255, 112), (0, 255, 133),
                    (255, 0, 0), (255, 163, 0), (194, 255, 0), (0, 143, 255),
                    (51, 255, 0), (0, 82, 255), (0, 255, 41), (0, 255, 173),
                    (10, 0, 255), (173, 255, 0), (255, 92, 0), (255, 0, 245),
                    (255, 0, 102), (255, 173, 0), (255, 0, 20), (0, 31, 255),
                    (0, 255, 61), (0, 71, 255), (255, 0, 204), (0, 255, 194),
                    (0, 255, 82), (0, 112, 255), (51, 0, 255), (0, 122, 255),
                    (255, 153, 0), (0, 255, 10), (163, 255, 0), (255, 235, 0),
                    (8, 184, 170), (184, 0, 255), (255, 0, 31), (0, 214, 255),
                    (255, 0, 112), (92, 255, 0), (70, 184, 160), (163, 0, 255),
                    (71, 255, 0), (255, 0, 163), (255, 204, 0), (255, 0, 143),
                    (133, 255, 0), (255, 0, 235), (245, 0, 255), (255, 0, 122),
                    (255, 245, 0), (214, 255, 0), (0, 204, 255), (255, 255, 0),
                    (0, 153, 255), (0, 41, 255), (0, 255, 204), (41, 0, 255),
                    (41, 255, 0), (173, 0, 255), (0, 245, 255), (0, 255, 184),
                    (0, 92, 255), (184, 255, 0), (255, 214, 0), (25, 194, 194),
                    (102, 255, 0), (92, 0, 255)]
                    
                    

metainfo = dict(classes=class_name, palette=palette)
num_classes = len(class_name)
model = dict(bbox_head=dict(num_classes=num_classes))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    sampler=dict(_delete_=True, type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        _delete_=True,
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type='CocoDataset',
            data_root=data_root,
            metainfo=metainfo,
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=train_pipeline,
            return_classes=True,
            data_prefix=dict(img='images/training'),
            ann_file='ade20k_instance_train.json')))
 
backend_args = None
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(2560, 640), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]           


val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='ade20k_instance_val.json',
        data_prefix=dict(img='images/validation')))        
        
       
test_dataloader = val_dataloader


val_evaluator = dict(ann_file=data_root + 'ade20k_instance_val.json')
test_evaluator = val_evaluator

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=0.01),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.),
        'backbone': dict(lr_mult=0.1),
        'language_model': dict(lr_mult=0.0)
    }))

# learning policy
#max_epochs = 
max_iter = 10000
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iter,
    val_interval=1000)
    
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_iter,
        by_epoch=False,
        milestones=[2000, 2500],
        gamma=0.1)
]

default_hooks = dict(checkpoint=dict(max_keep_ckpts=1, save_best='auto'))


