# 定义数据集类型，这里是 CocoDataset，表示数据格式遵循 COCO 标注格式
dataset_type = "CocoDataset"
# 数据根目录，所有数据的相对路径将以这个目录为基准。
data_root = "data/coco/"
#  图像归一化配置
# mean,std用于归一化图像的均值和标准差，通常为预训练模型的标准值
# 如果to_rgb为 True，表示输入图像需要从 BGR 转为 RGB
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

# 数据管道
# 训练数据管道
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="Resize",
        img_scale=[
            (1333, 800),
            (1333, 768),
            (1333, 736),
            (1333, 704),
            (1333, 672),
            (1333, 640),
        ],
        multiscale_mode="value",
        keep_ratio=True,
    ),
    dict(type="RandomFlip", flip_ratio=0.5),
    dict(
        type="Normalize",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True,
    ),
    dict(type="Pad", size_divisor=32),
    dict(type="DefaultFormatBundle"),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
]

# 测试数据管道
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
            ),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

# 数据分布配置
data = dict(
    # 每张 GPU 上的批量大小
    samples_per_gpu=1,
    # 每张 GPU 分配的数据加载线程数
    workers_per_gpu=1,
    # 定义训练、验证和测试集的标注文件和图像路径
    train=dict(
        type="CocoDataset",
        ann_file="data/coco/annotations/train.json",
        img_prefix="data/coco/JPEGImages/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
            dict(
                type="Resize",
                img_scale=[
                    (1333, 800),
                    (1333, 768),
                    (1333, 736),
                    (1333, 704),
                    (1333, 672),
                    (1333, 640),
                ],
                multiscale_mode="value",
                keep_ratio=True,
            ),
            dict(type="RandomFlip", flip_ratio=0.5),
            dict(
                type="Normalize",
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True,
            ),
            dict(type="Pad", size_divisor=32),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_masks"]),
        ],
    ),
    # 验证
    val=dict(
        type="CocoDataset",
        ann_file="data/coco/annotations/val.json",
        img_prefix="data/coco/JPEGImages/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(
                        type="Normalize",
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True,
                    ),
                    dict(type="Pad", size_divisor=32),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
    ),
    # 测试
    test=dict(
        type="CocoDataset",
        ann_file="data/coco/annotations/test.json",
        img_prefix="data/coco/JPEGImages/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(1333, 800),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(
                        type="Normalize",
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True,
                    ),
                    dict(type="Pad", size_divisor=32),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
    ),
)

# 训练配置
# 评估，自动保存最佳模型，评估指标包括 bbox 和 segm
evaluation = dict(save_best="auto", metric=["bbox", "segm"])
# 优化器：SGD,设置了学习率、动量和权重衰减
optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# 学习率设置
lr_config = dict(
    policy="step",
    # 训练初期采用线性预热（warmup）
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    # 在第 27 和 33 个 epoch 减小学习率。
    step=[27, 33],
)
# 训练时长，总共训练 35 个 epoch
runner = dict(type="EpochBasedRunner", max_epochs=35)
# 检查点保存，每隔 5 个 epoch 保存一次模型权重
checkpoint_config = dict(interval=5)
# 日志，每 50 个迭代打印一次日志，支持控制台和 Tensorboard
log_config = dict(
    interval=50, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)
custom_hooks = [dict(type="NumClassCheckHook")]
dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None
workflow = [("train", 1), ("val", 1)]
opencv_num_threads = 0
mp_start_method = "fork"
auto_scale_lr = dict(enable=False, base_batch_size=16)

# 模型配置
# 模型总体结构：SOLOv2
model = dict(
    type="SOLOv2",
    # 主干网络
    backbone=dict(
        # ResNet-101
        type="ResNet",
        depth=101,
        num_stages=4,
        # 输出特征的层级索引
        out_indices=(0, 1, 2, 3),
        # frozen_stages: 冻结的网络层数，用于固定某些参数（如预训练模型的早期层）
        frozen_stages=1,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="pretained/solov2_r101_dcn_fpn_3x_coco_20220513_214734-16c966cb.pth",
        ),
        style="pytorch",
        dcn=dict(type="DCNv2", deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
    ),

    # 特征金字塔：FPN
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5,
    ),

    # 分割头
    mask_head=dict(
        type="SOLOV2Head",
        # num_classes: 类别数，这里是 1（假设是单类别任务）
        num_classes=1,
        in_channels=256,
        feat_channels=512,
        stacked_convs=4,
        strides=[8, 8, 16, 32, 32],
        scale_ranges=((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048)),
        pos_scale=0.2,
        num_grids=[40, 36, 24, 16, 12],
        cls_down_index=0,
        mask_feature_head=dict(
            feat_channels=128,
            start_level=0,
            end_level=3,
            out_channels=256,
            mask_stride=4,
            norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
            conv_cfg=dict(type="DCNv2"),
        ),
        # 掩码损失与分类损失
        loss_mask=dict(type="DiceLoss", use_sigmoid=True, loss_weight=3.0),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),
        dcn_cfg=dict(type="DCNv2"),
        dcn_apply_to_all_conv=True,
    ),

    # 测试配置
    test_cfg=dict(
        nms_pre=500,
        # score_thr: 最低置信度阈值
        score_thr=0.1,
        # mask_thr: 掩码二值化的阈值
        mask_thr=0.5,
        filter_thr=0.05,
        kernel="gaussian",
        sigma=2.0,
        max_per_img=100,
    ),
)
