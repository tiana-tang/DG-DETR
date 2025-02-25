# �������ݼ����ͣ������� CocoDataset����ʾ���ݸ�ʽ��ѭ COCO ��ע��ʽ
dataset_type = "CocoDataset"
# ���ݸ�Ŀ¼���������ݵ����·���������Ŀ¼Ϊ��׼��
data_root = "data/coco/"
#  ͼ���һ������
# mean,std���ڹ�һ��ͼ��ľ�ֵ�ͱ�׼�ͨ��ΪԤѵ��ģ�͵ı�׼ֵ
# ���to_rgbΪ True����ʾ����ͼ����Ҫ�� BGR תΪ RGB
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)

# ���ݹܵ�
# ѵ�����ݹܵ�
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

# �������ݹܵ�
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

# ���ݷֲ�����
data = dict(
    # ÿ�� GPU �ϵ�������С
    samples_per_gpu=1,
    # ÿ�� GPU ��������ݼ����߳���
    workers_per_gpu=1,
    # ����ѵ������֤�Ͳ��Լ��ı�ע�ļ���ͼ��·��
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
    # ��֤
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
    # ����
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

# ѵ������
# �������Զ��������ģ�ͣ�����ָ����� bbox �� segm
evaluation = dict(save_best="auto", metric=["bbox", "segm"])
# �Ż�����SGD,������ѧϰ�ʡ�������Ȩ��˥��
optimizer = dict(type="SGD", lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# ѧϰ������
lr_config = dict(
    policy="step",
    # ѵ�����ڲ�������Ԥ�ȣ�warmup��
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.3333333333333333,
    # �ڵ� 27 �� 33 �� epoch ��Сѧϰ�ʡ�
    step=[27, 33],
)
# ѵ��ʱ�����ܹ�ѵ�� 35 �� epoch
runner = dict(type="EpochBasedRunner", max_epochs=35)
# ���㱣�棬ÿ�� 5 �� epoch ����һ��ģ��Ȩ��
checkpoint_config = dict(interval=5)
# ��־��ÿ 50 ��������ӡһ����־��֧�ֿ���̨�� Tensorboard
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

# ģ������
# ģ������ṹ��SOLOv2
model = dict(
    type="SOLOv2",
    # ��������
    backbone=dict(
        # ResNet-101
        type="ResNet",
        depth=101,
        num_stages=4,
        # ��������Ĳ㼶����
        out_indices=(0, 1, 2, 3),
        # frozen_stages: �����������������ڹ̶�ĳЩ��������Ԥѵ��ģ�͵����ڲ㣩
        frozen_stages=1,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="pretained/solov2_r101_dcn_fpn_3x_coco_20220513_214734-16c966cb.pth",
        ),
        style="pytorch",
        dcn=dict(type="DCNv2", deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
    ),

    # ������������FPN
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        num_outs=5,
    ),

    # �ָ�ͷ
    mask_head=dict(
        type="SOLOV2Head",
        # num_classes: ������������� 1�������ǵ��������
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
        # ������ʧ�������ʧ
        loss_mask=dict(type="DiceLoss", use_sigmoid=True, loss_weight=3.0),
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),
        dcn_cfg=dict(type="DCNv2"),
        dcn_apply_to_all_conv=True,
    ),

    # ��������
    test_cfg=dict(
        nms_pre=500,
        # score_thr: ������Ŷ���ֵ
        score_thr=0.1,
        # mask_thr: �����ֵ������ֵ
        mask_thr=0.5,
        filter_thr=0.05,
        kernel="gaussian",
        sigma=2.0,
        max_per_img=100,
    ),
)
