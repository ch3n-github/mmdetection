# model settings
model = dict(
    type='RetinaNet',
    backbone=dict(
        # 网络骨架名
        type='ResNet',
        # 使用ResNet50
        depth=50,
        # Resnet系列包括stem+4个stage输出
        num_stages=4,
        # 表示本模块输出的特征图索引,(0,1,2,3)表示4个stage输出均需要
        out_indices=(0, 1, 2, 3),
        # 表示固定stem加上第一个stage的权重，不进行训练
        frozen_stages=1,
        # 所有的BN层的可学习参数都不需要梯度，也不会进行参数更新
        norm_cfg=dict(type='BN', requires_grad=True),
        # backbone所有的BN层的均值和方差都直接采用全局预训练值，不进行更新
        norm_eval=True,
        # 默认采用pytorch模式
        style='pytorch',
        # 使用pytorch提供的在imagenet上面训练好的权重作为预训练权重
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    neck=dict(
        type='FPN',
        # ResNet模块输出的4个尺度特征图通道数
        in_channels=[256, 512, 1024, 2048],
        # FPN输出的每个尺度输出特征图通道
        out_channels=256,
        # 从输入多尺度特征图的第几个开始计算
        # 虽然输入是 4 个特征图，但是实际上 FPN 中仅仅用了后面三个
        start_level=1,
        # 额外输出特征图来源
        # 说明额外输出的 2 个特征图的来源是骨架网络输出，而不是 FPN 层本身输出又作为后面层的输入
        add_extra_convs='on_input',
        # FPN输出特征图个数
        num_outs=5
    ),
    # 下面对代码运行流程进行描述：
    # 1. 将 c3、c4 和 c5 三个特征图全部经过各自 1x1 卷积进行通道变换得到 m3~m5，输出通道统一为 256
    # 2. 从 m5(特征图最小)开始，先进行 2 倍最近邻上采样，然后和 m4 进行 add 操作，得到新的 m4
    # 3. 将新 m4 进行 2 倍最近邻上采样，然后和 m3 进行 add 操作，得到新的 m3
    # 4. 对 m5 和新融合后的 m4、m3，都进行各自的 3x3 卷积，得到 3 个尺度的最终输出 P5～P3
    # 5. 将 c5 进行 3x3 且 stride=2 的卷积操作，得到 P6
    # 6. 将 P6 再一次进行 3x3 且 stride=2 的卷积操作，得到 P7

    # P6 和 P7 目的是提供一个大感受野强语义的特征图，有利于大物体和超大物体检测。 在 RetinaNet 的 FPN 模块中只包括卷积，不包括 BN 和 ReLU。

    # 总结：FPN 模块接收 c3, c4, c5 三个特征图，输出 P3-P7 五个特征图，通道数都是 256, stride 为 (8,16,32,64,128)，
    # 其中大 stride (特征图小)用于检测大物体，小 stride (特征图大)用于检测小物体。
    bbox_head=dict(
        type='RetinaHead',
        # COCO数据集类别个数
        num_classes=80,
        # FPN 层输出特征图通道数
        in_channels=256,
        # 每个分支堆叠4层卷积
        stacked_convs=4,
        # 中间特征图通道数
        feat_channels=256,
        # 得到每个输出特征图的anchor
        anchor_generator=dict(
            type='AnchorGenerator',
            # 特征图 anchor 的 base scale, 值越大，所有 anchor 的尺度都会变大
            octave_base_scale=4,
            # 每个特征图有3个尺度，2**0, 2**(1/3), 2**(2/3)
            scales_per_octave=3,
            # 每个特征图有3个高宽比例
            ratios=[0.5, 1.0, 2.0],
            # 特征图对应的 stride，必须特征图 stride 一致，不可以随意更改
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            # 最大 IoU 原则分配器
            type='MaxIoUAssigner',
            # 正样本阈值
            pos_iou_thr=0.5,
            # 负样本阈值
            neg_iou_thr=0.4,
            # 正样本阈值下限
            min_pos_iou=0,
            # 忽略 bboes 的阈值，-1表示不忽略
            ignore_iof_thr=-1
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        # nms 前每个输出层最多保留1000个预测框
        nms_pre=1000,
        # 过滤掉的最小 bbox 尺寸
        min_bbox_size=0,
        # 分值阈值
        score_thr=0.05,
        # nms 方法和 nms 阈值
        nms=dict(type='nms', iou_threshold=0.5),
        # nms 方法和 nms 阈值
        max_per_img=100
    )
)
