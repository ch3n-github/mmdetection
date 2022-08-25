# model settings
model = dict(
    type='FasterRCNN',

    # 1. backbone
    backbone=dict(
        # 骨架网络类名
        type='ResNet',
        # 表示使用 ResNet50
        depth=50,
        # ResNet 系列包括 stem+4个 stage 输出
        num_stages=4,
        # 表示本模块输出的特征图索引，(0, 1, 2, 3),表示4个 stage 输出都需要，
        # 其 stride 为 (4,8,16,32)，channel 为 (256, 512, 1024, 2048)
        out_indices=(0, 1, 2, 3),
        # 表示固定 stem 加上第一个 stage 的权重，不进行训练
        frozen_stages=1,
        # 所有的 BN 层的可学习参数都不需要梯度，也就不会进行参数更新
        norm_cfg=dict(type='BN', requires_grad=True),
        # backbone 所有的 BN 层的均值和方差都直接采用全局预训练值，不进行更新
        norm_eval=True,
        # 默认采用 pytorch 模式
        style='pytorch',
        # 使用 pytorch 提供的在 imagenet 上面训练过的权重作为预训练权重
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),

    # 2. neck
    neck=dict(
        type='FPN',
        # ResNet 模块输出的4个尺度特征图通道数
        in_channels=[256, 512, 1024, 2048],
        # FPN 输出的每个尺度输出特征图通道
        out_channels=256,
        # FPN 输出特征图个数
        num_outs=5
    ),

    # 3. rpn_head
    rpn_head=dict(
        type='RPNHead',
        # FPN 层输出特征图通道数
        in_channels=256,
        # 中间特征图通道数
        feat_channels=256,
        # anchor_generator
        anchor_generator=dict(
            type='AnchorGenerator',
            # 相当于 octave_base_scale，表示每个特征图的 base scales
            scales=[8],
            # 每个特征图有 3 个高宽比例
            ratios=[0.5, 1.0, 2.0],
            # 特征图对应的 stride，必须和特征图 stride 一致，不可以随意更改
            strides=[4, 8, 16, 32, 64]
        ),
        # bbox_coder
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        # loss
        loss_cls=dict(
            type='CrossEntropyLoss', 
            use_sigmoid=True,
            loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),

    # 4. roi_head
    roi_head=dict(
        # 一次 refine head，另外对应的是级联结构
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=dict(
            # 2 个共享 FC 模块
            type='Shared2FCBBoxHead',
            # 输入通道数，相等于 FPN 输出通道
            in_channels=256,
            # 中间 FC 层节点个数
            fc_out_channels=1024,
            # RoIAlign 或 RoIPool 输出的特征图大小
            roi_feat_size=7,
            # 类别个数
            num_classes=80,
            # bbox 编解码策略，除了参数外和 RPN 相同
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            # 影响 bbox 分支的通道数，True 表示 4 通道输出，False 表示 4×num_classes 通道输出
            reg_class_agnostic=False,
            # CE Loss
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            # L1 Loss
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        )
    ),

    # 5. model training and testing settings
    train_cfg=dict(
        rpn=dict(
            # BBox Assigner
            assigner=dict(
                # 最大 IoU 原则分配器
                type='MaxIoUAssigner',
                # 正样本阈值
                pos_iou_thr=0.7,
                # 负样本阈值
                neg_iou_thr=0.3,
                # 正样本阈值下限
                min_pos_iou=0.3,
                match_low_quality=True,
                # 忽略 bboxes 的阈值，-1 表示不忽略
                ignore_iof_thr=-1
            ),
            # BBox Sampler
            # 和 RetinaNet 采用 Focal Loss 处理正负样本不平衡不同，
            # Faster R-CNN 是通过正负样本采样模块来克服
            sampler=dict(
                # 随机采样
                type='RandomSampler',
                # 采样后每张图片的训练样本总数，不包括忽略样本
                num=256,
                # 正样本比例
                pos_fraction=0.5,
                # 正负样本比例，用于确定负样本采样个数上界
                neg_pos_ub=-1,
                # 是否加入 gt 作为 proposals 以增加高质量正样本数
                add_gt_as_proposals=False
            ),
            allowed_border=-1,
            pos_weight=-1,
            debug=False
        ),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0
        ),
        rcnn=dict(
            assigner=dict(
                # 和 RPN 一样，正负样本定义参数不同
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1
            ),
            sampler=dict(
                # 和 RPN 一样，随机采样参数不同
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False
        )
    ),

    test_cfg=dict(
        # RPN 不仅仅要自己进行训练，还要同时输出 RoI
        # 一个核心的问题是如何得到这些 RoI，实际上是调用了 RPN 的 test 过程
        rpn=dict(
            # nms 前每个输出层最多保留 1000 个预测框
            nms_pre=1000,
            # nms 后每张图片最多保留 1000 个预测框
            max_per_img=1000,
            # nms 阈值
            nms=dict(type='nms', iou_threshold=0.7),
            # 过滤掉的最小 bbox 尺寸
            min_bbox_size=0
        ),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100
        )
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))
