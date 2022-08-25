# 这个新的配置文件继承自一个原始配置文件，只需要突出必要的修改部分即可
_base_ = '/media/hp3090/HDD-2T/renjunjie/SAR_DETECTION/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'

# 我们需要对头中的类别数量进行修改来匹配数据集的标注
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
    )
)

# 修改数据集相关设置
dataset_type = 'CocoDataset'
classes = ('ship',)
data = dict( 
    train=dict(
        img_prefix='/media/hp3090/HDD-2T/renjunjie/SAR_DETECTION/mmdetection/datasets/SSDD/JPEGImages/',
        classes=classes,
        ann_file='/media/hp3090/HDD-2T/renjunjie/SAR_DETECTION/mmdetection/datasets/SSDD/train_set.json'
    ),
    val=dict(
        img_prefix='/media/hp3090/HDD-2T/renjunjie/SAR_DETECTION/mmdetection/datasets/SSDD/JPEGImages/',
        classes=classes,
        ann_file='/media/hp3090/HDD-2T/renjunjie/SAR_DETECTION/mmdetection/datasets/SSDD/val_set.json'
    ),
    test=dict(
        img_prefix='/media/hp3090/HDD-2T/renjunjie/SAR_DETECTION/mmdetection/datasets/SSDD/JPEGImages/',
        classes=classes,
        ann_file='/media/hp3090/HDD-2T/renjunjie/SAR_DETECTION/mmdetection/datasets/SSDD/test_set.json'
    )
)

# 我们可以使用预训练的 Faster R-CNN 来获取更好的性能
load_from = '/media/hp3090/HDD-2T/renjunjie/SAR_DETECTION/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'