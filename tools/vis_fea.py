# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from mmdet.apis import (inference_detector,
                        init_detector, show_result_pyplot)

                    
device = 'cuda'
config = r"/media/hp3090/HDD-2T/renjunjie/SAR_DETECTION/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_ssdd_20220411/faster_rcnn_r50_fpn_1x_ssdd_20220411.py"
ckpt =+ r"/media/hp3090/HDD-2T/renjunjie/SAR_DETECTION/mmdetection/work_dirs/faster_rcnn_r50_fpn_1x_ssdd_20220411/latest.pth"
img = r"/media/hp3090/HDD-2T/renjunjie/SAR_DETECTION/mmdetection/datasets/SSDD/JPEGImages/000736.jpg"
score_thr = 0.8


class HookValues():
    """
        后期新增，记录中间反传梯度
    """
    def __init__(self, layer, layerName):
        # register a hook to save values of activations and gradients
        self.layerName = layerName
        self.activations = None
        self.gradients = None
        self.forward_hook = layer.register_forward_hook(self.hook_fn_act)
        self.backward_hook = layer.register_backward_hook(self.hook_fn_grad)

    def hook_fn_act(self, module, input, output):
        self.activations = output

    def hook_fn_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def show_feature_map(out_video, img_src, conv_features) -> None:
    conv_features = conv_features.cuda().squeeze(0)

    for i in range(conv_features.shape[0]):
        heatmap = conv_features[i].cpu().numpy()
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()
        heatmap = cv2.resize(heatmap,(img_src.shape[0],img_src.shape[1]))
        heatmap = np.uint8(255*heatmap)

        # cv2.imwrite('./superimg.jpg',heatmap)#保存结果
        out_video.write(heatmap)  # 写入帧

        # cv2.imshow('superimg',heatmap)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # q退出
            break
        # cv2.waitKey(30)


if __name__ == '__main__':

    # build the model from a config file and a checkpoint file
    model = init_detector(config, ckpt, device=device)
    # hook values
    hookValuesMap = {}
    for idx, (name, module) in enumerate(model.named_modules()):
        print(idx, "-", name)
        if isinstance(module, torch.nn.Conv2d):
            hookValuesMap[name] = HookValues(module, name)
    result = inference_detector(model, img)

    
    # img_src = cv2.imread(img, 2)
    # fourcc = cv2.VideoWriter_fourcc(*"MJPG")uperimg',heatmap)
    #             # if cv2.waitKey(1) & 0xFF == ord('q'):  # q退出
    #             #     break
    #             # cv2.waitKey(3)):
    #     if isinstance(module, torch.nn.Conv2d):
    #         print(name, "->", hookValuesMap[name].activations.shape)
    #         conv_features = torch.sigmoid(hookValuesMap[name].activations)
    #         conv_features = conv_features.cuda().squeeze(0)

    #         for i in range(conv_features.shape[0]):
    #             heatmap = conv_features[i].cpu().numpy()
    #             heatmap -= heatmap.min()
    #             heatmap /= heatmap.max()
    #             heatmap = cv2.resize(heatmap,(img_src.shape[1],img_src.shape[0]))
    #             heatmap = np.uint8(255*heatmap)

    #             # 保存视频
    #             cv2.putText(heatmap, name+"_"+str(i), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    #             out_video.write(heatmap)  # 写入帧

    #             # 保存图像
    #             # 判断保存的子文件夹是否存在，不存在则创建
    #             if not os.path.exists('../out_img/'+name):
    #                 os.makedirs('../out_img/'+name)
    #             cv2.imwrite("../out_img/"+name+"/"+str(i)+".jpg", heatmap)
    #             # 保存卷积核
    #             # 判断保存的子文件夹是否存在，不存在则创建
    #             # if not os.path.exists('../out_img/'+name+"/kernel"):
    #             #     os.makedirs('../out_img/'+name+"/kernel")
    #             #     kernal_value = module.weight.data.cpu().numpy()
    #             #     kernal_value = kernal_value.squeeze(0)
    #                 # 将kernal_value转换为图像




    #             # cv2.imshow('superimg',heatmap)
    #             # if cv2.waituperimg',heatmap)
    #             # if cv2.waitKey(1) & 0xFF == ord('q'):  # q退出
    #             #     break
    #             # cv2.waitKey(30)

    # out_video.release()
    # cv2.destroyAllWindows()
    
    # 定义列表用于存储中间层的输入或者输出
    # module_name = []
    # p_in = []
    # p_out = []

    # def hook_fn(module, inputs, outputs):
    #     print(module_name)
    #     module_name.append(module.__class__)
    #     p_in.append(inputs)
    #     p_out.append(outputs)

    # model.rpn_head.rpn_cls.register_forward_hook(hook_fn)


    # show the results
    show_result_pyplot(model, img, result, score_thr=score_thr)
    # model.extract_feat()
    # model_str = model.__str__()

    # for k in range(len(module_name)):
    #     print(p_in[k][0].shape)
    #     print(p_out[k].shape)
    #     # show_feature_map(img_file, p_in[k][0])
    #     show_feature_map(img, torch.sigmoid(p_out[k]))
    #     print()