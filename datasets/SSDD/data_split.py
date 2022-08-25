import os    
import random

trainset_ratio = 0.6
testset_ratio = 0.2
valset_ratio = 0.2
images_root = "/media/hp3090/HDD-2T/renjunjie/SAR_DETECTION/mmdetection/datasets/SSDD/Annotations/"
save_txt_root = "/media/hp3090/HDD-2T/renjunjie/SAR_DETECTION/mmdetection/datasets/SSDD/txt_list/"
if not os.path.isdir(save_txt_root):
    os.makedirs(save_txt_root)

train_txt = open(f'{save_txt_root}/train_set.txt', 'w')
test_txt = open(f'{save_txt_root}/test_set.txt', 'w')
val_txt = open(f'{save_txt_root}/val_set.txt', 'w')

testset_ratio /= 1-trainset_ratio
images_list = []
for file in os.listdir(images_root):
    # 遍历该类别下的.jpg文件
    if os.path.splitext(file)[1] == '.xml':
        image_path = os.path.join(images_root, file)
        images_list.append(image_path)

train_images = random.sample(images_list, int(len(images_list)*trainset_ratio))
test_val = list(set(images_list).difference(set(train_images)))
test_images = random.sample(test_val, int(len(test_val)*testset_ratio))
val_images = list(set(test_val).difference(set(test_images)))

print("train_images:", len(train_images))
print("test_images:", len(test_images))
print("val_images:", len(val_images))

for image in train_images:
    train_txt.write(image + '\n')
train_txt.close()
for image in test_images:
    test_txt.write(image + '\n')
test_txt.close()
for image in val_images:
    val_txt.write(image + '\n')
val_txt.close()







