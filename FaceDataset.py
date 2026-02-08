import os
import numbers
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import mxnet as mx
import numpy as np
from PIL import Image

class FaceDataset(Dataset):
    def __init__(self, root_dir, mode='train', target_size=224, num_classes=600, images_per_class=20):
        """
        Args:
            root_dir (str): 数据集根目录 (包含 train.rec, train.idx)
            mode (str): 'train'
            target_size (int): 输出图像尺寸 (ViT 建议 224)
            num_classes (int): 需要筛选的总人数 (默认 600)
            images_per_class (int): 每人抽取的图像张数 (默认 20)
        """
        super(FaceDataset, self).__init__()
        self.root_dir = root_dir
        self.target_size = target_size
        
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        
        if not os.path.exists(path_imgrec) or not os.path.exists(path_imgidx):
            raise RuntimeError(f"Dataset files not found in {root_dir}. Please check the path.")

        # 1. 加载 MXNet RecordIO
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        
        # 2. 读取 Root Header (Index 0)
        # 在 InsightFace 格式中，index 0 存储了身份索引的范围
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        
        if header.flag > 0:
            # 获取身份标识(Identity)的索引范围
            # id_start 是第一个人的索引位置，id_end 是最后一个人的位置
            self.id_start = int(header.label[0])
            self.id_end = int(header.label[1])
            print(f"🔥 Found {self.id_end - self.id_start} identities in total.")
        else:
            raise RuntimeError("The dataset is not in the standard indexed format. Cannot filter by Identity.")

        # 3. 筛选数据：600人 x 20张
        self.filtered_img_indices = []
        self.remapped_labels = []
        
        print(f"🔍 Filtering: Selecting {num_classes} IDs with at least {images_per_class} images each...")
        
        actual_class_count = 0
        for i in range(self.id_start, self.id_end):
            # 读取该身份对应的 Header，header.label 存储了该人所有图片的索引范围
            s = self.imgrec.read_idx(i)
            h, _ = mx.recordio.unpack(s)
            
            # 获取该 ID 的图片索引区间 [start, end)
            img_idx_range = np.arange(int(h.label[0]), int(h.label[1]))
            
            if len(img_idx_range) >= images_per_class:
                # 选取前 images_per_class 张图
                selected_indices = img_idx_range[:images_per_class]
                self.filtered_img_indices.extend(selected_indices)
                
                # 关键：执行标签重映射，将原始 ID 映射为 0 ~ (num_classes - 1)
                self.remapped_labels.extend([actual_class_count] * images_per_class)
                
                actual_class_count += 1
                if actual_class_count >= num_classes:
                    break
        
        if actual_class_count < num_classes:
            print(f"⚠️ Warning: Only found {actual_class_count} classes meeting the requirement.")

        print(f"✅ Final Dataset: {actual_class_count} classes, {len(self.filtered_img_indices)} total images.")

        # 4. 数据增强
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((target_size, target_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            # 使用 BLIP-2/CLIP 标准均值方差
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])

    def __getitem__(self, index):
        # 获取图片索引和重映射后的标签
        img_idx = self.filtered_img_indices[index]
        label = self.remapped_labels[index]
        
        # 读取图片数据
        s = self.imgrec.read_idx(img_idx)
        _, img = mx.recordio.unpack(s)
        
        # 解码并转换
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.filtered_img_indices)

# 测试代码
if __name__ == "__main__":
    # 填入你的数据集路径进行验证
    DATA_ROOT = "./datasets/faces_emore" 
    try:
        ds = FaceDataset(DATA_ROOT, num_classes=600, images_per_class=20)
        img, lbl = ds[0]
        print(f"Image shape: {img.shape}, Label: {lbl}")
        
        # 验证标签范围
        all_labels = np.array(ds.remapped_labels)
        print(f"Label range: {all_labels.min()} to {all_labels.max()}")
        print(f"Unique labels: {len(np.unique(all_labels))}")
    except Exception as e:
        print(f"Setup failed: {e}")
