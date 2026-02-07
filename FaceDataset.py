import os
import numbers
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import mxnet as mx
import numpy as np
from PIL import Image
class FaceDataset(Dataset):
    def __init__(self, root_dir, mode='train', target_size=112, max_samples=None):
        """
        Args:
            root_dir (str): 数据集根目录
            mode (str): 'train'
            target_size (int): 
                - 如果是 iResNet，必须设为 112
                - 如果是 ViT，通常设为 224
            max_samples (int): 强制只使用前 N 张图片。
        """
        super(FaceDataset, self).__init__()
        self.root_dir = root_dir
        self.target_size = target_size
        
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        if not os.path.exists(path_imgrec) or not os.path.exists(path_imgidx):
            raise RuntimeError(f"Dataset files not found in {root_dir}")
        # 1. 加载 MXNet RecordIO
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        # 2. 读取 Header
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        
        # 3. 获取所有图片索引
        if header.flag > 0:
            max_idx = int(header.label[0])
            self.imgidx = np.array(range(1, max_idx))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
        # 4. [Lite Mode] 强制截断数据量
        if max_samples is not None and max_samples < len(self.imgidx):
            self.imgidx = self.imgidx[:max_samples]
            print(f"⚡ [Lite Mode] Dataset truncated to first {max_samples} images only.")
        else:
            print(f"📚 [Full Mode] Using all {len(self.imgidx)} images.")
        # 5. 数据增强
        # 注意：这里我们根据 target_size 动态调整 Resize
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((target_size, target_size), interpolation=Image.BICUBIC),
            # transforms.RandomHorizontalFlip(p=0.5), # 提取教师特征建议关闭翻转，保持确定性
            transforms.ToTensor(),
            # 改为 BLIP-2/CLIP 标准均值方差
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])
    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
            
        return sample, label
    def __len__(self):
        return len(self.imgidx)
