import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import math
from tqdm import tqdm
from transformers import Blip2VisionModel, Blip2Config
import glob
# ==========================================
# 0. 基础设置与导入
# ==========================================
try:
    from FaceDataset import FaceDataset
except ImportError:
    print("Error: FaceDataset.py not found.")
    # creating a dummy dataset if not found to prevent immediate crash during import check, 
    # though it will fail at runtime if data is missing
    from torch.utils.data import Dataset
    class FaceDataset(Dataset):
        def __init__(self, root_dir, mode, target_size, max_samples):
            self.root_dir = root_dir
            self.len = 1000 # dummy
        def __len__(self): return self.len
        def __getitem__(self, idx): return torch.randn(3, 224, 224), 0
# ==========================================
# 1. 损失函数 (保持数学严密性)
# ==========================================
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super(ArcFaceLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, embedding, label):
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight))
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label[index, None], 1.0) * self.m
        cosine.acos_()
        cosine[index] += m_hot
        cosine.cos_().mul_(self.s)
        return F.cross_entropy(cosine, label)
class ATKLLoss(nn.Module):
    def __init__(self, temp=1.0):
        super(ATKLLoss, self).__init__()
        self.temp = temp
        self.kl = nn.KLDivLoss(reduction='batchmean')
    def forward(self, s_map, t_prob):
        B, N, D = s_map.shape
        H_s = W_s = int(math.sqrt(N)) 
        att_s = torch.sum(torch.pow(s_map, 2), dim=2).view(B, H_s, W_s)
        
        # Resize to Teacher Size (7x7)
        att_s = att_s.unsqueeze(1)
        att_s_resized = F.interpolate(att_s, size=(7, 7), mode='bilinear', align_corners=False)
        att_s_vec = att_s_resized.view(B, -1)
        
        log_prob_s = F.log_softmax(att_s_vec / self.temp, dim=1)
        t_prob = torch.clamp(t_prob, min=1e-8)
        return self.kl(log_prob_s, t_prob)
class RKDLoss(nn.Module):
    def __init__(self):
        super(RKDLoss, self).__init__()
        self.huber = nn.SmoothL1Loss()
    def pdist(self, e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
        if not squared: res = res.sqrt()
        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res
    def forward(self, s_emb, t_emb):
        d_s = self.pdist(s_emb)
        d_t = self.pdist(t_emb)
        d_s_norm = d_s / (d_s.mean() + 1e-8)
        d_t_norm = d_t / (d_t.mean() + 1e-8)
        return self.huber(d_s_norm, d_t_norm)
# ==========================================
# 2. 模型定义 (修复权重加载 + Adapter)
# ==========================================
class FaceAdapter(nn.Module):
    def __init__(self, embed_dim, bottleneck_dim, scale=0.1):
        super().__init__()
        self.down_proj = nn.Linear(embed_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(bottleneck_dim, embed_dim)
        self.scale = nn.Parameter(torch.tensor(scale))
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
    def forward(self, x):
        return self.scale * self.up_proj(self.act(self.down_proj(x)))
class AdapterInjectedMLP(nn.Module):
    def __init__(self, original_mlp, adapter):
        super().__init__()
        self.original_mlp = original_mlp
        self.adapter = adapter
    def forward(self, hidden_states):
        return self.original_mlp(hidden_states) + self.adapter(hidden_states)
class StudentViT(nn.Module):
    def __init__(self, blip_path, num_classes):
        super().__init__()
        print(f"Loading BLIP-2 Vision Config from: {blip_path}")
        
        # 1. 加载配置
        config = Blip2Config.from_pretrained(blip_path)
        
        # 2. 初始化空模型 (随机权重)
        self.vit = Blip2VisionModel(config.vision_config)
        
        # 3. 智能加载权重 (支持分片和单文件)
        self._load_clean_weights(blip_path)
        # 4. 冻结主干
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # 5. 均匀插入 Adapter
        layers = self.vit.encoder.layers
        total_layers = len(layers) # 通常为 39
        num_inject = 12
        insert_indices = np.linspace(0, total_layers - 1, num_inject, dtype=int)
        
        print(f"Injecting Adapters into layers: {insert_indices}")
        
        self.trainable_params = []
        adapter_dim = config.vision_config.hidden_size
        bottleneck_dim = adapter_dim // 4
        
        for i in insert_indices:
            adapter = FaceAdapter(adapter_dim, bottleneck_dim)
            original_mlp = layers[i].mlp
            layers[i].mlp = AdapterInjectedMLP(original_mlp, adapter)
            self.trainable_params.extend(list(adapter.parameters()))
            
        # 6. 分类头
        self.head = ArcFaceLoss(adapter_dim, num_classes)
        self.trainable_params.extend(list(self.head.parameters()))
    def _load_clean_weights(self, blip_path):
        """
        鲁棒的权重加载函数：
        1. 自动搜索分片权重 (*.bin 或 *.safetensors)
        2. 自动提取 vision_model. 开头的参数
        3. 自动去掉前缀以匹配 standalone vision model
        """
        print(f"Scanning {blip_path} for weights...")
        
        # 搜索所有权重文件
        bin_files = glob.glob(os.path.join(blip_path, "*.bin"))
        safe_files = glob.glob(os.path.join(blip_path, "*.safetensors"))
        if len(safe_files) > 0:
            print("Found .safetensors files. Using them (faster & safer).")
            all_files = safe_files
        else:
            bin_files = glob.glob(os.path.join(blip_path, "*.bin"))
            print("Found .bin files. Using them.")
            all_files = bin_files
        
        if len(all_files) == 0:
             raise FileNotFoundError(f"No .bin or .safetensors files found in {blip_path}")
             
        print(f"Found {len(all_files)} weight files. Extracting Vision Encoder...")
        
        vision_dict = {}
        prefix = "vision_model."
        
        # 遍历所有文件提取参数
        for file_path in all_files:
            # 跳过索引文件
            if "index" in file_path: continue 
            
            # 加载单个文件到 CPU
            try:
                if file_path.endswith(".safetensors"):
                    from safetensors.torch import load_file
                    state_dict = load_file(file_path, device="cpu")
                else:
                    state_dict = torch.load(file_path, map_location="cpu")
            except Exception as e:
                print(f"Skipping {os.path.basename(file_path)}: {e}")
                continue
                
            # 筛选并清洗 Key
            keys_added = 0
            for key, value in state_dict.items():
                if key.startswith(prefix):
                    # 去掉 'vision_model.' 前缀
                    new_key = key[len(prefix):] 
                    vision_dict[new_key] = value
                    keys_added += 1
            
            if keys_added > 0:
                print(f"  - Extracted {keys_added} vision keys from {os.path.basename(file_path)}")
                
        if len(vision_dict) == 0:
            raise RuntimeError("Found weight files but NO keys starting with 'vision_model.' were found. Check if this is a BLIP-2 checkpoint.")
        # 加载进模型
        missing, unexpected = self.vit.load_state_dict(vision_dict, strict=False)
        
        # 验证加载结果
        real_missing = [k for k in missing if "position_ids" not in k]
        if len(real_missing) > 0:
            print(f"⚠️ Warning: Missing keys: {real_missing[:5]}... (Total {len(real_missing)})")
        else:
            print("✅ Successfully loaded and cleaned Vision Weights!")
    def forward(self, x, label=None):
        outputs = self.vit(pixel_values=x, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state 
        cls_emb = last_hidden_state[:, 0, :] 
        patch_tokens = last_hidden_state[:, 1:, :] 
        
        loss = None
        if label is not None:
            loss = self.head(cls_emb, label)
        return cls_emb, patch_tokens, loss
# ==========================================
# 3. 数据集 Wrapper
# ==========================================
class DatasetWithIndex(FaceDataset):
    def __getitem__(self, index):
        data, label = super().__getitem__(index)
        return data, label, index
# ==========================================
# 4. 主训练流程
# ==========================================
if __name__ == "__main__":
    # --- 配置 ---
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    
    # 默认路径 (相对于项目根目录)
    # 请确保 download_weight.py 和数据集解压位置符合此结构，或在此处修改为实际绝对路径
    BLIP_PATH = os.path.join(PROJECT_ROOT, "blip2_weights")
    DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets", "faces_emore")
    TEACHER_FEAT_DIR = os.path.join(PROJECT_ROOT, "teacher_features_12k")
    LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "adapter_face_distill")
    
    # 修正：批量大小保持 128 (是调试请自行修改)
    BATCH_SIZE = 128 
    LR = 5e-4 
    EPOCHS = 100
    LAMBDA_AT = 1000.0
    LAMBDA_RKD = 1.0
    
    # --- 显卡设置 (核心修正) ---
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    # 2. PyTorch 会将可见显卡重新编号为 0, 1, ...
    # 自动获取当前可见的显卡数量并生成索引列表
    DEVICE_IDS = [i for i in range(torch.cuda.device_count())] 
    
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using GPUs (Logical IDs): {DEVICE_IDS} (Mapped from CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR)
    print(f"TensorBoard logging to: {LOG_DIR}")
    # 1. Load Teacher
    print("Loading Teacher Features...")
    if not os.path.exists(TEACHER_FEAT_DIR):
        print(f"Error: {TEACHER_FEAT_DIR} not found.")
        # exit() 
        
    # Mock data if file missing for robust script execution check
    if os.path.exists(os.path.join(TEACHER_FEAT_DIR, "teacher_embedding.npy")):
        t_embeddings = torch.from_numpy(np.load(os.path.join(TEACHER_FEAT_DIR, "teacher_embedding.npy"))).float()
        t_feat_maps = torch.from_numpy(np.load(os.path.join(TEACHER_FEAT_DIR, "teacher_feat_map.npy"))).float()
    else:
         print("Warning: Teacher features not found, creating dummy tensors.")
         t_embeddings = torch.randn(1000, 768)
         t_feat_maps = torch.randn(1000, 49, 768)
    # 2. Dataset
    MAX_SAMPLES = len(t_embeddings)
    try:
        dataset = DatasetWithIndex(root_dir=DATA_ROOT, mode='train', target_size=224, max_samples=MAX_SAMPLES)
        # num_workers suggestion: 4 * num_gpus = 16 or just keep 8
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True)
    except Exception as e:
        print(f"Dataset init failed: {e}")
        exit()
    
    NUM_CLASSES = 85742 
    
    # 3. Model
    student = StudentViT(BLIP_PATH, NUM_CLASSES)
    
    # Enable Multi-CPU
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs: {DEVICE_IDS}")
        # 注意：这里传入的是逻辑 ID (如 [0, 1])
        student = nn.DataParallel(student, device_ids=DEVICE_IDS)
    
    student = student.to(DEVICE)
    
    if isinstance(student, nn.DataParallel):
        params_to_optimize = student.module.trainable_params
    else:
        params_to_optimize = student.trainable_params
    optimizer = optim.AdamW(params_to_optimize, lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    loss_at_fn = ATKLLoss(temp=1.0).to(DEVICE)
    loss_rkd_fn = RKDLoss().to(DEVICE)
    
    print("Start Training...")
    global_step = 0
    
    # --- Early Stopping Variables ---
    best_loss = float('inf')
    early_stop_counter = 0
    PATIENCE_LIMIT = 5
    
    for epoch in range(EPOCHS):
        student.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        # Track epoch loss for early stopping
        epoch_loss_accum = 0.0
        num_batches = 0
        
        for imgs, labels, indices in pbar:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            batch_t_emb = t_embeddings[indices].to(DEVICE)
            batch_t_map = t_feat_maps[indices].to(DEVICE)
            
            s_emb, s_tokens, l_arc = student(imgs, labels)
            l_arc = l_arc.mean()
            
            l_at = loss_at_fn(s_tokens, batch_t_map)
            l_rkd = loss_rkd_fn(s_emb, batch_t_emb)
            
            if epoch < 1:
                loss = l_arc
            else:
                loss = l_arc + LAMBDA_AT * l_at + LAMBDA_RKD * l_rkd
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # --- Logging ---
            global_step += 1
            writer.add_scalar('Loss/Total', loss.item(), global_step)
            writer.add_scalar('Loss/ArcFace', l_arc.item(), global_step)
            writer.add_scalar('Loss/AT_KL', l_at.item(), global_step)
            writer.add_scalar('Loss/RKD', l_rkd.item(), global_step)
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step)
            
            epoch_loss_accum += loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                "Total": f"{loss.item():.3f}", 
                "Arc": f"{l_arc.item():.3f}",
                "AT": f"{l_at.item():.4f}",
                "RKD": f"{l_rkd.item():.4f}"
            })
        
        # Calculate Average Loss for the Epoch
        avg_epoch_loss = epoch_loss_accum / num_batches if num_batches > 0 else float('inf')
        print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
        # --- Early Stopping Check ---
        if avg_epoch_loss < best_loss:
            print(f"Loss Improved ({best_loss:.4f} -> {avg_epoch_loss:.4f}). Saving Best Model...")
            best_loss = avg_epoch_loss
            early_stop_counter = 0
            
            # Save the Best Model
            save_path = f"checkpoints/best_student_adapter_face.pth"
            os.makedirs("checkpoints", exist_ok=True)
            state_dict = student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict()
            torch.save(state_dict, save_path)
            
        else:
            early_stop_counter += 1
            print(f"No Improvement. Early Stopping Counter: {early_stop_counter}/{PATIENCE_LIMIT}")
        
        if early_stop_counter >= PATIENCE_LIMIT:
            print(f"Early Stopping Triggered after {epoch+1} epochs.")
            break
        scheduler.step()
        
        # Regular checkpoint
        if (epoch + 1) % 5 == 0:
            save_path = f"checkpoints/student_adapter_face_epoch_{epoch+1}.pth"
            state_dict = student.module.state_dict() if isinstance(student, nn.DataParallel) else student.state_dict()
            torch.save(state_dict, save_path)
    writer.close()
    print("Training Finished.")
