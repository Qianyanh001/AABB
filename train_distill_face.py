# -*- coding: utf-8 -*-
import os

# ==========================================================
# 0. 环境与 GPU 配置
# ==========================================================
GPU_LIST = os.environ.get("GPU_LIST", "0,1,2,3") 
os.environ["CUDA_VISIBLE_DEVICES"] = GPU_LIST
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import Blip2Config, Blip2VisionModel

# ==========================================
# 1. 基础设置与导入
# ==========================================
try:
    from FaceDataset import FaceDataset
except ImportError:
    raise ImportError("❌ FaceDataset.py not found.")

# ==========================================
# 2. 损失函数定义
# ==========================================
class ArcFaceLoss(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embedding, label):
        cosine = F.linear(F.normalize(embedding), F.normalize(self.weight))
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = cosine.acos()
        m_hot = torch.zeros_like(cosine)
        m_hot.scatter_(1, label.view(-1, 1), self.m)
        target_logit = (theta + m_hot).cos()
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        output = (one_hot * target_logit) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return F.cross_entropy(output, label)

class ATKLLoss(nn.Module):
    def __init__(self, temp=1.0):
        super().__init__()
        self.temp = temp
        self.kl = nn.KLDivLoss(reduction="batchmean")

    def forward(self, s_map, t_prob):
        bsz, n, _ = s_map.shape
        hw = int(math.sqrt(n))
        att_s = torch.sum(torch.pow(s_map, 2), dim=2).view(bsz, hw, hw).unsqueeze(1)
        att_s_resized = F.interpolate(att_s, size=(7, 7), mode="bilinear", align_corners=False)
        att_s_vec = att_s_resized.view(bsz, -1)
        log_prob_s = F.log_softmax(att_s_vec / self.temp, dim=1)
        return self.kl(log_prob_s, t_prob)

class RKDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.SmoothL1Loss()

    @staticmethod
    def pdist(e, eps=1e-12):
        e_square = (e ** 2).sum(dim=1, keepdim=True)
        dist2 = e_square + e_square.t() - 2.0 * (e @ e.t())
        dist = torch.sqrt(torch.clamp(dist2, min=eps))
        bsz = e.size(0)
        eye = torch.eye(bsz, device=e.device, dtype=torch.bool)
        dist = dist.masked_fill(eye, 0.0)
        return dist

    def forward(self, s_emb, t_emb):
        d_s = self.pdist(s_emb)
        d_t = self.pdist(t_emb)
        d_s_norm = d_s / (d_s.mean() + 1e-8)
        d_t_norm = d_t / (d_t.mean() + 1e-8)
        return self.huber(d_s_norm, d_t_norm)

# ==========================================
# 3. 模型定义
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
        config = Blip2Config.from_pretrained(blip_path)
        self.vit = Blip2VisionModel(config.vision_config)
        self._load_clean_weights(blip_path)
        for p in self.vit.parameters(): p.requires_grad = False
        layers = self.vit.encoder.layers
        insert_indices = np.linspace(0, len(layers) - 1, 12, dtype=int)
        self.trainable_params = []
        adapter_dim = config.vision_config.hidden_size
        bottleneck_dim = adapter_dim // 4
        for i in insert_indices:
            adapter = FaceAdapter(adapter_dim, bottleneck_dim)
            layers[i].mlp = AdapterInjectedMLP(layers[i].mlp, adapter)
            self.trainable_params.extend(list(adapter.parameters()))
        self.head = ArcFaceLoss(adapter_dim, num_classes)
        self.trainable_params.extend(list(self.head.parameters()))

    def _load_clean_weights(self, blip_path):
        all_files = glob.glob(os.path.join(blip_path, "*.safetensors")) + glob.glob(os.path.join(blip_path, "*.bin"))
        vision_dict = {}
        for f in all_files:
            if "index" in f: continue
            sd = torch.load(f, map_location="cpu") if f.endswith(".bin") else None
            if f.endswith(".safetensors"):
                from safetensors.torch import load_file
                sd = load_file(f, device="cpu")
            for k, v in sd.items():
                if k.startswith("vision_model."): vision_dict[k[len("vision_model."):]] = v
        self.vit.load_state_dict(vision_dict, strict=False)

    def forward(self, x, label=None):
        outputs = self.vit(pixel_values=x, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state
        cls_emb = last_hidden_state[:, 0, :]
        patch_tokens = last_hidden_state[:, 1:, :]
        loss = self.head(cls_emb, label) if label is not None else None
        return cls_emb, patch_tokens, loss

class DatasetWithIndex(FaceDataset):
    def __getitem__(self, index):
        data, label = super().__getitem__(index)
        return data, label, index

# ==========================================
# 4. 主训练流程
# ==========================================
if __name__ == "__main__":
    assert torch.cuda.is_available(), "❌ CUDA 不可用"

    logical_gpu_count = torch.cuda.device_count()
    device = torch.device("cuda:0")
    device_ids = list(range(logical_gpu_count))
    use_dp = logical_gpu_count > 1

    project_root = os.path.dirname(os.path.abspath(__file__))
    blip_path = os.path.join(project_root, "blip2_weights")
    data_root = "./datasets/faces_emore/"
    teacher_feat_dir = os.path.join(project_root, "teacher_features_12k")
    log_dir = os.path.join(project_root, "logs", "adapter_distill")
    
    per_gpu_batch = 128
    batch_size = per_gpu_batch * max(1, logical_gpu_count)
    lr = 1e-4
    epochs = 100
    num_classes = 600

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    print("⏳ Loading Teacher Features...")
    t_embeddings = torch.from_numpy(np.load(os.path.join(teacher_feat_dir, "teacher_embedding.npy"))).float()
    t_feat_maps = torch.from_numpy(np.load(os.path.join(teacher_feat_dir, "teacher_feat_map.npy"))).float()

    dataset = DatasetWithIndex(root_dir=data_root, num_classes=600, images_per_class=20, target_size=224)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

    student = StudentViT(blip_path, num_classes)
    if use_dp: student = nn.DataParallel(student, device_ids=device_ids)
    student = student.to(device)

    train_params = student.module.trainable_params if use_dp else student.trainable_params
    optimizer = optim.AdamW(train_params, lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_at_fn = ATKLLoss(temp=1.0).to(device)
    loss_rkd_fn = RKDLoss().to(device)
    scaler = torch.cuda.amp.GradScaler()

    best_loss = float('inf')
    global_step = 0

    print(f"🚀 启动训练 | BatchSize: {batch_size} | 显卡: {GPU_LIST}")

    for epoch in range(epochs):
        student.train()
        
        # --- 关键修改：增加子损失的累加器 ---
        epoch_loss_total = 0.0
        epoch_loss_arc = 0.0
        epoch_loss_at = 0.0
        epoch_loss_rkd = 0.0
        step_count = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        curr_lambda_at = min(1.0, 0.1 * (epoch + 1))
        curr_lambda_rkd = min(1000.0, 100 * (epoch + 1))

        for imgs, labels, indices in pbar:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            batch_t_emb = t_embeddings[indices].to(device, non_blocking=True)
            batch_t_map = t_feat_maps[indices].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                s_emb, s_tokens, l_arc = student(imgs, labels)
                if use_dp: l_arc = l_arc.mean()
                
                # 蒸馏损失
                l_at = loss_at_fn(s_tokens.float(), batch_t_map.float())
                l_rkd = loss_rkd_fn(s_emb.float(), batch_t_emb.float())
                
                # 总损失
                loss = l_arc + curr_lambda_at * l_at + curr_lambda_rkd * l_rkd

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # --- 累加各项损失 ---
            epoch_loss_total += loss.item()
            epoch_loss_arc += l_arc.item()
            epoch_loss_at += l_at.item()
            epoch_loss_rkd += l_rkd.item()
            step_count += 1
            global_step += 1

            pbar.set_postfix({
                "Loss": f"{loss.item():.2f}",
                "Arc": f"{l_arc.item():.2f}",
                "AT": f"{l_at.item():.4f}"
            })

        # --- Epoch 结束：输出详细的平均损失 ---
        avg_loss = epoch_loss_total / step_count
        avg_arc = epoch_loss_arc / step_count
        avg_at = epoch_loss_at / step_count
        avg_rkd = epoch_loss_rkd / step_count
        
        print(f"📊 Epoch {epoch+1} Summary:")
        print(f"   >> Avg Total Loss: {avg_loss:.4f}")
        print(f"   >> Avg ArcFace Loss: {avg_arc:.4f}")
        print(f"   >> Avg AT Loss: {avg_at:.4f} (scaled: {avg_at * curr_lambda_at:.4f})")
        print(f"   >> Avg RKD Loss: {avg_rkd:.4f} (scaled: {avg_rkd * curr_lambda_rkd:.4f})")

        # 保存最优模型
        if avg_loss < best_loss:
            print(f"⭐ New Best! ({best_loss:.4f} -> {avg_loss:.4f}). Saving...")
            best_loss = avg_loss
            ckpt_path = "checkpoints/best_student_adapter.pth"
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': student.module.state_dict() if use_dp else student.state_dict(),
                'loss': best_loss,
                'sub_losses': {'arc': avg_arc, 'at': avg_at, 'rkd': avg_rkd}
            }
            torch.save(save_dict, ckpt_path)

        scheduler.step()
        torch.cuda.empty_cache()

    writer.close()
