import numpy as np
from scipy.linalg import solve
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn

class DeformationField(nn.Module):
    def __init__(self, D, num_control_points=256):
        super().__init__()
        # 全局刚性参数
        self.rotation = nn.Parameter(torch.eye(D))
        self.translation = nn.Parameter(torch.zeros(D))
        
        # 局部非刚性形变（基于TPS）
        self.control_points = nn.Parameter(torch.randn(num_control_points, D))
        self.weights = nn.Parameter(torch.zeros(num_control_points, D))

    def tps_basis(self, x):
        """ 计算TPS基函数 """
        dist = torch.cdist(x, self.control_points)
        return torch.sqrt(dist**2 + 1e-16)  # 避免除零

    def forward(self, x):
        # 全局变换
        x_global = torch.mm(x, self.rotation.T) + self.translation
        
        # 局部形变
        basis = self.tps_basis(x_global)
        displacement = torch.mm(basis, self.weights)
        
        return x_global + displacement


def cpd_loss(source, target, source_feat, target_feat, deformation, sigma_sq, lambda_feat=0.5, gamma=0.1):
    # 形变后源点云
    warped_source = deformation(source)
    
    # 计算坐标和特征距离
    coord_dist = torch.cdist(warped_source, target)
    feat_dist = torch.cdist(source_feat, target_feat)
    joint_dist = coord_dist**2 + lambda_feat * feat_dist**2
    
    # GMM似然项
    exp_term = torch.exp(-joint_dist / (2 * sigma_sq))
    likelihood = torch.log(torch.sum(exp_term, dim=1) + 1e-6)
    energy = -torch.mean(likelihood)
    
    # 正则化项
    R_global = torch.norm(deformation.rotation.T @ deformation.rotation - torch.eye(3)) + \
               torch.norm(deformation.translation)
    R_local = torch.mean(torch.autograd.grad(deformation.weights.sum(), deformation.control_points, retain_graph=True)[0]**2)
    
    total_loss = energy + gamma * (R_global + R_local)
    return total_loss

def multi_scale_cpd(source, target, encoder, scales=[0.1, 0.5, 1.0]):
    device = source.device
    sigma_sq = torch.tensor(1.0, device=device)
    
    for scale in scales:
        # 下采样
        idx_src = torch.randperm(source.size(0))[:int(scale*source.size(0))]
        idx_tgt = torch.randperm(target.size(0))[:int(scale*target.size(0))]
        src_sub = source[idx_src]
        tgt_sub = target[idx_tgt]
        
        # 提取特征
        with torch.no_grad():
            feat_src = encoder(src_sub)
            feat_tgt = encoder(tgt_sub)
        
        # 初始化形变场
        deform_net = DeformationField().to(device)
        optimizer = torch.optim.Adam(deform_net.parameters(), lr=0.01)
        
        # 迭代优化
        for epoch in range(100):
            optimizer.zero_grad()
            loss = cpd_loss(src_sub, tgt_sub, feat_src, feat_tgt, deform_net, sigma_sq)
            loss.backward()
            optimizer.step()
            sigma_sq.data *= 0.95  # 退火噪声方差
    
    return deform_net

class zstack_reconstraction:
    def __init__(self, slices, features, lambda1=1.0, lambda2=0.1, alpha=0.5, max_iter=100):
        self.slices = slices          # List of N (M_k, 3) arrays
        self.features = features      # List of N (M_k, d) arrays
        self.lambda1 = lambda1        # 局部平滑权重
        self.lambda2 = lambda2        # 矩相干权重
        self.alpha = alpha            # 特征权重
        self.max_iter = max_iter
        self.xp = torch
        self.w  = 0.1

    def gaussian_kernel(self, X, beta=None):
        N, D = X.shape
        Dist = self.xp.cdist(X, X, p=2)
        Dist.pow_(2)
        if beta is None:
            beta = self.xp.sum(Dist) / (D*N*N)
        Dist.div_(-2 * beta**2)
        Dist.exp_()
        return Dist

    def expectation(self, X, Y):
        P = self.update_P(X, Y)
        self.Pt1 = self.xp.sum(P, 0).to_dense()
        self.P1 = self.xp.sum(P, 1).to_dense()
        self.Np = self.xp.sum(self.P1)
        self.PX = P @ self.X

    def update_P(self, X, Y, sigma2):
        N, D =X.shape
        M, D = Y.shape

        gs = self.xp.log(M/N*self.w/(1. - self.w))

        P = self.xp.cdist(Y, X, p=2)
        P.pow_(2)
        P.div_(-2*self.sigma2)

        P.exp_()
        cs = 0.5*(
            D*self.xp.log(2*self.xp.pi*sigma2)
        )
        cs = self.xp.exp(cs+gs)

        cdfs = self.xp.sum(P, 0).to_dense()
        cdfs.masked_fill_(cdfs == 0, self.eps)
        # P.div_(cdfs+cs)
        return P

    def register(self):
        N = len(self.slices)
        M = [s.shape[0] for s in self.slices]
        D = self.slices[0].shape[1]  # 3D
        G = [ self.gaussian_kernel(self.slices[k]) for i in range(N) ]
        R = [ self.xp.eye(D) for i in range(N) ]
        T = [ np.zeros(D) for i in range(N) ]
        W = [np.zeros((m, D)) for m in M]
        sigma2 = 1.0

        for iter in range(self.max_iter):
            # E步：计算对应概率
            P = []
            for k in range(N):
                X = self.slices[k]
                
                P = self.update_P(X, X, sigma2)
            P = self.array(P) / np.sum(P, axis=1, keepdims=True)

            # M步：更新W
            for k in range(N):
                G = self.gaussian_kernel(self.slices[k])
                # 构建线性系统
                A = G + self.lambda1 * np.eye(M[k])
                b = np.zeros((M[k], D))
                for n in range(M[k]):
                    b[n] = np.sum(P[n] * (self.target_points - self.slices[k][n]), axis=0)
                W[k] = solve(A, b)

                # 应用矩相干约束
                if k > 0:
                    mu_prev = np.mean(W[k-1], axis=0)
                    mu_curr = np.mean(W[k], axis=0)
                    W[k] += self.lambda2 * (mu_prev - mu_curr)

        return W

# # 使用示例
# slices = [np.random.rand(100, 3) for _ in range(5)]  # 5张切片
# features = [np.random.rand(100, 10) for _ in range(5)]
# cpd = zstack_reconstraction(slices, features, lambda1=1.0, lambda2=0.1)
# W = cpd.register()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==============================
# 1. 分片参数化模型
# ==============================
class SliceWiseDeformation(nn.Module):
    def __init__(self, num_slices=10, num_control_points=64):
        super().__init__()
        self.num_slices = num_slices
        
        # 初始化每片的刚性变换参数
        self.rotations = nn.Parameter(torch.stack([torch.eye(3) for _ in range(num_slices)]))
        self.translations = nn.Parameter(torch.zeros(num_slices, 3))
        
        # 初始化每片的非刚性形变参数
        self.control_points = nn.ParameterList([
            nn.Parameter(torch.randn(num_control_points, 3)) 

            for _ in range(num_slices)
        ])
        self.weights = nn.ParameterList([
            nn.Parameter(torch.zeros(num_control_points, 3))
            for _ in range(num_slices)
        ])
    
    def gaussian_kernel(self, x, control_points, beta=1.0):
        pairwise_dist = torch.cdist(x, control_points)
        return torch.exp(-beta * pairwise_dist**2)
    
    def forward(self, x, slice_idx):
        """ 处理指定切片的数据 """
        # 全局刚性变换
        R = self.rotations[slice_idx]
        t = self.translations[slice_idx]
        x_global = x @ R.T + t
        
        # 局部非刚性形变
        basis = self.gaussian_kernel(x_global, self.control_points[slice_idx])
        displacement = basis @ self.weights[slice_idx]
        
        return x_global + displacement

# ==============================
# 2. 平滑约束损失函数
# ==============================
class SmoothnessLoss(nn.Module):
    def __init__(self, lambda_rot=0.1, lambda_trans=0.1, lambda_deform=0.2):
        super().__init__()
        self.lambda_rot = lambda_rot
        self.lambda_trans = lambda_trans
        self.lambda_deform = lambda_deform
    
    def forward(self, model):
        # 旋转平滑项
        rot_diff = torch.stack([torch.norm(model.rotations[i] - model.rotations[i+1], p='fro') 
                              for i in range(model.num_slices-1)])
        rot_loss = torch.mean(rot_diff)
        
        # 平移平滑项
        trans_diff = torch.stack([torch.norm(model.translations[i] - model.translations[i+1])
                                for i in range(model.num_slices-1)])
        trans_loss = torch.mean(trans_diff)

        # 形变平滑项（二阶微分近似）
        deform_loss = 0
        for i in range(model.num_slices):
            w = model.weights[i]
            cp = model.control_points[i]
            # 计算拉普拉斯算子
            grad_w = torch.autograd.grad(w.sum(), cp, create_graph=True)[0]
            laplacian = torch.autograd.grad(grad_w.sum(), cp, retain_graph=True)[0]
            deform_loss += torch.mean(laplacian**2)
        
        total_loss = (self.lambda_rot * rot_loss + 
                     self.lambda_trans * trans_loss + 
                     self.lambda_deform * deform_loss)
        return total_loss

# ==============================
# 3. 大数据加载器
# ==============================
class ZStackDataset(Dataset):
    def __init__(self, num_slices=100, points_per_slice=1000):
        self.data = [torch.randn(points_per_slice, 3) for _ in range(num_slices)]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], idx  # 返回点云和切片索引

# ==============================
# 4. 分布式训练引擎
# ==============================
def train_large_scale():
    # 配置参数
    num_slices = 1000
    batch_size = 32
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化模型和损失
    model = SliceWiseDeformation(num_slices).to(device)
    criterion = nn.MSELoss()
    smoothness_criterion = SmoothnessLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # 数据加载
    dataset = ZStackDataset(num_slices=num_slices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=num_workers, pin_memory=True)
    
    # 训练循环
    for epoch in range(100):
        total_loss = 0
        for batch_points, batch_indices in dataloader:
            batch_points = batch_points.to(device)
            optimizer.zero_grad()
            
            # 前向计算
            losses = []
            for points, idx in zip(batch_points, batch_indices):
                warped = model(points, idx)
                # 假设target预先加载到显存
                target = ...  # 根据实际数据获取
                loss = criterion(warped, target)
                losses.append(loss)
            
            # 合并损失
            data_loss = torch.mean(torch.stack(losses))
            smooth_loss = smoothness_criterion(model)
            total_batch_loss = data_loss + smooth_loss
            
            # 反向传播
            total_batch_loss.backward()
            optimizer.step()
            total_loss += total_batch_loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

# ==============================
# 5. 可视化与验证
# ==============================
def validate_alignment(model, validation_loader):
    model.eval()
    with torch.no_grad():
        for points, idx in validation_loader:
            points = points.to(device)
            aligned = model(points, idx)
            # 计算RMSE
            target = ...
            rmse = torch.sqrt(torch.mean((aligned - target)**2))
            print(f"Slice {idx} RMSE: {rmse:.4f}")


smoothness_criterion = SmoothnessLoss(
    lambda_rot=0.5,   # 控制旋转平滑
    lambda_trans=0.3,  # 控制平移平滑 
    lambda_deform=1.0  # 控制形变平滑
)

# # 使用相邻切片初始化提升收敛速度
# for i in range(1, num_slices):
#     model.rotations.data[i] = model.rotations[i-1].clone()
#     model.translations.data[i] = model.translations[i-1].clone()

# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
# # 跨切片共享控制点网格
# self.shared_control_points = nn.Parameter(torch.randn(K, 3))

# from torch.cuda.amp import autocast, GradScaler
# scaler = GradScaler()
# with autocast():
#     warped = model(points, idx)
# from torch.utils.checkpoint import checkpoint
# warped = checkpoint(model, points, idx)  # 减少显存占用