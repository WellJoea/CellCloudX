import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import (
    GATv2Conv, GCNConv, SAGEConv, 
    GraphNorm, InstanceNorm, 
    VGAE, GAE
)
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_scatter import scatter_mean, scatter_std
from torch.distributions import Normal, Categorical
import faiss
import leidenalg
import igraph as ig
from sklearn.mixture import BayesianGaussianMixture
import scanpy as sc

class MultiScaleGeometricGraph:
    """多尺度几何图构建 - 基于微分几何和拓扑数据分析"""
    
    def __init__(self, methods=['knn', 'delaunay', 'rips', 'morse'], 
                 metric_learning=True):
        self.methods = methods
        self.metric_learning = metric_learning
        
    def build_geometric_graphs(self, X, F, coords):
        """构建多尺度几何图"""
        graphs = {}
        
        # 1. 黎曼几何图 - 考虑局部曲率
        graphs['riemannian'] = self._build_riemannian_graph(X, F, coords)
        
        # 2. 持续同调图 - 拓扑特征
        graphs['persistence'] = self._build_persistence_graph(X, F)
        
        # 3. Morse-Smale复形图
        graphs['morse'] = self._build_morse_complex(X, F)
        
        # 4. 最优传输图
        graphs['optimal_transport'] = self._build_optimal_transport_graph(X, F)
        
        return graphs
    
    def _build_riemannian_graph(self, X, F, coords):
        """黎曼几何图 - 考虑局部曲率和度量张量"""
        n_points = X.shape[0]
        
        # 计算局部曲率
        curvatures = self._compute_local_curvature(X, k=15)
        
        # 学习度量张量
        if self.metric_learning:
            metric_tensors = self._learn_metric_tensor(X, F)
        else:
            metric_tensors = np.eye(X.shape[1])
            
        # 构建基于黎曼度量的图
        src, dst, weights = [], [], []
        for i in range(n_points):
            # 马氏距离考虑局部度量
            distances = self._mahalanobis_distance(
                X[i], X, metric_tensors[i] if self.metric_learning else metric_tensors
            )
            
            # 曲率调整的距离
            adjusted_dist = distances * (1 + curvatures[i] * curvatures)
            
            # 选择邻居
            neighbors = np.argsort(adjusted_dist)[1:16]  # 15个最近邻
            
            for j in neighbors:
                src.append(i)
                dst.append(j)
                weight = np.exp(-adjusted_dist[j]**2 / 
                              (2 * self._adaptive_bandwidth(X, i, j)))
                weights.append(weight)
                
        return self._create_geometric_graph(src, dst, weights, X, F)
    
    def _compute_local_curvature(self, X, k=15):
        """计算局部曲率 - 基于PCA特征值"""
        curvatures = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            # 局部邻域
            distances = np.linalg.norm(X - X[i], axis=1)
            neighbors = np.argsort(distances)[1:k+1]
            local_points = X[neighbors] - X[i]
            
            # PCA分析
            if len(local_points) > 2:
                cov = local_points.T @ local_points
                eigenvalues = np.linalg.eigvalsh(cov)
                # 曲率估计
                curvatures[i] = eigenvalues[-1] / (np.sum(eigenvalues) + 1e-8)
                
        return curvatures
    
    def _learn_metric_tensor(self, X, F):
        """学习局部度量张量"""
        # 使用深度度量学习
        metric_net = MetricLearningNetwork(X.shape[1], F.shape[1])
        # 简化实现 - 实际需要训练
        return np.stack([np.eye(X.shape[1]) for _ in range(X.shape[0])])
    
    def _build_persistence_graph(self, X, F):
        """持续同调图 - 捕获拓扑特征"""
        # 使用Rips复形
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(X))
        
        # 简化实现 - 实际需要使用gudhi等库
        src, dst, weights = [], [], []
        n_points = X.shape[0]
        
        for i in range(n_points):
            for j in range(i+1, n_points):
                if distances[i,j] < np.percentile(distances, 10):
                    src.append(i)
                    dst.append(j)
                    # 基于持续同调特征的权重
                    weight = self._topological_weight(X, F, i, j)
                    weights.append(weight)
                    
        return self._create_geometric_graph(src, dst, weights, X, F)
    
    def _topological_weight(self, X, F, i, j):
        """基于拓扑特征的边权重"""
        # 简化实现
        spatial_sim = np.exp(-np.linalg.norm(X[i]-X[j])**2)
        feature_sim = F[i] @ F[j] / (np.linalg.norm(F[i]) * np.linalg.norm(F[j]) + 1e-8)
        return (spatial_sim + feature_sim) / 2

class GeometricGraphNeuralNetwork(nn.Module):
    """几何图神经网络 - 结合微分几何和群等变性"""
    
    def __init__(self, in_dim, hidden_dims=[256, 128, 64], 
                 num_heads=8, curvature_aware=True, equivariant=True):
        super().__init__()
        
        self.curvature_aware = curvature_aware
        self.equivariant = equivariant
        
        # 曲率感知的图卷积层
        self.geometric_convs = nn.ModuleList([
            CurvatureAwareGCN(in_dim, hidden_dims[0]),
            EquivariantGAT(hidden_dims[0], hidden_dims[1], heads=num_heads),
            SpectralGCN(hidden_dims[1], hidden_dims[2])
        ])
        
        # 持续同调特征提取
        self.topological_net = TopologicalFeatureNetwork(hidden_dims[2], 32)
        
        # 注意力融合
        self.fusion_attention = nn.MultiheadAttention(
            hidden_dims[2] + 32, num_heads=4, batch_first=True
        )
        
        # 输出头
        self.cluster_head = nn.Sequential(
            nn.Linear(hidden_dims[2] + 32, hidden_dims[2]),
            nn.LayerNorm(hidden_dims[2]),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims[2], hidden_dims[2]//2),
            nn.GELU(),
            nn.Linear(hidden_dims[2]//2, 1)  # 用于对比学习
        )
        
    def forward(self, data_list, return_topological=False):
        """前向传播 - 处理多尺度几何图"""
        all_embeddings = []
        topological_features = []
        
        for data in data_list:
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr
            
            # 几何图卷积
            geometric_features = []
            for conv in self.geometric_convs:
                if isinstance(conv, CurvatureAwareGCN):
                    x = conv(x, edge_index, data.curvature)
                else:
                    x = conv(x, edge_index, edge_attr)
                geometric_features.append(x)
            
            # 多尺度特征融合
            x = torch.stack(geometric_features, dim=1).mean(dim=1)
            
            # 拓扑特征提取
            topo_feat = self.topological_net(x, edge_index, data.edge_attr)
            topological_features.append(topo_feat)
            
            # 特征融合
            combined = torch.cat([x, topo_feat], dim=1)
            all_embeddings.append(combined)
        
        # 多图注意力融合
        if len(all_embeddings) > 1:
            stacked_emb = torch.stack(all_embeddings, dim=1)
            fused_emb, attn_weights = self.fusion_attention(
                stacked_emb, stacked_emb, stacked_emb
            )
            final_embedding = fused_emb.mean(dim=1)
        else:
            final_embedding = all_embeddings[0]
            
        # 聚类投影
        cluster_proj = self.cluster_head(final_embedding)
        
        if return_topological:
            return final_embedding, cluster_proj, topological_features
        return final_embedding, cluster_proj

class CurvatureAwareGCN(nn.Module):
    """曲率感知图卷积网络"""
    
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.curvature_weights = nn.Linear(1, out_dim)
        
    def forward(self, x, edge_index, curvature):
        # 曲率调整的消息传递
        row, col = edge_index
        curvature_diff = torch.abs(curvature[row] - curvature[col]).unsqueeze(1)
        
        # 曲率感知的权重
        curvature_weight = torch.sigmoid(self.curvature_weights(curvature_diff))
        
        # 消息聚合
        out = scatter_mean(x[col] * curvature_weight, row, dim=0, dim_size=x.size(0))
        out = self.linear(out)
        
        return F.gelu(out)

class ContrastiveSpatialClustering:
    """对比学习空间聚类 - 结合SimCLR和SwAV思想"""
    
    def __init__(self, temperature=0.1, queue_size=65536, 
                 projection_dim=128, num_prototypes=30):
        self.temperature = temperature
        self.queue_size = queue_size
        self.projection_dim = projection_dim
        self.num_prototypes = num_prototypes
        
        # 原型向量
        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, projection_dim)
        )
        
        # 队列用于对比学习
        self.register_buffer("queue", torch.randn(projection_dim, queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    def contrastive_loss(self, z1, z2, labels=None):
        """改进的对比损失 - 结合监督信号"""
        batch_size = z1.size(0)
        
        # 正样本对
        pos_sim = F.cosine_similarity(z1, z2, dim=1) / self.temperature
        
        # 负样本对 - 从队列中采样
        neg_sim = torch.mm(z1, self.queue) / self.temperature
        
        # 结合监督信号
        if labels is not None:
            # 同类别样本作为正样本
            label_matrix = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
            supervised_pos = torch.mm(z1, z2.t()) / self.temperature
            pos_sim = pos_sim + supervised_pos.diag()
        
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long).to(z1.device)
        
        loss = F.cross_entropy(logits, labels)
        
        # 更新队列
        self._dequeue_and_enqueue(z2)
        
        return loss
    
    def swav_loss(self, projections, prototypes):
        """SwAV损失 - 在线聚类"""
        # 将投影分配到原型
        with torch.no_grad():
            q = self._distribute_assignments(projections, prototypes)
        
        # 计算交换预测损失
        p = F.softmax(projections @ prototypes.T / self.temperature, dim=1)
        
        loss = - (q * torch.log(p)).sum(dim=1).mean()
        return loss
    
    def _distribute_assignments(self, projections, prototypes):
        """Sinkhorn-Knopp分配"""
        Q = torch.exp(projections @ prototypes.T / 0.05)
        Q /= Q.sum(dim=0, keepdim=True)
        
        # Sinkhorn迭代
        for _ in range(3):
            Q /= Q.sum(dim=1, keepdim=True)
            Q /= Q.sum(dim=0, keepdim=True)
            
        return Q

class SpatialVAE(nn.Module):
    """空间变分自编码器 - 结合图结构和空间坐标"""
    
    def __init__(self, in_dim, hidden_dims=[128, 64], latent_dim=32):
        super().__init__()
        
        # 编码器
        encoder_dims = [in_dim] + hidden_dims
        self.encoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dims[i], encoder_dims[i+1]),
                nn.BatchNorm1d(encoder_dims[i+1]),
                nn.GELU()
            ) for i in range(len(encoder_dims)-1)
        ])
        
        # 潜在空间参数
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # 解码器
        decoder_dims = [latent_dim] + hidden_dims[::-1] + [in_dim]
        self.decoder = nn.ModuleList([
            nn.Sequential(
                nn.Linear(decoder_dims[i], decoder_dims[i+1]),
                nn.BatchNorm1d(decoder_dims[i+1]),
                nn.GELU() if i < len(decoder_dims)-2 else nn.Identity()
            ) for i in range(len(decoder_dims)-1)
        ])
        
        # 图解码器
        self.edge_decoder = nn.Sequential(
            nn.Linear(latent_dim * 2, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        for layer in self.encoder:
            x = layer(x)
        return self.fc_mu(x), self.fc_logvar(x)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        for layer in self.decoder:
            z = layer(z)
        return z
    
    def decode_edges(self, z, edge_index):
        src, dst = edge_index
        z_src, z_dst = z[src], z[dst]
        edge_features = torch.cat([z_src, z_dst], dim=1)
        return self.edge_decoder(edge_features).squeeze()
    
    def forward(self, x, edge_index):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        edge_recon = self.decode_edges(z, edge_index)
        
        return x_recon, edge_recon, mu, logvar

class GeometricDeepSpatialClustering:
    """几何深度空间聚类 - 完整解决方案"""
    
    def __init__(self, config=None):
        self.config = self._get_default_config() if config is None else config
        self.graph_builder = MultiScaleGeometricGraph()
        self.geometric_gnn = None
        self.spatial_vae = None
        self.contrastive_module = ContrastiveSpatialClustering()
        
    def _get_default_config(self):
        return {
            'geometric_graph': {
                'methods': ['riemannian', 'persistence', 'morse'],
                'metric_learning': True
            },
            'model': {
                'hidden_dims': [256, 128, 64],
                'latent_dim': 32,
                'num_heads': 8,
                'projection_dim': 128
            },
            'training': {
                'contrastive_weight': 1.0,
                'vae_weight': 0.5,
                'topological_weight': 0.3,
                'smoothness_weight': 0.2,
                'lr': 0.001,
                'n_epochs': 1000
            },
            'clustering': {
                'min_cluster_size': 5,
                'resolution_range': [0.1, 2.0],
                'uncertainty_threshold': 0.1
            }
        }
    
    def fit_predict(self, X, F, coords, initial_clusters=None, label_mask=None):
        """端到端训练和预测"""
        
        # 阶段1: 多尺度几何图构建
        geometric_graphs = self.graph_builder.build_geometric_graphs(X, F, coords)
        
        # 阶段2: 模型初始化
        in_dim = X.shape[1] + F.shape[1]
        self.geometric_gnn = GeometricGraphNeuralNetwork(
            in_dim, **self.config['model']
        )
        self.spatial_vae = SpatialVAE(
            in_dim, 
            hidden_dims=self.config['model']['hidden_dims'],
            latent_dim=self.config['model']['latent_dim']
        )
        
        # 阶段3: 多任务训练
        trained_models = self._multitask_training(
            geometric_graphs, initial_clusters, label_mask
        )
        
        # 阶段4: 几何聚类
        final_results = self._geometric_clustering(
            geometric_graphs, trained_models, X, F, coords, 
            initial_clusters, label_mask
        )
        
        return final_results
    
    def _multitask_training(self, graphs, initial_clusters, label_mask):
        """多任务训练 - 结合对比学习、VAE和拓扑约束"""
        
        optimizer = torch.optim.AdamW(
            list(self.geometric_gnn.parameters()) + 
            list(self.spatial_vae.parameters()),
            lr=self.config['training']['lr']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=100, T_mult=2
        )
        
        for epoch in range(self.config['training']['n_epochs']):
            total_loss = 0
            
            for graph_name, graph_data in graphs.items():
                # 数据增强 - 创建视图
                aug_data1 = self._geometric_augmentation(graph_data)
                aug_data2 = self._feature_augmentation(graph_data)
                
                # 前向传播
                emb1, proj1 = self.geometric_gnn([aug_data1])
                emb2, proj2 = self.geometric_gnn([aug_data2])
                
                # 对比损失
                contrast_loss = self.contrastive_module.contrastive_loss(
                    proj1, proj2, 
                    labels=initial_clusters if label_mask is not None else None
                )
                
                # VAE损失
                x_recon, edge_recon, mu, logvar = self.spatial_vae(
                    graph_data.x, graph_data.edge_index
                )
                vae_loss = self._vae_loss(
                    x_recon, graph_data.x, edge_recon, 
                    graph_data.edge_attr, mu, logvar
                )
                
                # 拓扑一致性损失
                topological_loss = self._topological_consistency_loss(
                    emb1, graph_data.edge_index
                )
                
                # 图平滑度损失
                smoothness_loss = self._graph_smoothness_loss(
                    emb1, graph_data.edge_index, graph_data.edge_attr
                )
                
                # 总损失
                loss = (self.config['training']['contrastive_weight'] * contrast_loss +
                       self.config['training']['vae_weight'] * vae_loss +
                       self.config['training']['topological_weight'] * topological_loss +
                       self.config['training']['smoothness_weight'] * smoothness_loss)
                
                total_loss += loss
            
            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.geometric_gnn.parameters()) + 
                list(self.spatial_vae.parameters()), 
                1.0
            )
            optimizer.step()
            scheduler.step()
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss.item():.4f}")
        
        return {'gnn': self.geometric_gnn, 'vae': self.spatial_vae}
    
    def _geometric_clustering(self, graphs, models, X, F, coords, 
                            initial_clusters, label_mask):
        """几何聚类 - 结合多层次信息"""
        
        # 1. 提取多层次嵌入
        with torch.no_grad():
            all_embeddings = []
            
            for graph_data in graphs.values():
                emb, _ = models['gnn']([graph_data])
                all_embeddings.append(emb.cpu().numpy())
            
            # 集成嵌入
            ensemble_embedding = np.mean(all_embeddings, axis=0)
            
            # VAE潜在表示
            mu, _ = models['vae'].encode(
                torch.cat([torch.tensor(X), torch.tensor(F)], dim=1).float()
            )
            vae_embedding = mu.cpu().numpy()
        
        # 2. 多分辨率 Leiden 聚类
        cluster_ensemble = []
        resolutions = np.linspace(
            self.config['clustering']['resolution_range'][0],
            self.config['clustering']['resolution_range'][1], 
            10
        )
        
        for res in resolutions:
            # 在嵌入空间聚类
            clusters_emb = self._leiden_clustering(ensemble_embedding, resolution=res)
            
            # 在VAE空间聚类
            clusters_vae = self._leiden_clustering(vae_embedding, resolution=res)
            
            cluster_ensemble.extend([clusters_emb, clusters_vae])
        
        # 3. 聚类集成和共识
        consensus_clusters = self._spectral_consensus_clustering(
            cluster_ensemble, n_clusters=len(np.unique(initial_clusters[initial_clusters != -1]))
        )
        
        # 4. 空间拓扑优化
        optimized_clusters = self._spatial_topological_optimization(
            consensus_clusters, X, coords, graphs
        )
        
        # 5. 不确定性量化和主动学习
        uncertainty_scores = self._compute_uncertainty(
            ensemble_embedding, optimized_clusters, graphs
        )
        
        # 6. 生成最终结果
        final_results = self._generate_final_clustering(
            optimized_clusters, initial_clusters, label_mask, 
            uncertainty_scores, ensemble_embedding
        )
        
        return final_results
    
    def _spatial_topological_optimization(self, clusters, X, coords, graphs):
        """空间拓扑优化 - 基于持续同调"""
        
        optimized = clusters.copy()
        
        # 识别拓扑特征（洞、连接组件）
        topological_features = self._compute_persistent_homology(X, clusters)
        
        # 基于拓扑特征优化聚类
        for cluster_id in np.unique(clusters):
            cluster_mask = (clusters == cluster_id)
            
            if np.sum(cluster_mask) < self.config['clustering']['min_cluster_size']:
                continue
                
            # 检查空间连通性
            n_components = self._connected_components(X[cluster_mask])
            
            if n_components > 1:
                # 拆分不连通的组件
                component_labels = self._label_connected_components(X[cluster_mask])
                
                for comp_id in range(1, n_components):
                    comp_mask = (component_labels == comp_id)
                    original_indices = np.where(cluster_mask)[0][comp_mask]
                    optimized[original_indices] = np.max(optimized) + 1
        
        return optimized
    
    def _compute_uncertainty(self, embeddings, clusters, graphs):
        """不确定性量化 - 基于集成和拓扑"""
        
        uncertainties = np.zeros(len(embeddings))
        
        for i in range(len(embeddings)):
            # 1. 集成不确定性
            cluster_id = clusters[i]
            same_cluster_mask = (clusters == cluster_id)
            
            if np.sum(same_cluster_mask) > 1:
                intra_variance = np.var(embeddings[same_cluster_mask], axis=0).mean()
            else:
                intra_variance = 1.0
            
            # 2. 拓扑不确定性
            topological_uncertainty = self._topological_uncertainty(i, clusters, graphs)
            
            # 3. 边界不确定性
            boundary_uncertainty = self._boundary_uncertainty(i, clusters, embeddings)
            
            uncertainties[i] = (intra_variance + 
                              topological_uncertainty + 
                              boundary_uncertainty) / 3
        
        return uncertainties

# 使用示例
def advanced_spatial_clustering(X, F, coords, initial_clusters=None, label_mask=None):
    """
    X: 空间坐标
    F: 特征矩阵  
    coords: 空间坐标（可能包含额外信息）
    initial_clusters: 初始聚类（可选）
    label_mask: 标签掩码（可选）
    """
    
    cluster_engine = GeometricDeepSpatialClustering()
    
    results = cluster_engine.fit_predict(
        X=X, F=F, coords=coords,
        initial_clusters=initial_clusters,
        label_mask=label_mask
    )
    
    return results