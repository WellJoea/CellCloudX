from tqdm import tqdm
import itertools
import numpy as np
import torch.multiprocessing as mp
import torch as th
import scipy.sparse as ssp
from .neighbors_ensemble import Neighbors
from ...utilis._clean_cache import clean_cache
from ...io._logger import logger

class deformable_regularizer(object):
    def __init__(self, beta =1.0, kw=15, kl=20, num_eig=100,
                gamma1=0.0, gamma2=0.0, use_p1=True,
                use_fast_low_rank = False, use_low_rank=False,
                low_rank_type = 'keops',
                use_unique=False, kd_method='sknn', xp = None, verbose=True, **kwargs):
        self.beta = beta
        self.kw = kw
        self.kl = kl
        self.low_rank_type = low_rank_type # keops or fgt
        self.use_fast_low_rank = bool(use_fast_low_rank)
        self.use_low_rank = bool(use_low_rank)
        self.use_fast = self.use_fast_low_rank or self.use_low_rank

        self.use_unique = bool(use_unique)
        self.kd_method = kd_method
        self.num_eig = num_eig
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.use_p1 = bool(use_p1)
        self.xp =  xp
        self.verbose = verbose
        for ia in ['G', 'U', 'S', 'L', 'A', 'I', 'LV', 'LY', 'AV', 'J']:
            setattr(self, ia, None)

    def compute(self, Y, device=None, dtype=None,):
        Y = Y.detach().clone().to(device, dtype=dtype)
        
        if self.verbose: logger.info(f'compute G EVD: use fast/low_rank({self.low_rank_type})'
                                     f' = {self.use_fast_low_rank}/{self.use_low_rank}...')
        if self.use_fast:
            if self.low_rank_type == 'keops':
                # U, S = low_rank_evd_grbf_keops(Y, self.beta, self.num_eig) #TODO
                U, S  = low_rank_evd_grbf(Y, self.beta, self.num_eig, use_keops=True)
            elif self.low_rank_type == 'fgt':
                if self.use_fast_low_rank:
                    U, S  = low_rank_evd_grbf(Y, self.beta, self.num_eig, sw_h=0, use_keops=False)
                elif self.use_low_rank:
                    G = rbf_kernal(Y, Y, self.beta, temp=1.0, use_keops=False,)
                    U, S = low_rank_evd( G, self.num_eig) #(G+G.T)/2
            else:
                raise ValueError(f'low_rank_type {self.low_rank_type} not supported')
        else:
            G = rbf_kernal(Y, Y, self.beta, temp=1.0, use_keops=False,)

        if (self.gamma1>0):
            if self.verbose: logger.info('compute Y lle...')
            L = lle_w(Y, use_unique = self.use_unique, kw=self.kw, method=self.kd_method)

        if (self.gamma2>0):
            if self.verbose: logger.info('compute Y gl...')
            A = gl_w(Y, kw=self.kl, method=self.kd_method)

        if self.use_fast:
            self.G = None
            self.U = U
            self.S = self.xp.diag(S)
            self.I = self.xp.eye(self.num_eig, dtype=dtype, device=device)
            V = self.U @ self.S
            if (self.use_p1):
                if (self.gamma1>0):
                    self.L = L
                    self.LV = L @ V
                    self.LY = L @ Y
                if (self.gamma2>0):
                    self.A = A
                    self.AV = A @ V
            else:
                self.J = 0
                if (self.gamma2>0):
                    RV = A.T @ (A @ V)
                else:
                    RV = 0
            
                if (self.gamma1>0):
                    QV = L.T @ (L @ V)
                    self.J =  QV + RV*(self.gamma2/self.gamma1)
                    self.QY = L.T @ (L @ Y)
                else:
                    self.J = RV
                    self.QY = 0
        else:
            self.G = G
            self.U, self.S = None, None
            if (self.use_p1):
                if (self.gamma1>0):
                    self.L = L
                    self.LG = L @ G
                    self.LY = L @ Y
                if (self.gamma2>0):
                    self.A = A
                    self.AG = A @ G
            else:
                if (self.gamma2>0):
                    self.RG = A.T @ (A @ G)
                if (self.gamma1>0):
                    self.QG = L.T @ (L @ G)
                    self.QY = L.T @ (L @ Y)

class pwdeformable_regularizer(object):
    def __init__(self, beta =1.0, kw=15, kl=20, num_eig=100,
                gamma1=0.0, gamma2=0.0, use_p1=True,
                use_fast_low_rank = False, use_low_rank=False,
                low_rank_type = 'keops',
                use_unique=False, kd_method='sknn', xp = None, verbose=True, **kwargs):
        self.beta = beta
        self.kw = kw
        self.kl = kl
        self.low_rank_type = low_rank_type # keops or fgt
        self.use_fast_low_rank = bool(use_fast_low_rank)
        self.use_low_rank = bool(use_low_rank)
        self.use_fast = self.use_fast_low_rank or self.use_low_rank

        self.use_unique = bool(use_unique)
        self.kd_method = kd_method
        self.num_eig = num_eig
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.use_p1 = bool(use_p1)
        self.xp =  xp
        self.verbose = verbose
        for ia in ['G', 'U', 'S', 'I', 'J1', 'J2', 'J3', 'E', 'F', ]:
            setattr(self, ia, None)

    def compute(self, Y, device=None, dtype=None,):
        Y = Y.detach().clone().to(device, dtype=dtype)
        # beta_sf = centerlize(Y)[2] #TODO
    
        if self.verbose: logger.info(f'compute G EVD: use fast/low_rank({self.low_rank_type})'
                                     f' = {self.use_fast_low_rank}/{self.use_low_rank}...')
        if self.use_fast:
            if self.low_rank_type == 'keops':
                # U, S = low_rank_evd_grbf_keops(Y, self.beta, self.num_eig) #TODO
                U, S  = low_rank_evd_grbf(Y, self.beta, self.num_eig, use_keops=True)
            elif self.low_rank_type == 'fgt':
                if self.use_fast_low_rank:
                    U, S  = low_rank_evd_grbf(Y, self.beta, self.num_eig, sw_h=0, use_keops=False)
                elif self.use_low_rank:
                    G = rbf_kernal(Y, Y, self.beta, temp=1.0, use_keops=False,)
                    U, S = low_rank_evd( G, self.num_eig) #(G+G.T)/2
            else:
                raise ValueError(f'low_rank_type {self.low_rank_type} not supported')
        else:
            G = rbf_kernal(Y, Y, self.beta, temp=1.0, use_keops=False,)

        if any(self.gamma1>0):
            if self.verbose: logger.info('compute Y lle...')
            L = lle_w(Y, use_unique = self.use_unique, kw=self.kw, method=self.kd_method)

        if any(self.gamma2>0):
            if self.verbose: logger.info('compute Y gl...')
            A = gl_w(Y, kw=self.kl, method=self.kd_method)

        if self.use_fast:
            self.G = None
            self.U = U
            self.S = self.xp.diag(S)
            self.I = self.xp.eye(self.num_eig, dtype=dtype, device=device)
            V = self.U @ self.S
            if (self.use_p1):
                if any(self.gamma1>0):
                    self.E = L
                    self.J1 = L @ V
                    self.J3 = L @ Y
                if any(self.gamma2>0):
                    self.F = A
                    self.J2 = A @ V
            else:
                if any(self.gamma2>0):
                    self.J2 = A.T @ (A @ V)
                else:
                    self.J2 = 0
            
                if any(self.gamma1>0):
                    self.J1 = L.T @ (L @ V)
                    self.J3 = L.T @ (L @ Y)
                else:
                    self.J1 = 0
                    self.J3 = 0
        else:
            self.G = G
            self.U, self.S = None, None
            if (self.use_p1):
                if any(self.gamma1>0):
                    self.E = L
                    self.J1 = L @ G
                    self.J3 = L @ Y
                if any(self.gamma2>0):
                    self.F = A
                    self.J2 = A @ G
            else:
                if any(self.gamma2>0):
                    self.J2 = A.T @ (A @ G)
                else:
                    self.J2 = 0
            
                if any(self.gamma1>0):
                    self.J1 = L.T @ (L @ G)
                    self.J3 = L.T @ (L @ Y)
                else:
                    self.J1 = 0
                    self.J3 = 0

class rfdeformable_regularizer(object):
    def __init__(self, beta =1.0, kw=15, kl=20, num_eig=100,
                gamma1=0.0, gamma2=0.0, use_p1=True,
                low_rank=3000, fast_low_rank = 5000, 
                use_fast_low_rank = False, use_low_rank=False,
                low_rank_type = 'keops',
                use_unique=False, kd_method='sknn', xp = None, verbose=True, **kwargs):
        self.beta = beta
        self.kw = kw
        self.kl = kl
        self.low_rank_type = low_rank_type # keops or fgt
        self.low_rank = low_rank
        self.fast_low_rank = fast_low_rank
        self.use_fast_low_rank = bool(use_fast_low_rank)
        self.use_low_rank = bool(use_low_rank)
        self.use_fast = self.use_fast_low_rank or self.use_low_rank

        self.use_unique = bool(use_unique)
        self.kd_method = kd_method
        self.num_eig = num_eig
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.use_p1 = bool(use_p1)
        self.xp =  xp
        self.verbose = verbose
        for ia in ['G', 'U', 'S', 'I', 'J1', 'J2', 'J3', 'E', 'F', ]:
            setattr(self, ia, None)

    def compute(self, Y, device=None, dtype=None,):
        Y = Y.detach().clone().to(device, dtype=dtype)
        
        if self.verbose: logger.info(f'compute G EVD: use fast/low_rank({self.low_rank_type})'
                                     f' = {self.use_fast_low_rank}/{self.use_low_rank}...')
        if self.use_fast:
            if self.low_rank_type == 'keops':
                # U, S = low_rank_evd_grbf_keops(Y, self.beta, self.num_eig) #TODO
                U, S  = low_rank_evd_grbf(Y, self.beta, self.num_eig, use_keops=True)
            elif self.low_rank_type == 'fgt':
                if self.use_fast_low_rank:
                    U, S  = low_rank_evd_grbf(Y, self.beta, self.num_eig, sw_h=0, use_keops=False)
                elif self.use_low_rank:
                    G = rbf_kernal(Y, Y, self.beta, temp=1.0, use_keops=False,)
                    U, S = low_rank_evd( G, self.num_eig) #(G+G.T)/2
            else:
                raise ValueError(f'low_rank_type {self.low_rank_type} not supported')
        else:
            G = rbf_kernal(Y, Y, self.beta, temp=1.0, use_keops=False,)

        if any(self.gamma1>0):
            if self.verbose: logger.info('compute Y lle...')
            L = lle_w(Y, use_unique = self.use_unique, kw=self.kw, method=self.kd_method)

        if any(self.gamma2>0):
            if self.verbose: logger.info('compute Y gl...')
            A = gl_w(Y, kw=self.kl, method=self.kd_method)

        if self.use_fast:
            self.G = None
            self.U = U
            self.S = self.xp.diag(S)
            self.I = self.xp.eye(self.num_eig, dtype=dtype, device=device)
            V = self.U @ self.S
            if (self.use_p1):
                if any(self.gamma1>0):
                    self.E = L
                    self.J1 = L @ V
                    self.J3 = L @ Y
                if any(self.gamma2>0):
                    self.F = A
                    self.J2 = A @ V
            else:
                if any(self.gamma2>0):
                    self.J2 = A.T @ (A @ V)
                else:
                    self.J2 = 0
            
                if any(self.gamma1>0):
                    self.J1 = L.T @ (L @ V)
                    self.J3 = L.T @ (L @ Y)
                else:
                    self.J1 = 0
                    self.J3 = 0
        else:
            self.G = G
            self.U, self.S = None, None
            if (self.use_p1):
                if any(self.gamma1>0):
                    self.E = L
                    self.J1 = L @ G
                    self.J3 = L @ Y
                if any(self.gamma2>0):
                    self.F = A
                    self.J2 = A @ G
            else:
                if any(self.gamma2>0):
                    self.J2 = A.T @ (A @ G)
                else:
                    self.J2 = 0
            
                if any(self.gamma1>0):
                    self.J1 = L.T @ (L @ G)
                    self.J3 = L.T @ (L @ Y)
                else:
                    self.J1 = 0
                    self.J3 = 0

    def compute_pair(self, Y1,  Y2, device=None, dtype=None,):
        Y1 = Y1.detach().clone().to(device, dtype=dtype)
        Y2 = Y2.detach().clone().to(device, dtype=dtype)

        if self.low_rank_type == 'keops':
            U, S, Vh  = low_rank_svd_grbf(Y1, Y2, self.beta, self.num_eig, use_keops=True)
        elif self.low_rank_type == 'fgt':
            try:
                U, S, Vh  = low_rank_svd_grbf(Y1, Y2, self.beta, self.num_eig, sw_h=0, use_keops=False)
            except:
                G = rbf_kernal(Y1, Y2, self.beta, temp=1.0, use_keops=False,)
                U, S, Vh = low_rank_evd( G, self.num_eig) #(G+G.T)/2
        return U, S, Vh

    def compute_pair0(self, Y1,  Y2, device=None, dtype=None,):
        Y1 = Y1.detach().clone().to(device, dtype=dtype)
        Y2 = Y2.detach().clone().to(device, dtype=dtype)

        use_low_rank = ( self.low_rank if type(self.low_rank) == bool  
                                else bool( (Y1.shape[0]*Y2.shape[0])**0.5 >= self.low_rank) )
        use_fast_low_rank = ( self.fast_low_rank if type(self.fast_low_rank) == bool  
                                else bool( (Y1.shape[0]*Y2.shape[0])**0.5 >= self.fast_low_rank) )
        use_fast = use_low_rank or use_fast_low_rank
        if use_fast:
            if self.low_rank_type == 'keops':
                U, S, Vh  = low_rank_svd_grbf(Y1, Y2, self.beta, self.num_eig, use_keops=True)
            elif self.low_rank_type == 'fgt':
                if use_fast_low_rank:
                    U, S, Vh  = low_rank_svd_grbf(Y1, Y2, self.beta, self.num_eig, sw_h=0, use_keops=False)
                elif use_low_rank:
                    G = rbf_kernal(Y1, Y2, self.beta, temp=1.0, use_keops=False,)
                    U, S, Vh = low_rank_evd( G, self.num_eig) #(G+G.T)/2
            else:
                raise ValueError(f'low_rank_type {self.low_rank_type} not supported')
            return U, S, Vh
        else:
            G = rbf_kernal(Y1, Y2, self.beta, temp=1.0, use_keops=False,)
            return G

class ParallelGsPair:
    def __init__(self, num_eig=100, low_rank_g=False, devices=None, verbose=0):
        self.num_eig = num_eig
        self.low_rank_g = low_rank_g
        self.devices = devices or [torch.device('cuda' if torch.cuda.is_available() else 'cpu')]
        self.verbose = verbose
        self.manager = mp.Manager()
        self.results = self.manager.dict()
        
        # 预分配GPU内存池
        if self.verbose > 1:
            print(f"Initializing memory pools on {len(self.devices)} devices")
        self._init_memory_pools()

    def _init_memory_pools(self):
        """初始化各GPU的内存池"""
        self.mem_pools = {}
        for device in self.devices:
            if device.type == 'cuda':
                torch.cuda.init()
                self.mem_pools[device] = torch.cuda.CUDAPool(
                    device=device,
                    size=torch.cuda.get_device_properties(device).total_memory * 0.8
                )

    def compute_pairs(self, Ys, L, h2s, mask=None):
        """多GPU并行计算入口"""
        # 阶段1：任务预处理
        task_queue, total_tasks = self._preprocess_tasks(Ys, L, h2s, mask)
        
        # 阶段2：启动并行工作进程
        ctx = mp.get_context('spawn')
        processes = []
        for device_idx, device in enumerate(self.devices):
            p = ctx.Process(
                target=self._worker,
                args=(task_queue, Ys, h2s, device, device_idx)
            )
            p.start()
            processes.append(p)
        
        # 阶段3：进度监控与结果收集
        self._monitor_progress(processes, total_tasks)
        
        # 阶段4：后处理结果
        self._postprocess_results(L)

    def _preprocess_tasks(self, Ys, L, h2s, mask):
        """任务预处理：生成任务队列并处理对称性"""
        task_queue = mp.Queue()
        task_count = 0
        
        # 数据预分配到各GPU
        device_data = self._preallocate_data(Ys)
        
        for i, j in itertools.product(range(L), range(L)):
            if i == j:
                task_queue.put(('eigen', i, j))
                task_count += 1
            else:
                if mask is not None and mask[i,j] == 0:
                    continue
                # 对称性检查
                if self._check_symmetry(i, j, h2s):
                    self._handle_symmetry(i, j)
                    continue
                task_queue.put(('svd', i, j))
                task_count += 1
        return task_queue, task_count

    def _preallocate_data(self, Ys):
        """数据预分配到各GPU设备"""
        device_data = {}
        for device in self.devices:
            device_data[device] = [
                Y.to(device, non_blocking=True) 
                for Y in Ys
            ]
        return device_data

    def _check_symmetry(self, i, j, h2s):
        """检查是否存在对称可复用结果"""
        return (f'US{j}{i}' in self.results 
                and h2s[i] == h2s[j]
                and f'Vh{j}{i}' in self.results)

    def _handle_symmetry(self, i, j):
        """处理对称计算结果"""
        self.results[f'US{i}{j}'] = self.results[f'Vh{j}{i}'].T
        self.results[f'Vh{i}{j}'] = self.results[f'US{j}{i}'].T

    def _worker(self, queue, Ys, h2s, device, device_idx):
        """工作进程主函数"""
        torch.cuda.set_device(device) if device.type == 'cuda' else None
        
        # 启用内存池
        if device in self.mem_pools:
            torch.cuda.set_per_process_memory_fraction(0.8, device=device)
            torch.cuda.set_allocator(self.mem_pools[device].allocator)
        
        while not queue.empty():
            try:
                task_type, i, j = queue.get_nowait()
                
                # 获取预分配数据
                X = Ys[i].to(device, non_blocking=True)
                h2 = h2s[i]
                
                if task_type == 'eigen':
                    Q, S = self._compute_evd(X, h2, device)
                    self.results[f'Q{i}{j}'] = Q.cpu()
                    self.results[f'S{i}{j}'] = S.cpu()
                else:
                    Y = Ys[j].to(device, non_blocking=True)
                    U, S, Vh = self._compute_svd(X, Y, h2, device)
                    self.results[f'US{i}{j}'] = (U * S).cpu()
                    self.results[f'Vh{i}{j}'] = Vh.cpu()
                    
            except Exception as e:
                print(f"Device {device} error: {str(e)}")
                queue.put((task_type, i, j))  # 重新放回任务队列

    def _compute_svd(self, X, Y, h2, device):
        """设备感知的SVD计算"""
        with torch.cuda.device(device):
            if (X.shape[0] * Y.shape[0]) >= (self.low_rank_g / 100):
                return self._low_rank_svd_grbf(X, Y, h2**0.5)
            else:
                G = torch.cdist(X, Y)
                G.pow_(2).div_(-h2).exp_()
                return self._low_rank_evd(G)

    def _compute_evd(self, X, h2, device):
        """设备感知的特征分解"""
        with torch.cuda.device(device):
            if (X.shape[0] ** 2) >= self.low_rank_g:
                return self._low_rank_evd_grbf(X, h2**0.5)
            else:
                G = torch.cdist(X, X)
                G.pow_(2).div_(-h2).exp_()
                return self._low_rank_evd(G)

    def _monitor_progress(self, processes, total_tasks):
        """监控进度与异常处理"""
        with tqdm(total=total_tasks, desc="Parallel Computing", 
                 disable=not self.verbose) as pbar:
            while any(p.is_alive() for p in processes):
                current = total_tasks - self._count_remaining_tasks(processes)
                pbar.update(current - pbar.n)

        # 检查异常终止
        for p in processes:
            if p.exitcode != 0:
                raise RuntimeError(f"Process {p.pid} exited with code {p.exitcode}")

    def _count_remaining_tasks(self, processes):
        """统计剩余任务量"""
        return sum(p.is_alive() for p in processes)

    def _postprocess_results(self, L):
        """后处理：将结果设置到对象属性"""
        for key in self.results:
            if key.startswith('Q') or key.startswith('S'):
                setattr(self, key, self.results[key])
            elif key.startswith('US'):
                setattr(self, key, self.results[key])
            elif key.startswith('Vh'):
                setattr(self, key, self.results[key])

    # 以下是需要实现的低秩近似方法
    def _low_rank_svd_grbf(self, X, Y, gamma):
        """GPU优化的低秩SVD实现"""
        # 实现细节...
        
    def _low_rank_evd(self, G):
        """基于随机投影的特征分解"""
        # 实现细节...
        
    def _low_rank_evd_grbf(self, X, gamma):
        """核矩阵低秩近似"""
        # 实现细节...

# # 使用示例
# if __name__ == "__main__":
#     # 初始化配置
#     devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
#     gs = ParallelGsPair(
#         num_eig=100,
#         low_rank_g=1e6,
#         devices=devices,
#         verbose=1
#     )
    
#     # 输入数据
#     L = 10
#     M, D = 1000, 3
#     Ys = [torch.randn(M, D) for _ in range(L)]
#     beta = torch.randn(L)
#     delta = torch.ones(L, L)
    
#     # 执行计算
#     gs.compute_pairs(Ys, L, beta, mask=delta)


class Gspair(object):
    def __init__(self,  num_eig=100, use_ifgt = False, xp = None, njob=None, verbose=0):
        if xp is None:
            import torch as xp
        self.xp = xp
        self.njob = njob
        self.use_ifgt = use_ifgt
        self.num_eig = num_eig
        self.verbose = verbose

    def compute_pairs(self, Ys, L, h2s, mask=None):
        import itertools
        with tqdm(total=L*L, 
                    desc="Gs E/SVD",
                    colour='#AAAAAA', 
                    disable=(self.verbose==0)) as pbar:
            for i, j in itertools.product(range(L), range(L)):
                pbar.set_postfix(dict(i=int(i), j=int(j)))
                pbar.update()
                if i==j:
                    (Q, S) = compute_evd(Ys[i], h2s[i], self.num_eig, 
                                            use_ifgt=self.use_ifgt, xp=self.xp)
                    setattr(self, 'Q'+str(i)+str(j), Q)
                    setattr(self, 'S'+str(i)+str(j), S)
                else:
                    if (mask is not None) and (mask[i,j]==0):
                        continue
                    if hasattr(self, f'US{j}{i}') and h2s[i] == h2s[j]:
                        for ia, ib in zip(['US', 'Vh'], ['Vh', 'US']):
                            setattr(self, ia+str(i)+str(j), getattr(self, ib+str(j)+str(i)).T)
                    else:
                        (U, S, Vh) = compute_svd(Ys[i], Ys[j], h2s[i], self.num_eig, 
                                                 use_ifgt=self.use_ifgt/100, xp=self.xp)
                        setattr(self, 'US'+str(i)+str(j), U * S)
                        setattr(self, 'Vh'+str(i)+str(j), Vh)

    def compute_pairs_multhread(self, Ys, L, h2s, mask=None):
        import concurrent.futures
        tasks = []
        for i, j in itertools.product(range(L), range(L)):
            if i == j:
                tasks.append(('eigen', i, j))
            else:
                if mask is not None and mask[i, j] == 0:
                    continue
                if hasattr(self, f'US{j}{i}') and h2s[i] == h2s[j]:
                    # Handle symmetric cases without computation
                    for src_suffix, dst_suffix in zip(['Vh', 'US'], ['US', 'Vh']):
                        src_attr = f"{src_suffix}{j}{i}"
                        dst_attr = f"{dst_suffix}{i}{j}"
                        if hasattr(self, src_attr):
                            setattr(self, dst_attr, getattr(self, src_attr).T)
                tasks.append(('svd', i, j))

        with tqdm(total=len(tasks), 
                desc="Gs E/SVD",
                colour='#AAAAAA', 
                disable=(self.verbose==0)) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.njob) as executor:
                futures = []
                for task_type, i, j in tasks:
                    if task_type == 'eigen':
                        future = executor.submit(
                            self._process_eigen_task, Ys[i], h2s[i], i, j
                        )
                    else:
                        future = executor.submit(
                            self._process_svd_task, Ys[i], Ys[j], h2s[i], i, j
                        )
                    future.add_done_callback(lambda fn: pbar.update())
                    futures.append(future)
                # Wait for all tasks to complete
                concurrent.futures.wait(futures)

    def _process_eigen_task(self, Y, h2, i, j):
        (Q, S) = compute_evd(Y, h2, self.num_eig, 
                                use_ifgt=self.use_ifgt, xp=self.xp)
        setattr(self, f'Q{i}{j}', Q)
        setattr(self, f'S{i}{j}', S)

    def _process_svd_task(self, X, Y, h2, i, j):
        (U, S, Vh) = compute_svd(X, Y, h2, self.num_eig, 
                            use_ifgt=self.use_ifgt/100, xp=self.xp)
        setattr(self, f'US{i}{j}', U * S)
        setattr(self, f'Vh{i}{j}', Vh)


def rbf_kernal(X, Y, h2, use_keops=False,
                     device=None, dtype=None,
                    temp=1.0 ):
    device = X.device if device is None else device
    dtype  = X.dtype if dtype is None else dtype
    h = (temp/th.asarray(h2, device=device, dtype=dtype))**0.5
    if use_keops:
        d2f = kodist2( X.to(device, dtype=dtype) * h,
                       Y.to(device, dtype=dtype) * h)
        # d2f = d2f*(-1.0/float(tau2)/temp)
        d2f = d2f*(-1.0)
        d2f = d2f.exp()
    else:
        try:
            d2f = thdist2(X.to(device, dtype=dtype) * h, 
                          Y.to(device, dtype=dtype) * h)
            # d2f.mul_(-1.0/float(tau2)/temp)
            d2f.mul_(-1.0)
            d2f.exp_()
        except:
            clean_cache()
            raise('Memory Error in computing d2f')
    return d2f

def kodist2(X, Y):
    import pykeops
    pykeops.set_verbose(False)
    from pykeops.torch import LazyTensor

    x_i = LazyTensor(X[:, None, :])
    y_j = LazyTensor(Y[None, :, :])
    return ((x_i - y_j)**2).sum(dim=2)

def thdist2(X, Y):
    import torch as th
    D = th.cdist(X, Y, p=2)
    D.pow_(2)

    # D1 = (( X[:, None, :] - Y[None,:, :] )**2).sum(-1)
    return D

def lle_w(Y, kw=15, use_unique=False, rl_w=None,  method='sknn'): #TODO 
    #D2 =np.sum(np.square(Y[:, None, :]- Y[None, :, :]), axis=-1)
    #D3 = D2 + np.eye(D2.shape[0])*D2.max()
    #cidx = np.argpartition(D3, self.knn)[:, :self.knn]

    if hasattr(Y, 'detach'):
        uY = Y.detach().cpu().numpy()
        is_tensor = True
        device = Y.device
        dtype = Y.dtype
    else:
        uY = Y
        is_tensor = False

    if use_unique:
        uY, Yidx = np.unique(uY, return_inverse=True,  axis=0)
    eps = np.finfo(uY.dtype).eps
    M, D = uY.shape
    Mr =  Y.shape[0]

    if rl_w is None:
        rl_w = 1e-6 if(kw>D) else 0

    snn = Neighbors(method=method)
    snn.fit(uY)
    ckdout = snn.transform(uY, knn=kw+1)
    kdx = ckdout[1][:,1:]
    L = []
    for i in range(M):
        kn = kdx[i]
        z = (uY[kn] - uY[i]) #K*D
        G = z @ z.T # K*K
        Gtr = np.trace(G)
        if Gtr != 0:
            G = G +  np.eye(kw) * rl_w* Gtr
        else:
            G = G +  np.eye(kw) * rl_w
        w = np.sum(np.linalg.inv(G), axis=1) #K*1
        #w = solve(G, v, assume_a="pos")
        w = w/ np.sum(w).clip(eps, None)
        L.append(w)
    src = kdx.flatten('C')
    dst = np.repeat(np.arange(kdx.shape[0]), kdx.shape[1])
    L = ssp.csr_array((np.array(L).flatten(), (dst, src)), shape=(M, M))
    if use_unique:
        L = L[Yidx][:, Yidx]
    L  = ssp.eye(Mr) - L
    if is_tensor:
        L = spsparse_to_thsparse(L).to(dtype).to(device)
    return L

def gl_w(Y, Y_feat=None, kw=15, use_unique=False, rl_w=None,  method='sknn'): #TODO 
    if hasattr(Y, 'detach'):
        uY = Y.detach().cpu().numpy()
        is_tensor = True
        device = Y.device
        dtype = Y.dtype
    else:
        uY = Y
        is_tensor = False

    if use_unique:
        uY, Yidx = np.unique(uY, return_inverse=True,  axis=0)

    M, D = uY.shape

    if rl_w is None:
        rl_w = 1e-3 if(kw>D) else 0

    snn = Neighbors(method=method)
    snn.fit(uY)
    ckdout = snn.transform(uY, knn=kw+1)
    kdx = ckdout[1][:,1:]
    src = kdx.flatten('C')
    dst = np.repeat(np.arange(kdx.shape[0]), kdx.shape[1])

    if not Y_feat is None: #TODO
        pass
    else:
        A = np.ones_like(dst)

    L = ssp.csr_array((A, (dst, src)), shape=(M, M))
    if use_unique:
        L = L[Yidx][:, Yidx]

    D = ssp.diags((L.sum(1) )**(-0.5)) # TODO
    A = D @ L @ D
    K  = ssp.eye(A.shape[0]) - A

    if is_tensor:
        K = spsparse_to_thsparse(K).to(dtype).to(device)
    return K

def low_rank_evd_grbf(X, h2, num_eig, sw_h=0.0, eps=1e-10, use_keops=True):
    if hasattr(X, 'detach'):
        is_tensor = True
        device = X.device
        dtype = X.dtype
    else:
        is_tensor = False
        device = None
        dtype = None

    M, D  = X.shape
    k = min(M-1, num_eig)

    if use_keops:
        import pykeops
        import torch as th
        pykeops.set_verbose(False)
        from pykeops.torch import Genred, LazyTensor

        # genred = Genred(f'Exp(-SqDist(Xi, Xj) * H ) * Vj',
        #             [f'Xi = Vi({D})', f'Xj = Vj({D})', 
        #             'H = Pm(1)', 'Vj = Vj(1)' ],
        #             reduction_op='Sum', axis=1, 
        #             # dtype_acc='float64',
        #              use_double_acc=True,
        #             )
        genred = Genred(f'Exp(-SqDist(Xi, Xj) ) * Vj',
                    [f'Xi = Vi({D})', f'Xj = Vj({D})', 'Vj = Vj(1)'],
                    reduction_op='Sum', axis=1, 
                    # dtype_acc='float64',
                     use_double_acc=True,
                    )
        H = (1.0/th.asarray(h2, device=device, dtype=dtype))**0.5
        X = th.asarray(X, dtype=dtype, device=device)*H
        def matvec(x):
            K = genred(X, X, th.tensor(x, dtype=dtype, device=device)).squeeze(1)
            # K = genred(X, X, H, th.tensor(x, dtype=dtype, device=device)).squeeze(1)
            return K.detach().cpu().numpy()

    else:
        from ...third_party._ifgt_warp import GaussTransform, GaussTransform_fgt
        H = (1.0/np.array(h2))**0.5
        X = X.detach().cpu().numpy()*H
        trans = GaussTransform_fgt(X, 1.0, sw_h=sw_h, eps=eps) #XX*s/h, s
        def matvec(x):
            return trans.compute(X, x.T)

    lo = ssp.linalg.LinearOperator((M,M), matvec)
    S, Q = ssp.linalg.eigsh(lo, k=k, which='LM') # speed limitation

    eig_indices = np.argsort(-np.abs(S))[:k]
    Q = np.real(Q[:, eig_indices])  # eigenvectors
    S = np.real(S[eig_indices])  # eigenvalues.

    if is_tensor:
        import torch as th
        Q = th.tensor(Q, dtype=dtype, device=device)
        S = th.tensor(S, dtype=dtype, device=device)
    return Q, S

def low_rank_evd_grbf_keops(X, h2, num_eig, device=None, eps=0): #TODO
    import torch as xp
    import cupyx.scipy.sparse.linalg as cussp_lg
    import cupy as cp
    import pykeops
    pykeops.set_verbose(False)
    from pykeops.torch import Genred, LazyTensor

    device = xp.device(X.device if device is None else device)
    thdtype = X.dtype
    cpdtype = eval(f'cp.{str(thdtype).split('.')[-1]}')

    M, D= X.shape
    k = min(M-1, num_eig)
    genred = Genred(f'Exp(-SqDist(Xi, Xj) * H ) * Vj',
                     [f'Xi = Vi({D})', f'Xj = Vj({D})', 
                      'H = Pm(1)', 'Vj = Vj(1)' ],
                     reduction_op='Sum', axis=1, 
                     dtype_acc='float64',
                    #  use_double_acc=True,
                       )
    H = xp.tensor([1.0/h2], device=device, dtype=thdtype)

    with cp.cuda.Device(device.index):
    # cp.cuda.Device(device.index).use()
        def matvec(x):
            K = genred(X, X, H, xp.as_tensor(x)).squeeze(1)
            return cp.asarray(K)

        lo = cussp_lg.LinearOperator((M,M), matvec,  dtype=cp.float64)
        S, Q = cussp_lg.eigsh(lo, k=k, which='LM', maxiter = M*10, tol=eps)

        print(S)
        eig_indices = cp.argsort(-cp.abs(S))[:k]
        Q = cp.real(Q[:, eig_indices])  # eigenvectors
        S = cp.real(S[eig_indices])  # eigenvalues.

    Q = xp.as_tensor(Q)
    S = xp.as_tensor(S)
    return Q, S

def low_rank_svd_grbf(X, Y, h2, num_eig, sw_h=0.0, use_keops=True, 
                      eps =1e-10):
    if hasattr(X, 'detach'):
        is_tensor = True
        device = X.device
        dtype = X.dtype
    else:
        is_tensor = False
        device = None
        dtype = None

    # X = X.copy().astype(np.float32)
    N, (M, D) = X.shape[0], Y.shape
    k = min(M-1, N-1, num_eig)

    if use_keops:
        import pykeops
        import torch as th
        pykeops.set_verbose(False)
        from pykeops.torch import Genred
        genred = Genred(f'Exp(-SqDist(Xi, Xj) ) * Vj',
            [f'Xi = Vi({D})', f'Xj = Vj({D})', 'Vj = Vj(1)' ],
            reduction_op='Sum', axis=1, 
            # dtype_acc='float64',
            use_double_acc=True,
        )
        H = (1.0/th.asarray(h2, device=device, dtype=dtype))**0.5
        X = th.asarray(X, dtype=dtype, device=device) * H
        Y = th.asarray(Y, dtype=dtype, device=device) * H
        def matvec(x):
            mu = genred(X, Y, th.tensor(x, dtype=dtype, device=device)).squeeze(1)
            return mu.detach().cpu().numpy()
        def rmatvec(y):
            nu = genred(Y, X, th.tensor(y, dtype=dtype, device=device)).squeeze(1)
            return nu.detach().cpu().numpy()
    else:
        from ...third_party._ifgt_warp import GaussTransform_fgt
        H = (1.0/np.array(h2))**0.5
        X = X.detach().cpu().numpy() * H
        Y = Y.detach().cpu().numpy() * H
        tran1 = GaussTransform_fgt(X, 1.0, sw_h=sw_h, eps=eps)
        tran2 = GaussTransform_fgt(Y, 1.0, sw_h=sw_h, eps=eps)
        def matvec(x):
            return tran2.compute(X, x.T)
        def rmatvec(y):
            return tran1.compute(Y, y.T)


    lo = ssp.linalg.LinearOperator((N,M), matvec, rmatvec)
    U, S, Vh = ssp.linalg.svds(lo, k=k, which='LM')

    eig_indices = np.argsort(-np.abs(S))[:k]
    U = np.real(U[:, eig_indices])
    S = np.real(S[eig_indices])
    Vh = np.real(Vh[eig_indices,:])
    if is_tensor:
        import torch as th
        U = th.tensor(U, dtype=dtype, device=device)
        S = th.tensor(S, dtype=dtype, device=device)
        Vh = th.tensor(Vh, dtype=dtype, device=device)
    return U, S, Vh

def low_rank_evd(G, num_eig, xp=None):
    if hasattr(G, 'detach'):
        import torch as th
        is_tensor = True
        device = G.device
        dtype = G.dtype
        xp = th
    else:
        is_tensor = False
        xp = np if xp is None else xp
    if G.shape[0] == G.shape[1]:
        S, Q = xp.linalg.eigh(G) #only for symmetric 
        eig_indices = xp.argsort(-xp.abs(S))[:num_eig]
        Q = Q[:, eig_indices]
        S = S[eig_indices]
        if is_tensor:
            Q = Q.clone().to(dtype).to(device)
            S = S.clone().to(dtype).to(device)
        return Q, S
    else:
        U, S, Vh = xp.linalg.svd(G)
        eig_indices = xp.argsort(-xp.abs(S))[:num_eig]
        U = U[:, eig_indices]
        S = S[eig_indices]
        Vh = Vh[eig_indices,:]
        if is_tensor:
            U = U.clone().to(dtype).to(device)
            S = S.clone().to(dtype).to(device)
            Vh = Vh.clone().to(dtype).to(device)
        return U, S, Vh

def compute_svd( X, Y, h2, num_eig, use_ifgt = 1e7, use_keops=True, xp=None):
    if use_keops:
        U, S, Vh = low_rank_svd_grbf(X, Y, h2, num_eig, use_keops=use_keops)
    else:
        if (X.shape[0] * Y.shape[0]) >= use_ifgt:
            U, S, Vh = low_rank_svd_grbf(X, Y, h2, num_eig, use_keops=use_keops)
        else:
            H = xp.tensor(h2)**0.5
            G = xp.cdist(X/H, Y/H)
            G.pow_(2).mul_(-1.0)
            G.exp_()
            U, S, Vh = low_rank_evd(G, num_eig)
        return U, S, Vh

def compute_evd(X, h2, num_eig, use_ifgt = 1e9, use_keops=True, xp=None):
    if use_keops:
        Q, S = low_rank_evd_grbf(X, h2, num_eig, use_keops=use_keops)
    else:
        if (X.shape[0] **2) >= use_ifgt:
            Q, S = low_rank_evd_grbf(X, h2, num_eig, use_keops=use_keops)
        else:
            H = xp.tensor(h2)**0.5
            G = xp.cdist(X/H, X/H)
            G.pow_(2).mul_(-1.0)
            G.exp_()
            Q, S = low_rank_evd(G, num_eig)
        return Q, S

def WoodburyC(Av, U, Cv, V, xp=np):
    UCv = xp.linalg.inv(Cv  + Av * (V @ U))
    Fc = -(Av * Av) * (U @ UCv @ V)
    Fc.diagonal().add_(Av)
    return Fc

def WoodburyB(Av, U, Cv, V, xp=np):
    UCv = xp.linalg.inv(Cv  + V @ Av @ U)
    return  Av - (Av @ U) @ UCv @ (V @ Av)

def WoodburyA(A, U, C, V, xp=np):
    Av = xp.linalg.inv(A)
    Cv = xp.linalg.inv(C)
    UCv = xp.linalg.inv(Cv  + V @ Av @ U)
    return  Av - (Av @ U) @ UCv @ (V @ Av)

def spsparse_to_thsparse(X):
    import torch as th
    XX = X.tocoo()
    values = XX.data
    indices = np.vstack((XX.row, XX.col))
    i = th.LongTensor(indices)
    v = th.tensor(values, dtype=th.float64)
    shape = th.Size(XX.shape)
    return th.sparse_coo_tensor(i, v, shape)

def thsparse_to_spsparse(X):
    XX = X.to_sparse_coo().coalesce()
    values = XX.values().detach().cpu().numpy()
    indices = XX.indices().detach().cpu().numpy()
    shape = XX.shape
    return ssp.csr_array((values, indices), shape=shape)

def centerlize(X, Xm=None, Xs=None, device=None, xp = th):
    device = X.device if device is None else device
    if X.is_sparse: 
        X = X.to_dense()

    X = X.clone().to(device)
    N,D = X.shape
    Xm = xp.mean(X, 0) if Xm is None else Xm.to(device)

    X -= Xm
    Xs = xp.sqrt(xp.sum(xp.square(X))/(N*D/2)) if Xs is None else Xs.to(device) # N
    X /= Xs
    Xf = xp.eye(D+1, dtype=X.dtype, device=device)
    Xf[:D,:D] *= Xs
    Xf[:D, D] = Xm
    return [X, Xm, Xs, Xf]