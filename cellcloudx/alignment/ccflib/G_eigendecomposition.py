from tqdm import tqdm
import itertools

from .xp_utility import compute_svd, compute_eigen, low_rank_eigen_grbf, low_rank_svd_grbf, low_rank_eigen, WoodburyB, WoodburyC

import torch
import torch.multiprocessing as mp
from tqdm import tqdm
import itertools
import numpy as np

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
                    Q, S = self._compute_eigen(X, h2, device)
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
                return self._low_rank_eigen(G)

    def _compute_eigen(self, X, h2, device):
        """设备感知的特征分解"""
        with torch.cuda.device(device):
            if (X.shape[0] ** 2) >= self.low_rank_g:
                return self._low_rank_eigen_grbf(X, h2**0.5)
            else:
                G = torch.cdist(X, X)
                G.pow_(2).div_(-h2).exp_()
                return self._low_rank_eigen(G)

    def _monitor_progress(self, processes, total_tasks):
        """监控进度与异常处理"""
        with tqdm(total=total_tasks, desc="Parallel Computing", 
                 disable=not self.verbose) as pbar:
            while any(p.is_alive() for p in processes):
                current = total_tasks - self._count_remaining_tasks(processes)
                pbar.update(current - pbar.n)
                time.sleep(0.1)
                
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
        
    def _low_rank_eigen(self, G):
        """基于随机投影的特征分解"""
        # 实现细节...
        
    def _low_rank_eigen_grbf(self, X, gamma):
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
                    desc="Gs eigendecomposition",
                    colour='#AAAAAA', 
                    disable=(self.verbose==0)) as pbar:
            for i, j in itertools.product(range(L), range(L)):
                pbar.set_postfix(dict(i=int(i), j=int(j)))
                pbar.update()
                if i==j:
                    (Q, S) = compute_eigen(Ys[i], h2s[i], self.num_eig, 
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
                desc="Gs eigendecomposition",
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
        (Q, S) = compute_eigen(Y, h2, self.num_eig, 
                                use_ifgt=self.use_ifgt, xp=self.xp)
        setattr(self, f'Q{i}{j}', Q)
        setattr(self, f'S{i}{j}', S)

    def _process_svd_task(self, X, Y, h2, i, j):
        (U, S, Vh) = compute_svd(X, Y, h2, self.num_eig, 
                            use_ifgt=self.use_ifgt/100, xp=self.xp)
        setattr(self, f'US{i}{j}', U * S)
        setattr(self, f'Vh{i}{j}', Vh)
