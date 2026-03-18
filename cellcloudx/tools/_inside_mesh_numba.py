import numpy as np

try:
    from numba import njit, prange
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

def trim_points_from_mesh_numba(mesh, points, hash_resolution=512):
    intersector = MeshIntersector(mesh, resolution=hash_resolution)
    return intersector.query(points)

class MeshIntersector:
    def __init__(self, mesh, resolution=512):
        if hasattr(mesh, 'vertices'):
            triangles = mesh.vertices[mesh.faces].astype(np.float64, copy=False)
        elif hasattr(mesh, 'points'):
            faces = mesh.faces.reshape((mesh.n_faces_strict, 4))[:, 1:]
            triangles = mesh.points[faces].astype(np.float64, copy=False)
        else:
            raise ValueError('mesh must have vertices or points attribute')

        self.resolution = int(resolution)

        n_tri = triangles.shape[0]
        bbox_min = triangles.reshape(3 * n_tri, 3).min(axis=0)
        bbox_max = triangles.reshape(3 * n_tri, 3).max(axis=0)
        scale = (resolution - 1.0) / (bbox_max - bbox_min + 1e-12)
        translate = 0.5 - scale * bbox_min

        self.scale = scale
        self.translate = translate

        triangles = self._rescale(triangles)
        self._triangles = triangles
        self._triangles2d = triangles[:, :, :2].copy()

        # t1, t2, t3: (T, 3)
        t1 = triangles[:, 0, :]
        t2 = triangles[:, 1, :]
        t3 = triangles[:, 2, :]

        v1 = t3 - t1
        v2 = t2 - t1
        normals = np.cross(v1, v2)                    # (T, 3)
        nxy = normals[:, :2].copy()                   # (T, 2)
        n2 = normals[:, 2].copy()                     # (T,)

        s_n2 = np.sign(n2).astype(np.float64)         # (T,)
        abs_n2 = np.abs(n2)                            # (T,)
        t1_2_abs = t1[:, 2] * abs_n2                  # (T,)
        c_xy = (nxy * t1[:, :2]).sum(-1)              # (T,)


        valid_tri = (abs_n2 > 0)

        self._pre_nxy = nxy
        self._pre_cxy = c_xy
        self._pre_s_n2 = s_n2
        self._pre_abs_n2 = abs_n2
        self._pre_t1_2_abs = t1_2_abs
        self._valid_tri = valid_tri


        self._tri_intersector2d = TriangleIntersector2d(self._triangles2d, resolution)

    def _rescale(self, array):
        return self.scale * array + self.translate

    def query(self, points):
        pts = np.asarray(points, dtype=np.float64, order='C')
        pts_scaled = self._rescale(pts)

        contains = np.zeros(pts.shape[0], dtype=np.bool_)

        inside_aabb = np.all((pts_scaled >= 0.0) & (pts_scaled <= self.resolution), axis=1)
        if not inside_aabb.any():
            return contains

        mask = inside_aabb
        pts_in = pts_scaled[mask]                       # (M, 3)

        pidx, tidx = self._tri_intersector2d.query(pts_in[:, :2])  # pidx,tidx shape=(K,)

        if pidx.size == 0:
            contains[mask] = False
            return contains

        if self._valid_tri is not None:
            vmask = self._valid_tri[tidx]
            if not vmask.all():
                pidx = pidx[vmask]
                tidx = tidx[vmask]
                if pidx.size == 0:
                    contains[mask] = False
                    return contains

        inside2d = _pairs_inside2d(pts_in[pidx, :2], self._triangles2d[tidx])
        if not inside2d.any():
            contains[mask] = False
            return contains

        pidx = pidx[inside2d]
        tidx = tidx[inside2d]

        # 预计算常数项参与计算：depth = t1_2_abs + (c_xy - <nxy, p_xy>) * sgn(nz)
        depths, abs_n2 = _pairs_depth(
            pts_in[pidx], self._pre_nxy, self._pre_cxy, self._pre_s_n2, self._pre_t1_2_abs, self._pre_abs_n2, tidx
        )

        contains_sub = _parity_count(pts_in[:, 2], pidx, depths, abs_n2)

        contains[mask] = contains_sub
        return contains


class TriangleIntersector2d:
    def __init__(self, triangles, resolution=128):
        self.resolution = int(resolution)
        self.triangles = np.ascontiguousarray(triangles, dtype=np.float64)
        if self.triangles.ndim != 3 or self.triangles.shape[1:] != (3, 2):
            raise ValueError("triangles must be of shape (T,3,2)")


        if _HAS_NUMBA:
            self.offsets, self.tri_by_cell = _build_hash_numba(self.triangles, self.resolution)
        else:
            self.offsets, self.tri_by_cell = _build_hash_numpy(self.triangles, self.resolution)

    def query(self, points2d):
        pts = np.ascontiguousarray(points2d, dtype=np.float64)
        R = self.resolution

        xi = np.floor(pts[:, 0]).astype(np.int64)
        yi = np.floor(pts[:, 1]).astype(np.int64)
        keep = (xi >= 0) & (xi < R) & (yi >= 0) & (yi < R)
        if not keep.any():
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)

        xi = xi[keep]
        yi = yi[keep]
        base_idx = np.nonzero(keep)[0]                  # 原始点的索引

        cell_id = xi * R + yi                           # (M,)

        if _HAS_NUMBA:
            pidx, tidx = _hash_query_pairs_numba(base_idx, cell_id, self.offsets, self.tri_by_cell)
        else:
            pidx, tidx = _hash_query_pairs_numpy(base_idx, cell_id, self.offsets, self.tri_by_cell)

        return pidx, tidx


if _HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _build_hash_numba(triangles, R):
        T = triangles.shape[0]
        ncell = R * R

        # 第一遍：统计每个 cell 的三角形数量
        count = np.zeros(ncell, dtype=np.int64)
        for i in range(T):
            tri = triangles[i]
            # bbox (包含端点)
            xmin = int(max(0, min(tri[0, 0], tri[1, 0], tri[2, 0])))
            xmax = int(min(R - 1, max(tri[0, 0], tri[1, 0], tri[2, 0])))
            ymin = int(max(0, min(tri[0, 1], tri[1, 1], tri[2, 1])))
            ymax = int(min(R - 1, max(tri[0, 1], tri[1, 1], tri[2, 1])))

            for x in range(xmin, xmax + 1):
                row = x * R
                for y in range(ymin, ymax + 1):
                    count[row + y] += 1

        # 前缀和得到 offsets
        offsets = np.empty(ncell + 1, dtype=np.int64)
        offsets[0] = 0
        for i in range(ncell):
            offsets[i + 1] = offsets[i] + count[i]

        tri_by_cell = np.empty(offsets[-1], dtype=np.int64)
        # 游标拷贝（重用 count 作写指针）
        for i in range(ncell):
            count[i] = offsets[i]

        # 第二遍：填充
        for i in range(T):
            tri = triangles[i]
            xmin = int(max(0, min(tri[0, 0], tri[1, 0], tri[2, 0])))
            xmax = int(min(R - 1, max(tri[0, 0], tri[1, 0], tri[2, 0])))
            ymin = int(max(0, min(tri[0, 1], tri[1, 1], tri[2, 1])))
            ymax = int(min(R - 1, max(tri[0, 1], tri[1, 1], tri[2, 1])))

            for x in range(xmin, xmax + 1):
                row = x * R
                for y in range(ymin, ymax + 1):
                    cid = row + y
                    pos = count[cid]
                    tri_by_cell[pos] = i
                    count[cid] = pos + 1

        return offsets, tri_by_cell

    @njit(cache=True, fastmath=True)
    def _hash_query_pairs_numba(base_idx, cell_id, offsets, tri_by_cell):
        M = cell_id.shape[0]
        # 先算总对数，预分配
        total = 0
        for i in range(M):
            cid = cell_id[i]
            total += offsets[cid + 1] - offsets[cid]

        pidx = np.empty(total, dtype=np.int64)
        tidx = np.empty(total, dtype=np.int64)

        k = 0
        for i in range(M):
            cid = cell_id[i]
            s = offsets[cid]
            e = offsets[cid + 1]
            n = e - s
            # 写入 tri 索引
            for j in range(n):
                tidx[k + j] = tri_by_cell[s + j]
            # point 索引重复写入
            b = base_idx[i]
            for j in range(n):
                pidx[k + j] = b
            k += n
        return pidx, tidx

    @njit(cache=True, fastmath=True)
    def _pairs_inside2d(points_xy, triangles2d):
        # oriented-edge 测试，允许在边上（eps 容忍）
        eps = 1e-12
        K = points_xy.shape[0]
        out = np.zeros(K, dtype=np.uint8)
        for i in range(K):
            p0x = triangles2d[i, 0, 0]; p0y = triangles2d[i, 0, 1]
            p1x = triangles2d[i, 1, 0]; p1y = triangles2d[i, 1, 1]
            p2x = triangles2d[i, 2, 0]; p2y = triangles2d[i, 2, 1]
            x = points_xy[i, 0]; y = points_xy[i, 1]

            # cross(z) of edges with vector to point
            c0 = (p1x - p0x) * (y - p0y) - (p1y - p0y) * (x - p0x)
            c1 = (p2x - p1x) * (y - p1y) - (p2y - p1y) * (x - p1x)
            c2 = (p0x - p2x) * (y - p2y) - (p0y - p2y) * (x - p2x)

            has_pos = (c0 > eps) or (c1 > eps) or (c2 > eps)
            has_neg = (c0 < -eps) or (c1 < -eps) or (c2 < -eps)
            # inside 当且仅当三者同号或零（不同时出现正负）
            out[i] = 0 if (has_pos and has_neg) else 1
        return out.view(np.bool_)

    @njit(cache=True, fastmath=True)
    def _pairs_depth(pts3d, nxy, cxy, s_n2, t1_2_abs, abs_n2, tidx):
        K = pts3d.shape[0]
        depth = np.empty(K, dtype=np.float64)
        abnz = np.empty(K, dtype=np.float64)

        for i in range(K):
            ti = tidx[i]
            # alpha = c_xy - <n_xy, p_xy>
            alpha = cxy[ti] - (nxy[ti, 0] * pts3d[i, 0] + nxy[ti, 1] * pts3d[i, 1])
            depth[i] = t1_2_abs[ti] + alpha * s_n2[ti]
            abnz[i] = abs_n2[ti]
        return depth, abnz

    @njit(cache=True, fastmath=True)
    def _parity_count(pts_z, pair_pidx, depth, abs_n2):
        M = pts_z.shape[0]
        cnt0 = np.zeros(M, dtype=np.int64)
        cnt1 = np.zeros(M, dtype=np.int64)

        for i in range(pair_pidx.shape[0]):
            pj = pair_pidx[i]
            # 两个方向的分组
            if depth[i] >= pts_z[pj] * abs_n2[i]:
                cnt0[pj] += 1
            else:
                cnt1[pj] += 1

        out = np.zeros(M, dtype=np.uint8)
        for j in range(M):
            odd0 = (cnt0[j] & 1) == 1
            odd1 = (cnt1[j] & 1) == 1
            out[j] = 1 if (odd0 and odd1) else 0
        return out.view(np.bool_)

def _build_hash_numpy(triangles, R):
    T = triangles.shape[0]
    ncell = R * R
    # bbox（整型网格）
    xmin = np.clip(np.floor(triangles[:, :, 0].min(1)), 0, R - 1).astype(np.int64)
    xmax = np.clip(np.floor(triangles[:, :, 0].max(1)), 0, R - 1).astype(np.int64)
    ymin = np.clip(np.floor(triangles[:, :, 1].min(1)), 0, R - 1).astype(np.int64)
    ymax = np.clip(np.floor(triangles[:, :, 1].max(1)), 0, R - 1).astype(np.int64)

    # 逐三角形累加 cell 计数（Python 循环，但仅 4*尺寸，较快）
    count = np.zeros(ncell, dtype=np.int64)
    for i in range(T):
        xs = np.arange(xmin[i], xmax[i] + 1, dtype=np.int64)
        ys = np.arange(ymin[i], ymax[i] + 1, dtype=np.int64)
        # 注意：这里使用行主序 cell_id = x*R + y（与 Numba 版本一致）
        cell_ids = (xs[:, None] * R + ys[None, :]).ravel()
        np.add.at(count, cell_ids, 1)

    offsets = np.empty(ncell + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(count, out=offsets[1:])

    tri_by_cell = np.empty(offsets[-1], dtype=np.int64)
    cursor = offsets[:-1].copy()

    for i in range(T):
        xs = np.arange(xmin[i], xmax[i] + 1, dtype=np.int64)
        ys = np.arange(ymin[i], ymax[i] + 1, dtype=np.int64)
        cell_ids = (xs[:, None] * R + ys[None, :]).ravel()
        pos = cursor[cell_ids]
        tri_by_cell[pos] = i
        np.add.at(cursor, cell_ids, 1)

    return offsets, tri_by_cell

def _hash_query_pairs_numpy(base_idx, cell_id, offsets, tri_by_cell):
    # 为每个点取切片长度，做前缀和，集中分配
    seglen = offsets[cell_id + 1] - offsets[cell_id]
    pos = np.empty_like(seglen)
    np.cumsum(seglen, out=pos)
    total = int(pos[-1])
    pidx = np.empty(total, dtype=np.int64)
    tidx = np.empty(total, dtype=np.int64)

    start = 0
    for i, cid in enumerate(cell_id):
        s, e = int(offsets[cid]), int(offsets[cid + 1])
        n = e - s
        if n <= 0:
            continue
        pidx[start:start + n] = base_idx[i]
        tidx[start:start + n] = tri_by_cell[s:e]
        start += n
    return pidx, tidx

def _pairs_inside2d(points_xy, triangles2d):
    p0 = triangles2d[:, 0, :]
    p1 = triangles2d[:, 1, :]
    p2 = triangles2d[:, 2, :]
    v0 = p1 - p0
    v1 = p2 - p1
    v2 = p0 - p2
    w0 = points_xy - p0
    w1 = points_xy - p1
    w2 = points_xy - p2
    c0 = v0[:, 0] * w0[:, 1] - v0[:, 1] * w0[:, 0]
    c1 = v1[:, 0] * w1[:, 1] - v1[:, 1] * w1[:, 0]
    c2 = v2[:, 0] * w2[:, 1] - v2[:, 1] * w2[:, 0]
    eps = 1e-12
    has_pos = (c0 > eps) | (c1 > eps) | (c2 > eps)
    has_neg = (c0 < -eps) | (c1 < -eps) | (c2 < -eps)
    return ~(has_pos & has_neg)

def _pairs_depth(pts3d, nxy, cxy, s_n2, t1_2_abs, abs_n2, tidx):
    alpha = cxy[tidx] - (nxy[tidx] * pts3d[:, :2]).sum(axis=1)
    depth = t1_2_abs[tidx] + alpha * s_n2[tidx]
    abnz = abs_n2[tidx]
    return depth, abnz

def _parity_count(pts_z, pair_pidx, depth, abs_n2):
    smaller = depth >= pts_z[pair_pidx] * abs_n2
    bigger = ~smaller
    n = pts_z.shape[0]
    c0 = np.bincount(pair_pidx[smaller], minlength=n) & 1
    c1 = np.bincount(pair_pidx[bigger], minlength=n) & 1
    return (c0 == 1) & (c1 == 1)