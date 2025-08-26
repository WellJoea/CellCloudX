import numpy as np
def trim_points_from_mesh(mesh, points, hash_resolution=512):
    intersector = MeshIntersector(mesh, hash_resolution)
    contains = intersector.query(points)
    # del intersector
    return contains

class MeshIntersector:
    # https://github.com/autonomousvision/occupancy_networks/tree/master
    def __init__(self, mesh, resolution=512):
        if hasattr(mesh, 'vertices'):
            triangles = mesh.vertices[mesh.faces].astype(np.float64)
        elif hasattr(mesh, 'points'):
            faces = mesh.faces.reshape((mesh.n_faces_strict, 4))[:, 1:]
            triangles = mesh.points[faces].astype(np.float64)
        else:
            raise ValueError('mesh must have vertices or points attribute')
        n_tri = triangles.shape[0]

        self.resolution = resolution
        self.bbox_min = triangles.reshape(3 * n_tri, 3).min(axis=0)
        self.bbox_max = triangles.reshape(3 * n_tri, 3).max(axis=0)
        # Tranlate and scale it to [0.5, self.resolution - 0.5]^3
        self.scale = (resolution - 1) / (self.bbox_max - self.bbox_min)
        self.translate = 0.5 - self.scale * self.bbox_min

        self._triangles = triangles = self.rescale(triangles)
        # assert(np.allclose(triangles.reshape(-1, 3).min(0), 0.5))
        # assert(np.allclose(triangles.reshape(-1, 3).max(0), resolution - 0.5))

        triangles2d = triangles[:, :, :2]
        self._tri_intersector2d = TriangleIntersector2d(
            triangles2d, resolution)

    def query(self, points):
        # Rescale points
        points = self.rescale(points)

        # placeholder result with no hits we'll fill in later
        contains = np.zeros(len(points), dtype=np.bool_)

        # cull points outside of the axis aligned bounding box
        # this avoids running ray tests unless points are close
        inside_aabb = np.all(
            (0 <= points) & (points <= self.resolution), axis=1)
        if not inside_aabb.any():
            return contains

        # Only consider points inside bounding box
        mask = inside_aabb
        points = points[mask]

        # Compute intersection depth and check order
        points_indices, tri_indices = self._tri_intersector2d.query(points[:, :2])

        triangles_intersect = self._triangles[tri_indices]
        points_intersect = points[points_indices]

        depth_intersect, abs_n_2 = self.compute_intersection_depth(
            points_intersect, triangles_intersect)

        # Count number of intersections in both directions
        smaller_depth = depth_intersect >= points_intersect[:, 2] * abs_n_2
        bigger_depth = depth_intersect < points_intersect[:, 2] * abs_n_2
        points_indices_0 = points_indices[smaller_depth]
        points_indices_1 = points_indices[bigger_depth]

        nintersect0 = np.bincount(points_indices_0, minlength=points.shape[0])
        nintersect1 = np.bincount(points_indices_1, minlength=points.shape[0])
        
        # Check if point contained in mesh
        contains1 = (np.mod(nintersect0, 2) == 1)
        contains2 = (np.mod(nintersect1, 2) == 1)
        if (contains1 != contains2).any():
            print('Warning: contains1 != contains2 for some points.')
        contains[mask] = (contains1 & contains2)
        return contains

    def compute_intersection_depth(self, points, triangles):
        t1 = triangles[:, 0, :]
        t2 = triangles[:, 1, :]
        t3 = triangles[:, 2, :]

        v1 = t3 - t1
        v2 = t2 - t1
        # v1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
        # v2 = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)

        normals = np.cross(v1, v2)
        alpha = np.sum(normals[:, :2] * (t1[:, :2] - points[:, :2]), axis=1)

        n_2 = normals[:, 2]
        t1_2 = t1[:, 2]
        s_n_2 = np.sign(n_2)
        abs_n_2 = np.abs(n_2)

        mask = (abs_n_2 != 0)
    
        depth_intersect = np.full(points.shape[0], np.nan)
        depth_intersect[mask] = \
            t1_2[mask] * abs_n_2[mask] + alpha[mask] * s_n_2[mask]

        # Test the depth:
        # TODO: remove and put into tests
        # points_new = np.concatenate([points[:, :2], depth_intersect[:, None]], axis=1)
        # alpha = (normals * t1).sum(-1)
        # mask = (depth_intersect == depth_intersect)
        # assert(np.allclose((points_new[mask] * normals[mask]).sum(-1),
        #                    alpha[mask]))
        return depth_intersect, abs_n_2

    def rescale(self, array):
        array = self.scale * array + self.translate
        return array

class TriangleIntersector2d:
    def __init__(self, triangles, resolution=128):
        self.triangles = triangles
        self.tri_hash = _TriangleHash(triangles, resolution)

    def query(self, points):
        point_indices, tri_indices = self.tri_hash.query(points)
        point_indices = np.array(point_indices, dtype=np.int64)
        tri_indices = np.array(tri_indices, dtype=np.int64)
        points = points[point_indices]
        triangles = self.triangles[tri_indices]
        mask = self.check_triangles(points, triangles)
        point_indices = point_indices[mask]
        tri_indices = tri_indices[mask]
        return point_indices, tri_indices

    def check_triangles(self, points, triangles):
        contains = np.zeros(points.shape[0], dtype=np.bool_)
        A = triangles[:, :2] - triangles[:, 2:]
        A = A.transpose([0, 2, 1])
        y = points - triangles[:, 2]

        detA = A[:, 0, 0] * A[:, 1, 1] - A[:, 0, 1] * A[:, 1, 0]
        
        mask = (np.abs(detA) != 0.)
        A = A[mask]
        y = y[mask]
        detA = detA[mask]

        s_detA = np.sign(detA)
        abs_detA = np.abs(detA)

        u = (A[:, 1, 1] * y[:, 0] - A[:, 0, 1] * y[:, 1]) * s_detA
        v = (-A[:, 1, 0] * y[:, 0] + A[:, 0, 0] * y[:, 1]) * s_detA

        sum_uv = u + v
        contains[mask] = (
            (0 < u) & (u < abs_detA) & (0 < v) & (v < abs_detA)
            & (0 < sum_uv) & (sum_uv < abs_detA)
        )
        return contains

class _TriangleHash:
    def __init__(self, triangles, resolution):
        self.resolution = resolution
        self._build_hash(triangles)

    def _build_hash(self, triangles):
        assert(triangles.shape[1] == 3)
        assert(triangles.shape[2] == 2)

        bboxes_min = triangles.min(1).clip(0, self.resolution -1).astype(np.int64)
        bboxes_max = triangles.max(1).clip(0, self.resolution -1).astype(np.int64)

        def spatial_local(bbox_min, bbox_max, res):
            x = np.arange(bbox_min[0], bbox_max[0]+1) 
            y = np.arange(bbox_min[1], bbox_max[1]+1)[:,np.newaxis]
            return (res*x + y).flatten()

        spatial_dict = {}
        for i_tri in range(triangles.shape[0]):
            is_idx = spatial_local(bboxes_min[i_tri], bboxes_max[i_tri], self.resolution)
            for ikey in is_idx:
                if ikey in spatial_dict:
                    spatial_dict[ikey].append(i_tri)
                else:
                    spatial_dict[ikey] = [i_tri]
        self.spatial_dict =spatial_dict

    def query(self, points):
        assert(points.shape[1] == 2)
        points = points.copy().astype(np.int64)

        keep_idx = ((points[:,0] >= 0) & 
                     (points[:,0] < self.resolution) & 
                     (points[:,1] >= 0) & 
                     (points[:,1] < self.resolution))

        spatial_idx = self.resolution * points[:,0] +  points[:,1]
        keep_point = np.arange(points.shape[0])[keep_idx]
        tri_indices = [self.spatial_dict.get(spatial_idx[idx], []) for idx in keep_point ]
        tri_len = list(map(len, tri_indices))
        points_indices = np.repeat(keep_point, tri_len)
        tri_indices = np.concatenate(tri_indices)
        return points_indices, tri_indices