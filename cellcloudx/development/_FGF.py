import numpy as np

def get_canonical_simplex(dim):
    """
    Compute the canonical simplex for a given dimension.
    """
    canonical = np.zeros((dim + 1, dim + 1), dtype=int)
    for i in range(dim + 1):
        canonical[i, :dim + 1 - i] = i
        canonical[i, dim + 1 - i:] = i - (dim + 1)
    return canonical

def get_projection_matrix(dim):
    """
    Compute the projection matrix for a given dimension.
    """
    e_left_u = np.concatenate((np.triu(np.ones((dim, dim), dtype=np.float32)), np.zeros((1, dim), dtype=np.float32)), axis=0)
    e_left_d = np.concatenate((np.zeros((1, dim), dtype=np.float32), -np.diag(np.arange(dim, dtype=np.float32) + 1)), axis=0)
    e_left = e_left_u + e_left_d
    e_right = np.diag(1 / np.sqrt(np.arange(1, dim + 1, dtype=np.float32) * np.arange(2, dim + 2, dtype=np.float32)))
    e = np.dot(e_left, e_right)
    e *= np.sqrt(2.0 / 3.0) * (dim + 1)
    return e

def compute_rzp_rank(features, dim):
    """
    Compute the reminder-zero-point rank for features.
    """
    reminder_zero_points = np.rint(features / (dim + 1)).astype('int32') * (dim + 1)
    rank = np.argsort(np.argsort(-(features - reminder_zero_points.astype(np.float32)).astype(np.float32), axis=-1), axis=-1)
    sum_rzp = (reminder_zero_points / (dim + 1)).sum(axis=-1).astype(np.int32)

    rank += sum_rzp[:, :, None]
    # Handle boundary conditions
    rank[rank < 0] += dim + 1
    reminder_zero_points[rank < 0] += dim + 1
    reminder_zero_points[dim + 1 <= rank] -= dim + 1
    rank[dim + 1 <= rank] -= dim + 1

    return reminder_zero_points, rank

def compute_weights(features, reminder_zero_points, dim):
    """
    Compute barycentric weights for the features.
    """
    y = np.sort(features - reminder_zero_points, axis=-1)[:, :, ::-1]
    b = (y[:, :, :-1] - y[:, :, 1:])[:, :, ::-1] / (dim + 1)
    b = np.concatenate(((1 - b.sum(axis=-1))[:, :, None], b), axis=-1)
    return b

import numpy as np
from scipy.spatial import cKDTree
class HashMap1:
    def __init__(self, data, initial_size=2**16, load_factor=0.75):
        """
        Initialize the HashMap with the given data.
        Args:
            data: Input data, shape (batch_size, num_points, dim).
            initial_size: Initial hash table size.
            load_factor: Maximum load factor before resizing.
        """
        self.batch_size, self.num_points, self.dim = data.shape
        self.hash_tables = []  # Store hash tables (one per batch)
        self.load_factor = load_factor
        self.max_size = initial_size

        # Initialize hash tables for each batch
        for b in range(self.batch_size):
            points = data[b]
            self.hash_tables.append(self._build_kdtree(points))

    def _build_kdtree(self, points):
        """Build a KD-tree for efficient spatial indexing."""
        return cKDTree(points)

    def find_exact(self, query_points):
        """
        Find exact matches for query points.
        Args:
            query_points: Points to search for, shape (batch_size, num_points, dim).
        Returns:
            Indices of matching points in the original data, or -1 if not found.
        """
        results = []
        for b in range(self.batch_size):
            tree = self.hash_tables[b]
            points = query_points[b]

            # Find exact matches within a small tolerance
            indices = tree.query_ball_point(points, r=1e-8)
            batch_results = [
                i[0] if len(i) > 0 else -1 for i in indices
            ]
            results.append(batch_results)

        return np.array(results, dtype=np.int32)

    def find_nearest(self, query_points):
        """
        Find the nearest neighbors for query points.
        Args:
            query_points: Points to search for, shape (batch_size, num_points, dim).
        Returns:
            Indices of nearest points in the original data.
        """
        results = []
        for b in range(self.batch_size):
            tree = self.hash_tables[b]
            points = query_points[b]

            # Query for nearest neighbors
            _, indices = tree.query(points)
            results.append(indices)

        return np.array(results, dtype=np.int32)

    def insert(self, batch_idx, new_points):
        """
        Dynamically insert new points into the hash table.
        Args:
            batch_idx: Index of the batch to update.
            new_points: Points to insert, shape (num_new_points, dim).
        """
        existing_points = self.hash_tables[batch_idx].data
        updated_points = np.vstack((existing_points, new_points))
        self.hash_tables[batch_idx] = self._build_kdtree(updated_points)

class HashMap:
    def __init__(self, data, table_size=2**24):
        self.hash_factor = 2531011
        self.batch_size, self.num_points, self.dim = data.shape
        self.table_size = table_size

        # Initialize hash table
        self.indices = -np.ones((self.batch_size, self.table_size), dtype=np.int32)
        self.values = np.zeros((self.batch_size, self.table_size, self.dim), dtype=np.int32)
        self.value_list = np.zeros((self.batch_size, self.table_size, self.dim), dtype=np.int32)
        self.size = None

        # Initialize keys in the hash table
        self.init_keys(data)

    def _compute_key(self, value):
        """Compute the hash key for a given value."""
        key = 0
        for v in value:
            key = (key + v) * self.hash_factor
        return key % self.table_size

    def init_keys(self, data):
        data = np.ascontiguousarray(data)
        used = np.zeros((self.batch_size, self.table_size), dtype=np.int32)
        written = np.zeros((self.batch_size, self.table_size), dtype=np.int32)
        count = np.zeros((self.batch_size,), dtype=np.int32)

        for i in range(data.shape[0] * data.shape[1]):
            bn = i // self.num_points  # Batch index
            value_init = data[bn, i % self.num_points]
            # Compute initial key
            key = self._compute_key(value_init)

            for _ in range(100):
                if used[bn, key] == 0:
                    # Mark as used and insert the value
                    used[bn, key] = 1
                    self.values[bn, key] = value_init
                    written[bn, key] = 1

                    # Update indices and count
                    index = count[bn]
                    count[bn] += 1
                    self.indices[bn, key] = index
                    self.value_list[bn, index] = value_init
                    break
                else:
                    # Handle conflicts with linear probing
                    if np.array_equal(self.values[bn, key], value_init):
                        break
                    key = (key + 1) % self.table_size
            else:
                raise Exception("Hash table insertion failed due to excessive collisions")

        self.size = int(count.max())

    def find(self, data):
        """Find the indices of the values in the hash table."""
        ret = -np.ones(data.shape[:-1], dtype=np.int32)
        data = np.ascontiguousarray(data)

        for i in range(data.shape[0] * data.shape[1]):
            bn = i // self.num_points  # Batch index
            value = data[bn, i % self.num_points]

            # Compute initial key
            key = self._compute_key(value)

            for _ in range(100):
                if self.indices[bn, key] < 0:
                    ret[bn, i % self.num_points] = -1
                    break
                if np.array_equal(self.values[bn, key], value):
                    ret[bn, i % self.num_points] = self.indices[bn, key]
                    break
                key = (key + 1) % self.table_size
            else:
                raise Exception("Value not found after exhaustive search")

        return ret

class Lattice:
    def __init__(self, points, hash_size=2**24):
        """
        Initialize the Lattice structure.
        Inputs:
            points: [batch_size, num_points, dim_points]
        Outputs:
            lattice_indices: [batch_size, num_points, dim_points + 1]
            barycentric_weights: [batch_size, num_points, dim_points + 1]
            lattice_indices_n1: [batch_size, num_lattice_points, dim_points + 1]
            lattice_indices_n2: [batch_size, num_lattice_points, dim_points + 1]
        """
        self.bs, self.num_points, self.dim = points.shape

        # Step 1: Compute projection matrix
        projection_matrix = get_projection_matrix(self.dim)

        # Step 2: Project points into lattice space
        points = np.dot(points, projection_matrix.T)

        # Step 3: Compute ranks and reminder zero points
        reminder_zero_points, rank = compute_rzp_rank(points, self.dim)

        # Step 4: Compute barycentric weights
        self.barycentric_weights = compute_weights(points, reminder_zero_points, self.dim)

        # Step 5: Compute lattice points and initialize hash table
        canonical = get_canonical_simplex(self.dim)
        lattice_points = canonical[:, rank].transpose((1, 2, 0, 3)) + reminder_zero_points[:, :, None, :]
        lattice_points = lattice_points.reshape((lattice_points.shape[0], -1, self.dim + 1))
        self.lattice_points = lattice_points
        print('HashMap')

        # Step 6: Generate lattice indices
        self.hash_map = HashMap(lattice_points, hash_size)
        self.lattice_indices = self.hash_map.find(lattice_points).reshape((self.bs, self.num_points, self.dim + 1))


        # self.hash_map1 = HashMap1(lattice_points, initial_size=hash_size)
        # self.lattice_indices1 = self.hash_map1.find_exact(lattice_points).reshape((self.bs, self.num_points, self.dim + 1))
        
        # Step 7: Initialize neighbor indices
        self.lattice_indices_n1 = -np.ones((self.bs, self.hash_map.size, self.dim + 1), dtype=int)
        self.lattice_indices_n2 = -np.ones((self.bs, self.hash_map.size, self.dim + 1), dtype=int)

        lattice_list = self.hash_map.value_list[:, :self.hash_map.size]
        for d in range(self.dim + 1):
            li = lattice_list.copy() - 1
            li[:, :, d] += self.dim + 1
            self.lattice_indices_n1[:, :, d] = self.hash_map.find(li)
            li = lattice_list.copy() + 1
            li[:, :, d] -= self.dim + 1
            self.lattice_indices_n2[:, :, d] = self.hash_map.find(li)

    def compute(self, features, forward=True):
        """
        Perform splatting, blurring, and slicing on features.
        Inputs:
            features: [batch_size, num_points, dim_features]
        """
        bs, num_points, dim_features = features.shape
        num_lattice_points = self.hash_map.size

        # Step 1: Splatting
        lattice_features = np.zeros((num_lattice_points, dim_features), dtype=np.float32)
        for b in range(bs):
            for p in range(num_points):
                indices = self.lattice_indices[b, p]
                weights = self.barycentric_weights[b, p]
                for idx, weight in zip(indices, weights):
                    if idx >= 0:
                        lattice_features[idx] += weight * features[b, p]

        # Step 2: Blurring
        if forward:
            order = range(self.dim)
        else:
            order = range(self.dim)[::-1]

        for i in order:
            lattice_features_new = np.copy(lattice_features) * 0.5
            lin1 = self.lattice_indices_n1[:, :, i].flatten()
            lin2 = self.lattice_indices_n2[:, :, i].flatten()

            for idx in range(num_lattice_points):
                if lin1[idx] >= 0:
                    lattice_features_new[idx] += 0.25 * lattice_features[lin1[idx]]
                if lin2[idx] >= 0:
                    lattice_features_new[idx] += 0.25 * lattice_features[lin2[idx]]
            lattice_features = lattice_features_new

        # Step 3: Slicing
        features_out = np.zeros_like(features, dtype=np.float32)
        for b in range(bs):
            for p in range(num_points):
                indices = self.lattice_indices[b, p]
                weights = self.barycentric_weights[b, p]
                features_out[b, p] = sum(
                    lattice_features[idx] * weight for idx, weight in zip(indices, weights) if idx >= 0
                )

        return features_out

class FastGaussianFilter():
    def __init__(self, lattice, normalize=True):
        self.lattice = lattice
        self.weights = None
        self.normalize = normalize

    def forward_gpu(self, features):
        # features = inputs[0]
        if self.normalize:
            features = np.concatenate((features, np.ones((features.shape[0], features.shape[1], 1), dtype=np.float32)), axis=2)
            features = self.lattice.compute(features)
            features, weights = features[:, :, :-1], features[:, :, -1]
            features = features / (weights[:, :, None] + 1e-6)
            self.weights = weights
        else:
            features = self.lattice.compute(features)
        return features

    def backward_gpu(self, inputs, grad_outputs):
        grad_output = grad_outputs[0]
        if self.normalize:
            grad_output = grad_output / self.weights[:, :, None]
        grad_input = self.lattice.compute(grad_output, forward=False)
        return grad_input

    def forward_cpu(self, inputs):
        raise NotImplementedError

    def backward_cpu(self, inputs, grad_outputs):
        raise NotImplementedError

def fast_gaussian_filter(features, points=None, lattice=None, normalize=True):
    if lattice is None:
        lattice = Lattice(points)
    F = FastGaussianFilter(lattice, normalize)
    features = F.forward_gpu(features)
    return features

import unittest
import numpy as np
import skimage 
import time
class TestFastGaussianFilter():
    def test_forward(self):
        # reference time: 827 +- 4 ms
        # this gpu computation: 104 +- 1 ms
        std_spatial = 5
        std_color = 0.125
        image_in = skimage.io.imread('../ref/in.jpg').astype('float32') / 255.
        image_ref = np.array([[float(l2) for l2 in l.strip().split(',')] for l in open('../ref/ref.log').readlines()])
        image_ref = image_ref.reshape(image_in.shape)
        y, x = np.meshgrid(np.arange(image_in.shape[0]), np.arange(image_in.shape[1]), indexing='ij')
        points = np.concatenate((x[:, :, None], y[:, :, None], image_in), axis=-1)
        points[:, :, :2] /= std_spatial
        points[:, :, 2:] /= std_color

        points = points.reshape((1, -1, 5))
        features = image_in.reshape((1, -1, 3))
        points = points.astype(np.float32)
        features = features.astype(np.float32)

        ts = time.time()
        image_out = fast_gaussian_filter(features, points=points)
        te = time.time()
        print ('time (1)', te - ts)

        for i in range(10):
            ts = time.time()
            image_out = fast_gaussian_filter(features, points=points)
            image_out = image_out.reshape(image_ref.shape)
            te = time.time()
            print ('time (2-%d)' % i, te - ts)

        lattice = Lattice(points)
        for i in range(10):
            ts = time.time()
            image_out = fast_gaussian_filter(features, lattice=lattice)
            image_out = image_out.reshape(image_ref.shape)
            te = time.time()
            print ('time (3-%d)' % i, te - ts)

        diff = image_in - image_out.data.get()
        print ('diff_in', np.square(diff).mean())
        diff = image_ref - image_out.data.get()
        print ('diff_out', np.square(diff).mean())


    def test_backward(self):
        # reference time: 827 +- 4 ms
        std_spatial = 5
        std_color = 0.125
        image_in = skimage.io.imread('../ref/in.jpg').astype('float32') / 255.
        y, x = np.meshgrid(np.arange(image_in.shape[0]), np.arange(image_in.shape[1]), indexing='ij')
        points = np.concatenate((x[:, :, None], y[:, :, None], image_in), axis=-1)
        points[:, :, :2] /= std_spatial
        points[:, :, 2:] /= std_color

        points = points.reshape((1, -1, 5)).astype(np.float32)
        features = image_in.reshape((1, -1, 3)).astype(np.float32)


        gy = np.random.normal(size=features.shape).astype('float32')
        lattice = Lattice(points)
        function = FastGaussianFilter(lattice)
        # chainer.gradient_check.check_backward(function, features, gy, eps=1e-1, atol=1e-3, rtol=1e-3)

    def test_forward2(self):
        # reference time: 827 +- 4 ms
        # this gpu computation: 104 +- 1 ms
        std_spatial = 5
        std_color = 0.125
        image_in = skimage.io.imread('../ref/in.jpg').astype('float32') / 255.
        image_ref = np.array([[float(l2) for l2 in l.strip().split(',')] for l in open('./ref/ref.log').readlines()])
        image_ref = image_ref.reshape(image_in.shape)
        y, x = np.meshgrid(np.arange(image_in.shape[0]), np.arange(image_in.shape[1]), indexing='ij')
        points = np.concatenate((x[:, :, None], y[:, :, None], image_in), axis=-1)
        points[:, :, :2] /= std_spatial
        points[:, :, 2:] /= std_color

        points = points.reshape((1, -1, 5))
        features = image_in.reshape((1, -1, 3))
        points = np.tile(points, (2, 1, 1))
        features = np.tile(features, (2, 1, 1))
        points[0] = 0
        features[0] = 0
        points = points.astype(np.float32)
        features = features.astype(np.float32)

        image_out = fast_gaussian_filter(features, points=points)
        image_out = image_out.reshape((2, image_in.shape[0], image_in.shape[1], image_in.shape[2]))[1]

        diff = image_in - image_out.data.get()
        print ('diff_in', np.square(diff).mean())
        diff = image_ref - image_out.data.get()
        print ('diff_out', np.square(diff).mean())

    def test_forward3(self):
        # high-dimension
        dim = 64
        std_spatial = 5
        std_color = 0.125
        image_in = skimage.io.imread('../ref/in.jpg').astype('float32') / 255.
        image_in = image_in[:256, :256]
        y, x = np.meshgrid(np.arange(image_in.shape[0]), np.arange(image_in.shape[1]), indexing='ij')
        points = np.concatenate((x[:, :, None], y[:, :, None], image_in), axis=-1)
        points[:, :, :2] /= std_spatial
        points[:, :, 2:] /= std_color

        points = points.reshape((1, -1, 5))
        features = np.random.random(size=(image_in.shape[0], image_in.shape[1], dim))
        features = features.reshape((1, -1, features.shape[-1]))
        points = np.tile(points, (2, 1, 1))
        features = np.tile(features, (2, 1, 1))
        points[0] = 0
        features[0] = 0
        points = points.astype(np.float32)
        features = features.astype(np.float32)

        lattice = Lattice(points, hash_size=2 ** 20)
        features_out = fast_gaussian_filter(features, lattice=lattice)
        print (features_out[0, 0, 0])

        ts = time.time()
        features_out = fast_gaussian_filter(features, lattice=lattice)
        print (features_out[0, 0, 0])
        te = time.time()
        print (dim, te - ts)

# TestFastGaussianFilter().test_forward()

# class Lattice0:
#     def __init__(self, points, hash_size=2**24):
#         """
#         Initialize the lattice structure.
#         """
#         self.points = points
#         self.hash_size = hash_size
#         self.dim = points.shape[-1]
#         self.initialize_lattice()

#     def initialize_lattice(self):
#         """
#         Prepare lattice indices, weights, and neighbors.
#         """
#         projection_matrix = get_projection_matrix(self.dim)
#         points_proj = np.dot(self.points, projection_matrix.T)
#         reminder_zero_points, rank = compute_rzp_rank(points_proj, self.dim)
#         self.barycentric_weights = compute_weights(points_proj, reminder_zero_points, self.dim)
#         canonical = get_canonical_simplex(self.dim)
#         lattice_points = canonical[:, rank].transpose((1, 2, 0, 3)) + reminder_zero_points[:, :, None, :]
#         lattice_points = lattice_points.reshape((-1, self.dim + 1))

#         self.hash_map = HashMap(lattice_points, self.hash_size)
#         self.lattice_indices = np.array([self.hash_map.find(p) for p in lattice_points]).reshape(self.points.shape[0], -1, self.dim + 1)

#     def compute(self, features, forward=True):
#         """
#         Perform lattice computations, including splatting, blurring, and slicing.
#         """
#         bs, num_points, dim_features = features.shape
#         num_lattice_points = self.hash_map.size

#         # Splatting: Distribute features into lattice
#         lattice_features = np.zeros((num_lattice_points, dim_features), dtype=np.float32)
#         for b in range(bs):
#             for p in range(num_points):
#                 indices = self.lattice_indices[b, p]
#                 weights = self.barycentric_weights[b, p]
#                 for idx, weight in zip(indices, weights):
#                     if idx >= 0:
#                         lattice_features[idx] += weight * features[b, p]

#         # Blurring: Simple neighbor smoothing
#         lattice_features_blur = lattice_features * 0.5

#         # Slicing: Interpolate features from lattice
#         features_out = np.zeros_like(features)
#         for b in range(bs):
#             for p in range(num_points):
#                 indices = self.lattice_indices[b, p]
#                 weights = self.barycentric_weights[b, p]
#                 features_out[b, p] = sum(lattice_features_blur[idx] * weight for idx, weight in zip(indices, weights) if idx >= 0)

#         return features_out

# class FastGaussianFilter:
#     def __init__(self, lattice, normalize=True):
#         self.lattice = lattice
#         self.normalize = normalize

#     def forward(self, features):
#         """
#         Apply the forward Gaussian filter.
#         """
#         if self.normalize:
#             features = np.concatenate((features, np.ones((features.shape[0], features.shape[1], 1), dtype=np.float32)), axis=2)
#             features = self.lattice.compute(features)
#             weights = features[:, :, -1]
#             features = features[:, :, :-1] / (weights[:, :, None] + 1e-6)
#         else:
#             features = self.lattice.compute(features)
#         return features

# def fast_gaussian_filter(features, points, normalize=True):
#     """
#     High-level interface for applying the fast Gaussian filter.
#     """
#     lattice = Lattice(points)
#     filter_func = FastGaussianFilter(lattice, normalize)
#     return filter_func.forward(features)


# class HashMap0:
#     def __init__(self, data, hash_size=2**24):
#         """
#         Simple hash map to manage lattice points.
#         """
#         self.hash_size = hash_size
#         self.hash_table = {}
#         self.insert_data(data)

#     def compute_hash(self, key):
#         """
#         Compute a hash for the given key.
#         """
#         return hash(tuple(key)) % self.hash_size

#     def insert_data(self, data):
#         """
#         Insert data into the hash map.
#         """
#         for idx, key in enumerate(data):
#             h = self.compute_hash(key)
#             self.hash_table[h] = idx

#     def find(self, key):
#         """
#         Find an entry in the hash map.
#         """
#         h = self.compute_hash(key)
#         return self.hash_table.get(h, -1)

# class HashMap0:
#     def __init__(self, data, table_size=2**24):
#         self.hash_factor = 2531011
#         self.batch_size, self.num_points, self.dim = data.shape
#         self.table_size = table_size

#         # Initialize hash table
#         self.indices = -np.ones((self.batch_size, self.table_size), dtype=np.int32)
#         self.values = np.zeros((self.batch_size, self.table_size, self.dim), dtype=np.int32)
#         self.value_list = np.zeros((self.batch_size, self.table_size, self.dim), dtype=np.int32)
#         self.size = None

#         # Initialize keys in the hash table
#         self.init_keys(data)

#     def _compute_key(self, value):
#         """Compute the hash key for a given value."""
#         key = 0
#         for v in value:
#             key = (key + v) * self.hash_factor
#         return key % self.table_size

#     def init_keys(self, data):
#         data = np.ascontiguousarray(data)
#         used = np.zeros((self.batch_size, self.table_size), dtype=np.int32)
#         written = np.zeros((self.batch_size, self.table_size), dtype=np.int32)
#         count = np.zeros((self.batch_size,), dtype=np.int32)

#         for i in range(data.shape[0] * data.shape[1]):
#             bn = i // self.num_points  # Batch index
#             value_init = data[bn, i % self.num_points]

#             # Compute initial key
#             key = self._compute_key(value_init)

#             for _ in range(100):
#                 if used[bn, key] == 0:
#                     # Mark as used and insert the value
#                     used[bn, key] = 1
#                     self.values[bn, key] = value_init
#                     written[bn, key] = 1

#                     # Update indices and count
#                     index = count[bn]
#                     count[bn] += 1
#                     self.indices[bn, key] = index
#                     self.value_list[bn, index] = value_init
#                     break
#                 else:
#                     # Handle conflicts with linear probing
#                     if np.array_equal(self.values[bn, key], value_init):
#                         break
#                     key = (key + 1) % self.table_size
#             else:
#                 raise Exception("Hash table insertion failed due to excessive collisions")

#         self.size = int(count.max())

#     def find(self, data):
#         """Find the indices of the values in the hash table."""
#         ret = -np.ones(data.shape[:-1], dtype=np.int32)
#         data = np.ascontiguousarray(data)

#         for i in range(data.shape[0] * data.shape[1]):
#             bn = i // self.num_points  # Batch index
#             value = data[bn, i % self.num_points]

#             # Compute initial key
#             key = self._compute_key(value)

#             for _ in range(100):
#                 if self.indices[bn, key] < 0:
#                     ret[bn, i % self.num_points] = -1
#                     break
#                 if np.array_equal(self.values[bn, key], value):
#                     ret[bn, i % self.num_points] = self.indices[bn, key]
#                     break
#                 key = (key + 1) % self.table_size
#             else:
#                 raise Exception("Value not found after exhaustive search")

#         return ret
