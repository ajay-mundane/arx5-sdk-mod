import numpy as np
import pyrealsense2 as rs
from typing import Union, Dict, Optional
import os
import math
import numbers
import zarr
import numcodecs
import numpy as np
from functools import cached_property


class NumpyAccumulator:
    def __init__(self, initial_capacity=300, shape_suffix=(), dtype=np.float32):
        """
        initial_capacity: starting number of rows
        shape_suffix: trailing dimensions, e.g. (3,) for RGB, (4,4) for matrices
        dtype: dtype of the array
        """
        self.capacity = initial_capacity
        self._initial_capacity = initial_capacity  # Store for reset purposes
        self.size = 0   # number of actual elements stored
        
        self.shape_suffix = shape_suffix
        self.dtype = dtype

        self.buffer = np.zeros((self.capacity,) + shape_suffix, dtype=dtype)

    def __len__(self):
        return self.size

    def _ensure_capacity(self, needed_size):
        if needed_size <= self.capacity:
            return

        # Double until large enough
        new_capacity = self.capacity
        while new_capacity < needed_size:
            new_capacity *= 2

        new_buffer = np.zeros((new_capacity,) + self.shape_suffix, dtype=self.dtype)
        new_buffer[:self.size] = self.buffer[:self.size]
        
        self.buffer = new_buffer
        self.capacity = new_capacity

    def append(self, value):
        """Append a single item."""
        self._ensure_capacity(self.size + 1)
        self.buffer[self.size] = value
        self.size += 1

    def extend(self, values):
        """
        Append multiple values at once.
        values: np.ndarray with shape (N, *shape_suffix)
        """
        values = np.asarray(values)
        n = len(values)
        self._ensure_capacity(self.size + n)

        self.buffer[self.size:self.size+n] = values
        self.size += n

    @property
    def data(self):
        """Return a view of the valid portion."""
        return self.buffer[:self.size]
    
    def reset(self):
        self.size = 0
        # Reset capacity to initial size and create new buffer
        self.capacity = self._initial_capacity
        self.buffer = np.zeros((self.capacity,) + self.shape_suffix, dtype=self.dtype)



def check_chunks_compatible(chunks: tuple, shape: tuple):
    assert len(shape) == len(chunks)
    for c in chunks:
        assert isinstance(c, numbers.Integral)
        assert c > 0

def rechunk_recompress_array(group, name, 
        chunks=None, chunk_length=None,
        compressor=None, tmp_key='_temp'):
    old_arr = group[name]
    if chunks is None:
        if chunk_length is not None:
            chunks = (chunk_length,) + old_arr.chunks[1:]
        else:
            chunks = old_arr.chunks
    check_chunks_compatible(chunks, old_arr.shape)
    
    if compressor is None:
        compressor = old_arr.compressor
    
    if (chunks == old_arr.chunks) and (compressor == old_arr.compressor):
        # no change
        return old_arr

    # rechunk recompress
    group.move(name, tmp_key)
    old_arr = group[tmp_key]
    n_copied, n_skipped, n_bytes_copied = zarr.copy(
        source=old_arr,
        dest=group,
        name=name,
        chunks=chunks,
        compressor=compressor,
    )
    del group[tmp_key]
    arr = group[name]
    return arr

def get_optimal_chunks(shape, dtype, 
        target_chunk_bytes=2e6, 
        max_chunk_length=None):
    """
    Common shapes
    T,D
    T,N,D
    T,H,W,C
    T,N,H,W,C
    """
    itemsize = np.dtype(dtype).itemsize
    # reversed
    rshape = list(shape[::-1])
    if max_chunk_length is not None:
        rshape[-1] = int(max_chunk_length)
    split_idx = len(shape)-1
    for i in range(len(shape)-1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[:i+1])
        if this_chunk_bytes <= target_chunk_bytes \
            and next_chunk_bytes > target_chunk_bytes:
            split_idx = i

    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
    this_max_chunk_length = rshape[split_idx]
    next_chunk_length = min(this_max_chunk_length, math.ceil(
            target_chunk_bytes / item_chunk_bytes))
    rchunks.append(next_chunk_length)
    len_diff = len(shape) - len(rchunks)
    rchunks.extend([1] * len_diff)
    chunks = tuple(rchunks[::-1])
    # print(np.prod(chunks) * itemsize / target_chunk_bytes)
    return chunks


class ReplayBuffer:
    """
    Zarr-based temporal datastructure.
    Assumes first dimension to be time. Only chunk in time dimension.
    """
    def __init__(self, 
            root: Union[zarr.Group, 
            Dict[str,dict]]):
        """
        Dummy constructor. Use copy_from* and create_from* class methods instead.
        """
        assert('data' in root)
        assert('meta' in root)
        assert('episode_ends' in root['meta'])
        for key, value in root['data'].items():
            assert(value.shape[0] == root['meta']['episode_ends'][-1])
        self.root = root
    
    # ============= create constructors ===============
    @classmethod
    def create_empty_zarr(cls, storage=None, root=None):
        if root is None:
            if storage is None:
                storage = zarr.MemoryStore()
            root = zarr.group(store=storage)
        data = root.require_group('data', overwrite=False)
        meta = root.require_group('meta', overwrite=False)
        if 'episode_ends' not in meta:
            episode_ends = meta.zeros('episode_ends', shape=(0,), dtype=np.int64,
                compressor=None, overwrite=False)
        return cls(root=root)
    
    @classmethod
    def create_empty_numpy(cls):
        root = {
            'data': dict(),
            'meta': {
                'episode_ends': np.zeros((0,), dtype=np.int64)
            }
        }
        return cls(root=root)
    
    @classmethod
    def create_from_group(cls, group, **kwargs):
        if 'data' not in group:
            # create from stratch
            buffer = cls.create_empty_zarr(root=group, **kwargs)
        else:
            # already exist
            buffer = cls(root=group, **kwargs)
        return buffer

    @classmethod
    def create_from_path(cls, zarr_path, mode='r', **kwargs):
        """
        Open a on-disk zarr directly (for dataset larger than memory).
        Slower.
        """
        group = zarr.open(os.path.expanduser(zarr_path), mode)
        return cls.create_from_group(group, **kwargs)
    
    # ============= copy constructors ===============
    @classmethod
    def copy_from_store(cls, src_store, store=None, keys=None, 
            chunks: Dict[str,tuple]=dict(), 
            compressors: Union[dict, str, numcodecs.abc.Codec]=dict(), 
            if_exists='replace',
            **kwargs):
        """
        Load to memory.
        """
        src_root = zarr.group(src_store)
        root = None
        if store is None:
            # numpy backend
            meta = dict()
            for key, value in src_root['meta'].items():
                if len(value.shape) == 0:
                    meta[key] = np.array(value)
                else:
                    meta[key] = value[:]

            if keys is None:
                keys = src_root['data'].keys()
            data = dict()
            for key in keys:
                arr = src_root['data'][key]
                data[key] = arr[:]

            root = {
                'meta': meta,
                'data': data
            }
        else:
            root = zarr.group(store=store)
            # copy without recompression
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(source=src_store, dest=store,
                source_path='/meta', dest_path='/meta', if_exists=if_exists)
            data_group = root.create_group('data', overwrite=True)
            if keys is None:
                keys = src_root['data'].keys()
            for key in keys:
                value = src_root['data'][key]
                cks = cls._resolve_array_chunks(
                    chunks=chunks, key=key, array=value)
                cpr = cls._resolve_array_compressor(
                    compressors=compressors, key=key, array=value)
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = '/data/' + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=src_store, dest=store,
                        source_path=this_path, dest_path=this_path,
                        if_exists=if_exists
                    )
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value, dest=data_group, name=key,
                        chunks=cks, compressor=cpr, if_exists=if_exists
                    )
        buffer = cls(root=root)
        return buffer
    
    @classmethod
    def copy_from_path(cls, zarr_path, backend=None, store=None, keys=None, 
            chunks: Dict[str,tuple]=dict(), 
            compressors: Union[dict, str, numcodecs.abc.Codec]=dict(), 
            if_exists='replace',
            **kwargs):
        """
        Copy a on-disk zarr to in-memory compressed.
        Recommended
        """
        if backend == 'numpy':
            print('backend argument is deprecated!')
            store = None
        group = zarr.open(os.path.expanduser(zarr_path), 'r')
        return cls.copy_from_store(src_store=group.store, store=store, 
            keys=keys, chunks=chunks, compressors=compressors, 
            if_exists=if_exists, **kwargs)

    # ============= save methods ===============
    def save_to_store(self, store, 
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict(),
            if_exists='replace', 
            **kwargs):
        
        root = zarr.group(store)
        if self.backend == 'zarr':
            # recompression free copy
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                source=self.root.store, dest=store,
                source_path='/meta', dest_path='/meta', if_exists=if_exists)
        else:
            meta_group = root.create_group('meta', overwrite=True)
            # save meta, no chunking
            for key, value in self.root['meta'].items():
                _ = meta_group.array(
                    name=key,
                    data=value, 
                    shape=value.shape, 
                    chunks=value.shape)
        
        # save data, chunk
        data_group = root.create_group('data', overwrite=True)
        for key, value in self.root['data'].items():
            cks = self._resolve_array_chunks(
                chunks=chunks, key=key, array=value)
            cpr = self._resolve_array_compressor(
                compressors=compressors, key=key, array=value)
            if isinstance(value, zarr.Array):
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = '/data/' + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=self.root.store, dest=store,
                        source_path=this_path, dest_path=this_path, if_exists=if_exists)
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value, dest=data_group, name=key,
                        chunks=cks, compressor=cpr, if_exists=if_exists
                    )
            else:
                # numpy
                _ = data_group.array(
                    name=key,
                    data=value,
                    chunks=cks,
                    compressor=cpr
                )
        return store

    def save_to_path(self, zarr_path,             
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict(), 
            if_exists='replace', 
            **kwargs):
        store = zarr.DirectoryStore(os.path.expanduser(zarr_path))
        return self.save_to_store(store, chunks=chunks, 
            compressors=compressors, if_exists=if_exists, **kwargs)

    @staticmethod
    def resolve_compressor(compressor='default'):
        if compressor == 'default':
            compressor = numcodecs.Blosc(cname='lz4', clevel=5, 
                shuffle=numcodecs.Blosc.NOSHUFFLE)
        elif compressor == 'disk':
            compressor = numcodecs.Blosc('zstd', clevel=5, 
                shuffle=numcodecs.Blosc.BITSHUFFLE)
        return compressor

    @classmethod
    def _resolve_array_compressor(cls, 
            compressors: Union[dict, str, numcodecs.abc.Codec], key, array):
        # allows compressor to be explicitly set to None
        cpr = 'nil'
        if isinstance(compressors, dict):
            if key in compressors:
                cpr = cls.resolve_compressor(compressors[key])
            elif isinstance(array, zarr.Array):
                cpr = array.compressor
        else:
            cpr = cls.resolve_compressor(compressors)
        # backup default
        if cpr == 'nil':
            cpr = cls.resolve_compressor('default')
        return cpr
    
    @classmethod
    def _resolve_array_chunks(cls,
            chunks: Union[dict, tuple], key, array):
        cks = None
        if isinstance(chunks, dict):
            if key in chunks:
                cks = chunks[key]
            elif isinstance(array, zarr.Array):
                cks = array.chunks
        elif isinstance(chunks, tuple):
            cks = chunks
        else:
            raise TypeError(f"Unsupported chunks type {type(chunks)}")
        # backup default
        if cks is None:
            cks = get_optimal_chunks(shape=array.shape, dtype=array.dtype)
        # check
        check_chunks_compatible(chunks=cks, shape=array.shape)
        return cks
    
    # ============= properties =================
    @cached_property
    def data(self):
        return self.root['data']
    
    @cached_property
    def meta(self):
        return self.root['meta']

    def update_meta(self, data):
        # sanitize data
        np_data = dict()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np_data[key] = value
            else:
                arr = np.array(value)
                if arr.dtype == object:
                    raise TypeError(f"Invalid value type {type(value)}")
                np_data[key] = arr

        meta_group = self.meta
        if self.backend == 'zarr':
            for key, value in np_data.items():
                _ = meta_group.array(
                    name=key,
                    data=value, 
                    shape=value.shape, 
                    chunks=value.shape,
                    overwrite=True)
        else:
            meta_group.update(np_data)
        
        return meta_group
    
    @property
    def episode_ends(self):
        return self.meta['episode_ends']
    
    def get_episode_idxs(self):
        import numba
        numba.jit(nopython=True)
        def _get_episode_idxs(episode_ends):
            result = np.zeros((episode_ends[-1],), dtype=np.int64)
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                for idx in range(start, end):
                    result[idx] = i
            return result
        return _get_episode_idxs(self.episode_ends)
        
    
    @property
    def backend(self):
        backend = 'numpy'
        if isinstance(self.root, zarr.Group):
            backend = 'zarr'
        return backend
    
    # =========== dict-like API ==============
    def __repr__(self) -> str:
        if self.backend == 'zarr':
            return str(self.root.tree())
        else:
            return super().__repr__()

    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()
    
    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    # =========== our API ==============
    @property
    def n_steps(self):
        if len(self.episode_ends) == 0:
            return 0
        return self.episode_ends[-1]
    
    @property
    def n_episodes(self):
        return len(self.episode_ends)

    @property
    def chunk_size(self):
        if self.backend == 'zarr':
            return next(iter(self.data.arrays()))[-1].chunks[0]
        return None

    @property
    def episode_lengths(self):
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        lengths = np.diff(ends)
        return lengths
    
    def add_episode_separate(self, 
            data: Dict[str, np.ndarray], 
            chunks: Optional[Dict[str, tuple]] = dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict] = dict(),
            finalize_episode: bool = True):
            """
            Adds an episode to the buffer.
            
            Args:
                data: Dictionary of arrays to add.
                chunks: Chunking configuration for Zarr.
                compressors: Compressor configuration for Zarr.
                finalize_episode: If True, updates 'episode_ends' to mark the episode as complete. 
                                If False, writes data to arrays but keeps the episode 'open' 
                                (does not increment n_steps). Use False for the first N-1 processes 
                                and True for the final process.
            """
            assert(len(data) > 0)
            is_zarr = (self.backend == 'zarr')

            # Get current global length (based on the last finalized episode)
            # If finalize_episode was False in a previous call, n_steps has NOT moved yet,
            # so we correctly overwrite/append to the same index range.
            curr_len = self.n_steps
            
            episode_length = None
            for key, value in data.items():
                assert(len(value.shape) >= 1)
                if episode_length is None:
                    episode_length = len(value)
                else:
                    assert(episode_length == len(value))
                    
            # Calculate the target length for this specific write
            new_len = curr_len + episode_length

            for key, value in data.items():
                new_shape = (new_len,) + value.shape[1:]
                
                # 1. Create array if it doesn't exist
                if key not in self.data:
                    if is_zarr:
                        cks = self._resolve_array_chunks(
                            chunks=chunks, key=key, array=value)
                        cpr = self._resolve_array_compressor(
                            compressors=compressors, key=key, array=value)
                        arr = self.data.zeros(name=key, 
                            shape=new_shape, 
                            chunks=cks,
                            dtype=value.dtype,
                            compressor=cpr)
                    else:
                        # copy data to prevent modify
                        arr = np.zeros(shape=new_shape, dtype=value.dtype)
                        self.data[key] = arr
                else:
                    # 2. Resize existing array
                    arr = self.data[key]
                    assert(value.shape[1:] == arr.shape[1:])
                    
                    # Only resize if the array isn't already large enough.
                    # This handles cases where Process A (Action) resized it to new_len,
                    # and now Process B (Image) sees it's already big enough.
                    if arr.shape[0] < new_len:
                        if is_zarr:
                            arr.resize(new_shape)
                        else:
                            arr.resize(new_shape, refcheck=False)
                            
                # 3. Copy data to the specific slice
                # We write to the range [curr_len : new_len]
                arr[curr_len:new_len] = value
            
            # Only update the global episode tracker if this is the generic "finalizing" call
            if finalize_episode:
                # append to episode ends
                episode_ends = self.episode_ends
                if is_zarr:
                    episode_ends.resize(episode_ends.shape[0] + 1)
                else:
                    episode_ends.resize(episode_ends.shape[0] + 1, refcheck=False)
                episode_ends[-1] = new_len

                # rechunk
                if is_zarr:
                    if episode_ends.chunks[0] < episode_ends.shape[0]:
                        rechunk_recompress_array(self.meta, 'episode_ends', 
                            chunk_length=int(episode_ends.shape[0] * 1.5))

    def add_episode(self, 
            data: Dict[str, np.ndarray], 
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict()):
        assert(len(data) > 0)
        is_zarr = (self.backend == 'zarr')

        curr_len = self.n_steps
        episode_length = None
        for key, value in data.items():
            assert(len(value.shape) >= 1)
            if episode_length is None:
                episode_length = len(value)
            else:
                assert(episode_length == len(value))
        new_len = curr_len + episode_length

        for key, value in data.items():
            new_shape = (new_len,) + value.shape[1:]
            # create array
            if key not in self.data:
                if is_zarr:
                    cks = self._resolve_array_chunks(
                        chunks=chunks, key=key, array=value)
                    cpr = self._resolve_array_compressor(
                        compressors=compressors, key=key, array=value)
                    arr = self.data.zeros(name=key, 
                        shape=new_shape, 
                        chunks=cks,
                        dtype=value.dtype,
                        compressor=cpr)
                else:
                    # copy data to prevent modify
                    arr = np.zeros(shape=new_shape, dtype=value.dtype)
                    self.data[key] = arr
            else:
                arr = self.data[key]
                assert(value.shape[1:] == arr.shape[1:])
                # same method for both zarr and numpy
                if is_zarr:
                    arr.resize(new_shape)
                else:
                    arr.resize(new_shape, refcheck=False)
            # copy data
            arr[-value.shape[0]:] = value
        
        # append to episode ends
        episode_ends = self.episode_ends
        if is_zarr:
            episode_ends.resize(episode_ends.shape[0] + 1)
        else:
            episode_ends.resize(episode_ends.shape[0] + 1, refcheck=False)
        episode_ends[-1] = new_len

        # rechunk
        if is_zarr:
            if episode_ends.chunks[0] < episode_ends.shape[0]:
                rechunk_recompress_array(self.meta, 'episode_ends', 
                    chunk_length=int(episode_ends.shape[0] * 1.5))
    
    def drop_episode(self):
        is_zarr = (self.backend == 'zarr')
        episode_ends = self.episode_ends[:].copy()
        assert(len(episode_ends) > 0)
        start_idx = 0
        if len(episode_ends) > 1:
            start_idx = episode_ends[-2]
        for key, value in self.data.items():
            new_shape = (start_idx,) + value.shape[1:]
            if is_zarr:
                value.resize(new_shape)
            else:
                value.resize(new_shape, refcheck=False)
        if is_zarr:
            self.episode_ends.resize(len(episode_ends)-1)
        else:
            self.episode_ends.resize(len(episode_ends)-1, refcheck=False)
    
    def pop_episode(self):
        assert(self.n_episodes > 0)
        episode = self.get_episode(self.n_episodes-1, copy=True)
        self.drop_episode()
        return episode

    def extend(self, data):
        self.add_episode(data)

    def get_episode(self, idx, copy=False):
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        result = self.get_steps_slice(start_idx, end_idx, copy=copy)
        return result
    
    def get_episode_slice(self, idx):
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        return slice(start_idx, end_idx)

    def get_steps_slice(self, start, stop, step=None, copy=False):
        _slice = slice(start, stop, step)

        result = dict()
        for key, value in self.data.items():
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result
    
    # =========== chunking =============
    def get_chunks(self) -> dict:
        assert self.backend == 'zarr'
        chunks = dict()
        for key, value in self.data.items():
            chunks[key] = value.chunks
        return chunks
    
    def set_chunks(self, chunks: dict):
        assert self.backend == 'zarr'
        for key, value in chunks.items():
            if key in self.data:
                arr = self.data[key]
                if value != arr.chunks:
                    check_chunks_compatible(chunks=value, shape=arr.shape)
                    rechunk_recompress_array(self.data, key, chunks=value)

    def get_compressors(self) -> dict:
        assert self.backend == 'zarr'
        compressors = dict()
        for key, value in self.data.items():
            compressors[key] = value.compressor
        return compressors

    def set_compressors(self, compressors: dict):
        assert self.backend == 'zarr'
        for key, value in compressors.items():
            if key in self.data:
                arr = self.data[key]
                compressor = self.resolve_compressor(value)
                if compressor != arr.compressor:
                    rechunk_recompress_array(self.data, key, compressor=compressor)



def match_observations_to_actions(action_timestamps, cam_timestamps):
    """
    Returns indices of actions corresponding to the closest smaller 
    (or equal) timestamp for each camera observation.
    """
    # 1. Find insertion points
    # side='right' finds the first index where the value is strictly greater 
    # than the query.
    indices = np.searchsorted(action_timestamps, cam_timestamps, side='right')
    
    # 2. Shift left to get the "closest smaller/equal"
    # Since searchsorted returns the index *after* the match, we subtract 1
    matched_indices = indices - 1
    
    # 3. (Optional but recommended) Handle boundary cases
    # If a camera obs happened before the first action, index will be -1.
    # You might want to filter those out.
    valid_mask = matched_indices >= 0
    
    return matched_indices[valid_mask], valid_mask

def align_camera_timestamps(cameras_data):
    """
    Align camera timestamps using the camera with the least frames as baseline.
    Returns aligned camera data where all cameras have the same number of observations.
    """
    if not cameras_data:
        return {}
    
    # Find camera with minimum number of frames
    min_frames = float('inf')
    latest_start = -float('inf')
    baseline_serial = None
    for serial, data in cameras_data.items():
        n_frames = len(data['cam_timestamps'])
        if n_frames <= min_frames and data['cam_timestamps'][0] >= latest_start:
            min_frames = n_frames
            baseline_serial = serial
            latest_start = data['cam_timestamps'][0]
    
    print(f"Using camera {baseline_serial} as baseline with {min_frames} frames")
    baseline_timestamps = cameras_data[baseline_serial]['cam_timestamps']
    
    # Align all cameras to baseline timestamps
    aligned_cameras = {}
    for i, (serial, data) in enumerate(cameras_data.items()):
        # Create aligned data with camera index naming
        aligned_data = {}
        
        if serial == baseline_serial:
            # Baseline camera - just rename keys with index
            for key, value in data.items():
                if key == 'cam_timestamps':
                    aligned_data[key] = value
                elif key == 'color':
                    aligned_data[f'camera{i}_rgb'] = value
                elif key == 'depth':
                    aligned_data[f'camera{i}_depth'] = value
                else:
                    aligned_data[f'camera{i}_{key}'] = value
        else:
            # Other cameras - align to baseline
            action_idxs, mask = match_observations_to_actions(
                data['cam_timestamps'],
                baseline_timestamps
            )
            
            # Apply alignment
            for key, value in data.items():
                if key == 'cam_timestamps':
                    aligned_data[key] = baseline_timestamps  # Use baseline timestamps
                elif key == 'color':
                    aligned_data[f'camera{i}_rgb'] = value[action_idxs]
                elif key == 'depth':
                    aligned_data[f'camera{i}_depth'] = value[action_idxs]
                else:
                    aligned_data[f'camera{i}_{key}'] = value[action_idxs]
        
        aligned_cameras[serial] = aligned_data
    
    # Merge all camera data into single dictionary
    merged_data = {'cam_timestamps': baseline_timestamps}
    for serial, data in aligned_cameras.items():
        for key, value in data.items():
            if key != 'cam_timestamps':  # Don't duplicate timestamps
                merged_data[key] = value
    
    return merged_data



def preprocess_robot_actions(robot_data):
    """
    Combine robot action keys into a single 'action' array.
    Format: [joint_pos_L, gripper_pos_L, joint_pos_R, gripper_pos_R]
    """
    # Extract individual components
    joint_pos_L = robot_data['joint_pos_L']  # Shape: (N, dof)
    gripper_pos_L = robot_data['gripper_pos_L'].reshape(-1, 1)  # Shape: (N, 1)
    joint_pos_R = robot_data['joint_pos_R']  # Shape: (N, dof)
    gripper_pos_R = robot_data['gripper_pos_R'].reshape(-1, 1)  # Shape: (N, 1)
    
    # Concatenate all components
    action = np.concatenate([
        joint_pos_L,
        gripper_pos_L,
        joint_pos_R,
        gripper_pos_R
    ], axis=1)
    
    # Create new data dictionary with action and timestamps only
    processed_data = {
        'action': action,
        'action_timestamps': robot_data['action_timestamps']
    }
    
    return processed_data
