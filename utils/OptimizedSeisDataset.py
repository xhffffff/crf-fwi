import torch
import os
import glob
from torch.utils.data import DataLoader, Dataset
import numpy as np



def find_npy_files(folder_path):
    file_pattern = os.path.join(folder_path, '*.npy')
    npy_files = glob.glob(file_pattern)
    npy_files.sort()

    return npy_files


class OptimizedSeisDataset(Dataset):
    def __init__(self, folder, class_dict=None, seis_dict=None, h_o=70, w_o=70,
                 cache_strategy='memory_map', cache_size_mb=500):
        self.class_dict = class_dict if class_dict is not None else ['vmodel']
        self.seis_dict = seis_dict if seis_dict is not None else ['seis']
        self.folder = folder
        self.h_o = h_o
        self.w_o = w_o
        self.cache_strategy = cache_strategy
        self.cache_size_mb = cache_size_mb

        self.x_info = []
        self.seis_info = []
        self.file_cache = {}
        self.cache_order = []
        self.current_cache_size = 0

        self._collect_file_info()
        self._build_cumulative_counts()
        self._prepare_normalization()

    def _collect_file_info(self):
        for ci, c_v in enumerate(self.class_dict):
            files_path = find_npy_files(os.path.join(self.folder, c_v))
            for f_v in files_path:
                data = np.load(f_v, mmap_mode='r' if self.cache_strategy == 'memory_map' else None)
                tmp = data.reshape(-1, self.h_o, self.w_o)
                sample_count = tmp.shape[0]
                file_size_mb = os.path.getsize(f_v) / (1024 * 1024)
                self.x_info.append((f_v, ci, sample_count, file_size_mb))
                del data, tmp

        for c_s in self.seis_dict:
            seis_files_path = find_npy_files(os.path.join(self.folder, c_s))
            for f_s in seis_files_path:
                data = np.load(f_s, mmap_mode='r' if self.cache_strategy == 'memory_map' else None)
                tmp = data.reshape(-1, 5, 1000, 70)
                sample_count = tmp.shape[0]
                file_size_mb = os.path.getsize(f_s) / (1024 * 1024)
                self.seis_info.append((f_s, sample_count, file_size_mb))
                del data, tmp

    def _build_cumulative_counts(self):
        self.x_cum_counts = self._get_cumulative_counts(self.x_info, idx=2)
        self.seis_cum_counts = self._get_cumulative_counts(self.seis_info, idx=1)

    def _prepare_normalization(self):
        if not self.x_info:
            raise ValueError("vmodel文件夹下未找到任何文件")

        self.xmin = 1500.0
        self.xmax = 5500.0
        print(f"Global Normalization: [{self.xmin}, {self.xmax}] m/s")

        if self.seis_info:
            max_vals = []
            for i in range(min(10, len(self.seis_info))):
                seis_path = self.seis_info[i][0]
                data = np.load(seis_path, mmap_mode='r')
                max_vals.append(np.abs(data).max())

            self.seis_global_max = np.percentile(max_vals, 99)
            print(f"地震数据全局缩放因子（99分位数）: {self.seis_global_max:.4f}")
        else:
            self.seis_global_max = 1.0

    def _get_cumulative_counts(self, info_list, idx):
        cum_counts = [0]
        total = 0
        for info in info_list:
            total += info[idx]
            cum_counts.append(total)
        return cum_counts

    def _find_file_idx(self, idx, cum_counts):
        for i in range(len(cum_counts) - 1):
            if cum_counts[i] <= idx < cum_counts[i + 1]:
                return i
        raise IndexError(f"样本索引{idx}超出范围（总样本数{cum_counts[-1]}）")

    def _get_file_data(self, file_path, reshape_params):
        if self.cache_strategy == 'memory_map':
            return np.load(file_path, mmap_mode='r')

        elif self.cache_strategy == 'lru_cache':
            if file_path in self.file_cache:
                self.cache_order.remove(file_path)
                self.cache_order.append(file_path)
                return self.file_cache[file_path]

            data = np.load(file_path)
            data_size_mb = data.nbytes / (1024 * 1024)
            if self.current_cache_size + data_size_mb > self.cache_size_mb:
                oldest_file = self.cache_order.pop(0)
                oldest_size = self.file_cache[oldest_file].nbytes / (1024 * 1024)
                del self.file_cache[oldest_file]
                self.current_cache_size -= oldest_size

            # 缓存新文件
            self.file_cache[file_path] = data
            self.cache_order.append(file_path)
            self.current_cache_size += data_size_mb

            return data

        elif self.cache_strategy == 'preload':
            if file_path not in self.file_cache:
                self.file_cache[file_path] = np.load(file_path)
            return self.file_cache[file_path]

        else:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb < 50:
                if file_path not in self.file_cache:
                    self.file_cache[file_path] = np.load(file_path)
                return self.file_cache[file_path]
            else:
                return np.load(file_path, mmap_mode='r')

    def __len__(self):
        total_samples = sum([info[2] for info in self.x_info])
        return min(total_samples, 48000)

    def __getitem__(self, idx):
        x_file_idx = self._find_file_idx(idx, self.x_cum_counts)
        x_path, x_label, _, _ = self.x_info[x_file_idx]
        x_inner_idx = idx - self.x_cum_counts[x_file_idx]

        seis_file_idx = self._find_file_idx(idx, self.seis_cum_counts)
        seis_path, _, _ = self.seis_info[seis_file_idx]
        seis_inner_idx = idx - self.seis_cum_counts[seis_file_idx]

        x_data = self._get_file_data(x_path, (-1, self.h_o, self.w_o))
        x_sample = x_data.reshape(-1, self.h_o, self.w_o)[x_inner_idx]
        x = torch.Tensor(x_sample).unsqueeze(0).unsqueeze(0)
        x = x.squeeze(0)

        x = (x - self.xmin) / (self.xmax - self.xmin)
        x = x * 2.0 - 1.0

        seis_data = self._get_file_data(seis_path, (-1, 5, 1000, 70))
        seis_sample = seis_data.reshape(-1, 5, 1000, 70)[seis_inner_idx]
        seis = torch.Tensor(seis_sample)

        seis = seis / self.seis_global_max
        seis = torch.clamp(seis, -1, 1)

        if torch.rand(1).item() > 0.5:
            x = torch.flip(x, dims=[-1])
            seis = torch.flip(seis, dims=[-1])

        y = torch.tensor(x_label, dtype=torch.int32)

        return x, y, seis