import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False


def _is_torch(data):
    return _HAS_TORCH and isinstance(data, torch.Tensor)


def _to_numpy(data):
    if _is_torch(data):
        return data.detach().cpu().numpy(), True, data.device, data.dtype
    return np.asarray(data, dtype=np.float32), False, None, None


def _from_numpy(arr, is_torch_input, device, dtype, original):
    if is_torch_input:
        return torch.from_numpy(arr).to(device=device, dtype=dtype)
    return arr


def _get_shape_info(data_np):
    shape = data_np.shape
    n_time = shape[-2]
    n_trace = shape[-1]
    n_batch = int(np.prod(shape[:-2]))
    return n_batch, n_time, n_trace


# ==================== 1. 高斯白噪声 ====================
def add_gaussian_noise(data, snr_db=20):
    """
    高斯白噪声，支持任意形状 (..., time, trace) 的 numpy 或 torch 数据
    snr_db: 信噪比(dB)，常用10-40，值越小噪声越强
    """
    data_np, is_torch_input, device, dtype = _to_numpy(data)
    signal_power = np.mean(data_np ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(*data_np.shape).astype(data_np.dtype) * np.sqrt(noise_power)
    result = data_np + noise
    return _from_numpy(result, is_torch_input, device, dtype, data)


# ==================== 2. 均匀分布噪声 ====================
def add_uniform_noise(data, snr_db=20):
    """
    均匀分布噪声（平顶噪声），支持任意形状的 numpy 或 torch 数据
    snr_db: 信噪比(dB)
    """
    data_np, is_torch_input, device, dtype = _to_numpy(data)
    signal_power = np.mean(data_np ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    a = np.sqrt(3 * noise_power)
    noise = np.random.uniform(-a, a, data_np.shape).astype(data_np.dtype)
    result = data_np + noise
    return _from_numpy(result, is_torch_input, device, dtype, data)


# ==================== 3. 面波干扰（线性相干） ====================
def add_surface_wave(data, snr_db=20, dominant_freq=15, apparent_velocity=8):
    """
    面波干扰 - 低频、低速、线性同相轴，支持任意形状
    dominant_freq: 主频(Hz)，面波通常10-20Hz
    apparent_velocity: 视速度（道/采样点），越小越陡
    """
    data_np, is_torch_input, device, dtype = _to_numpy(data)
    orig_shape = data_np.shape
    n_batch, n_time, n_trace = _get_shape_info(data_np)
    data_3d = data_np.reshape(n_batch, n_time, n_trace)

    signal_power = np.mean(data_3d ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = np.zeros_like(data_3d)
    t = np.arange(n_time)

    for i in range(n_batch):
        for j in range(n_trace):
            shift = int(j * apparent_velocity)
            wave = np.sin(2 * np.pi * dominant_freq * t / 1000) * np.exp(-t / 800)
            if shift < n_time:
                wave = np.roll(wave, shift)
            noise[i, :, j] = wave

    actual_power = np.mean(noise ** 2)
    if actual_power > 0:
        noise *= np.sqrt(noise_power / actual_power)
    result = (data_3d + noise).reshape(orig_shape)
    return _from_numpy(result, is_torch_input, device, dtype, data)


# ==================== 4. 地滚波（频散面波） ====================
def add_ground_roll(data, snr_db=15):
    """地滚波 - 频散特性（频率越高速度越快），支持任意形状"""
    data_np, is_torch_input, device, dtype = _to_numpy(data)
    orig_shape = data_np.shape
    n_batch, n_time, n_trace = _get_shape_info(data_np)
    data_3d = data_np.reshape(n_batch, n_time, n_trace)

    signal_power = np.mean(data_3d ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.zeros_like(data_3d)
    t = np.arange(n_time)

    for i in range(n_batch):
        for j in range(n_trace):
            wave = np.zeros(n_time)
            for freq in [8, 12, 16]:
                velocity = 200 + freq * 5
                shift = int(j * 1000 / (velocity * 2))
                comp = np.sin(2 * np.pi * freq * t / 1000) * np.exp(-t / 600)
                if shift < n_time:
                    comp = np.roll(comp, shift)
                wave += comp
            noise[i, :, j] = wave

    actual_power = np.mean(noise ** 2)
    if actual_power > 0:
        noise *= np.sqrt(noise_power / actual_power)
    result = (data_3d + noise).reshape(orig_shape)
    return _from_numpy(result, is_torch_input, device, dtype, data)


# ==================== 5. 多次波（周期性） ====================
def add_multiples(data, snr_db=20, period=120, attenuation=0.8):
    """多次波 - 周期性重复信号，随时间衰减，支持任意形状"""
    data_np, is_torch_input, device, dtype = _to_numpy(data)
    orig_shape = data_np.shape
    n_batch, n_time, n_trace = _get_shape_info(data_np)
    data_3d = data_np.reshape(n_batch, n_time, n_trace)

    signal_power = np.mean(data_3d ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.zeros_like(data_3d)

    for i in range(n_batch):
        for j in range(n_trace):
            pulse = np.zeros(n_time)
            for k in range(0, n_time, period):
                if k < n_time:
                    pulse[k] = attenuation ** (k // period)

            b, a = signal.butter(4, [0.05, 0.4], 'band')
            wave = signal.filtfilt(b, a, pulse)
            noise[i, :, j] = wave * np.random.uniform(0.5, 1.5)

    actual_power = np.mean(noise ** 2)
    if actual_power > 0:
        noise *= np.sqrt(noise_power / actual_power)
    result = (data_3d + noise).reshape(orig_shape)
    return _from_numpy(result, is_torch_input, device, dtype, data)


# ==================== 6. 50Hz工业电干扰 ====================
def add_power_line_noise(data, snr_db=25, freq=50):
    """
    工业电干扰 - 单频正弦波（中国50Hz，欧美60Hz），支持任意形状
    """
    data_np, is_torch_input, device, dtype = _to_numpy(data)
    orig_shape = data_np.shape
    n_batch, n_time, n_trace = _get_shape_info(data_np)
    data_3d = data_np.reshape(n_batch, n_time, n_trace)

    signal_power = np.mean(data_3d ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.zeros_like(data_3d)
    t = np.arange(n_time)

    for i in range(n_batch):
        for j in range(n_trace):
            wave = np.sin(2 * np.pi * freq * t / 1000)
            spatial_mod = 1 + 0.5 * np.sin(j / 5)
            noise[i, :, j] = wave * spatial_mod

    actual_power = np.mean(noise ** 2)
    if actual_power > 0:
        noise *= np.sqrt(noise_power / actual_power)
    result = (data_3d + noise).reshape(orig_shape)
    return _from_numpy(result, is_torch_input, device, dtype, data)


# ==================== 7. 谐波干扰 ====================
def add_harmonic_noise(data, snr_db=25, base_freq=50, harmonics=[2, 3]):
    """
    谐波干扰 - 基频+多次谐波（如100Hz, 150Hz等），支持任意形状
    """
    data_np, is_torch_input, device, dtype = _to_numpy(data)
    orig_shape = data_np.shape
    n_batch, n_time, n_trace = _get_shape_info(data_np)
    data_3d = data_np.reshape(n_batch, n_time, n_trace)

    signal_power = np.mean(data_3d ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.zeros_like(data_3d)
    t = np.arange(n_time)

    for i in range(n_batch):
        for j in range(n_trace):
            wave = np.sin(2 * np.pi * base_freq * t / 1000)
            for h in harmonics:
                wave += (1 / h) * np.sin(2 * np.pi * base_freq * h * t / 1000)
            noise[i, :, j] = wave

    actual_power = np.mean(noise ** 2)
    if actual_power > 0:
        noise *= np.sqrt(noise_power / actual_power)
    result = (data_3d + noise).reshape(orig_shape)
    return _from_numpy(result, is_torch_input, device, dtype, data)


# ==================== 8. 死道（零值道） ====================
def add_dead_traces(data, num_bad=3):
    """
    死道 - 整道为零值（采集设备故障），支持任意形状
    num_bad: 坏道数量
    """
    data_np, is_torch_input, device, dtype = _to_numpy(data)
    orig_shape = data_np.shape
    n_batch, n_time, n_trace = _get_shape_info(data_np)
    result = data_np.copy().reshape(n_batch, n_time, n_trace)

    for _ in range(num_bad):
        b = np.random.randint(0, n_batch)
        tr = np.random.randint(0, n_trace)
        result[b, :, tr] = 0

    result = result.reshape(orig_shape)
    return _from_numpy(result, is_torch_input, device, dtype, data)


# ==================== 9. 坏道（强噪声道） ====================
def add_noisy_traces(data, num_bad=3, noise_factor=5):
    """
    坏道 - 整道被强随机噪声占据，支持任意形状
    num_bad: 坏道数量
    noise_factor: 噪声强度（相对于数据标准差的倍数）
    """
    data_np, is_torch_input, device, dtype = _to_numpy(data)
    orig_shape = data_np.shape
    n_batch, n_time, n_trace = _get_shape_info(data_np)
    result = data_np.copy().reshape(n_batch, n_time, n_trace)
    std = result.std()

    for _ in range(num_bad):
        b = np.random.randint(0, n_batch)
        tr = np.random.randint(0, n_trace)
        result[b, :, tr] = np.random.randn(n_time) * std * noise_factor

    result = result.reshape(orig_shape)
    return _from_numpy(result, is_torch_input, device, dtype, data)


# ==================== 10. 脉冲噪声（椒盐噪声） ====================
def add_spike_noise(data, ratio=0.01, amplitude=3):
    """
    脉冲噪声（野值）- 随机位置的强振幅点，支持任意形状
    ratio: 污染点比例（如0.01=1%）
    amplitude: 振幅倍数（相对于标准差）
    """
    data_np, is_torch_input, device, dtype = _to_numpy(data)
    orig_shape = data_np.shape
    n_batch, n_time, n_trace = _get_shape_info(data_np)
    result = data_np.copy().reshape(n_batch, n_time, n_trace)
    n_samples = int(result.size * ratio)
    std = result.std()

    for _ in range(n_samples):
        b = np.random.randint(0, n_batch)
        t = np.random.randint(0, n_time)
        tr = np.random.randint(0, n_trace)
        if np.random.rand() > 0.5:
            result[b, t, tr] = amplitude * std
        else:
            result[b, t, tr] = -amplitude * std

    result = result.reshape(orig_shape)
    return _from_numpy(result, is_torch_input, device, dtype, data)


# ==================== 11. 带限噪声 ====================
def add_band_limited_noise(data, snr_db=20, fmin=5, fmax=60):
    """
    带限随机噪声 - 只在特定频带内（模拟系统带内噪声），支持任意形状
    fmin, fmax: 截止频率(Hz)，假设采样率1000Hz
    """
    data_np, is_torch_input, device, dtype = _to_numpy(data)
    orig_shape = data_np.shape
    n_batch, n_time, n_trace = _get_shape_info(data_np)
    data_3d = data_np.reshape(n_batch, n_time, n_trace)

    signal_power = np.mean(data_3d ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = np.random.randn(*data_3d.shape).astype(data_3d.dtype)
    nyquist = 500

    low = fmin / nyquist
    high = fmax / nyquist
    if low < high < 1:
        b, a = signal.butter(4, [low, high], btype='band')
        for i in range(n_batch):
            for j in range(n_trace):
                noise[i, :, j] = signal.filtfilt(b, a, noise[i, :, j])

    current_power = np.mean(noise ** 2)
    if current_power > 0:
        noise *= np.sqrt(noise_power / current_power)
    result = (data_3d + noise).reshape(orig_shape)
    return _from_numpy(result, is_torch_input, device, dtype, data)


# ==================== 12. 低频噪声 ====================
def add_low_freq_noise(data, snr_db=20, cutoff=10):
    """
    低频噪声 - 重力变化、风干扰等（<10Hz），支持任意形状
    cutoff: 截止频率(Hz)
    """
    data_np, is_torch_input, device, dtype = _to_numpy(data)
    orig_shape = data_np.shape
    n_batch, n_time, n_trace = _get_shape_info(data_np)
    data_3d = data_np.reshape(n_batch, n_time, n_trace)

    signal_power = np.mean(data_3d ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = np.random.randn(*data_3d.shape).astype(data_3d.dtype)
    b, a = signal.butter(4, cutoff / 500, btype='low')

    for i in range(n_batch):
        for j in range(n_trace):
            noise[i, :, j] = signal.filtfilt(b, a, noise[i, :, j])

    current_power = np.mean(noise ** 2)
    if current_power > 0:
        noise *= np.sqrt(noise_power / current_power)
    result = (data_3d + noise).reshape(orig_shape)
    return _from_numpy(result, is_torch_input, device, dtype, data)


# ==================== 13. 空间相关噪声 ====================
def add_correlated_noise(data, snr_db=20, correlation_length=5):
    """
    空间相关噪声 - 相邻道噪声相关（环境噪声的空间连续性），支持任意形状
    correlation_length: 空间相关长度（道数）
    """
    data_np, is_torch_input, device, dtype = _to_numpy(data)
    orig_shape = data_np.shape
    n_batch, n_time, n_trace = _get_shape_info(data_np)
    data_3d = data_np.reshape(n_batch, n_time, n_trace)

    signal_power = np.mean(data_3d ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))

    noise = np.random.randn(*data_3d.shape).astype(data_3d.dtype)

    for i in range(n_batch):
        for t in range(n_time):
            noise[i, t, :] = gaussian_filter(noise[i, t, :], sigma=correlation_length)

    current_power = np.mean(noise ** 2)
    if current_power > 0:
        noise *= np.sqrt(noise_power / current_power)
    result = (data_3d + noise).reshape(orig_shape)
    return _from_numpy(result, is_torch_input, device, dtype, data)


# ==================== 14. 混合真实环境噪声 ====================
def add_realistic_field_noise(data, snr_db=20):
    """
    真实野外采集噪声组合（推荐用于最终测试），支持任意形状
    包含：高斯噪声 + 50Hz干扰 + 坏道 + 少量脉冲野值
    """
    noisy = add_gaussian_noise(data, snr_db=snr_db + 2)
    noisy = add_power_line_noise(noisy, snr_db=snr_db + 10, freq=50)
    noisy = add_noisy_traces(noisy, num_bad=1, noise_factor=4)
    noisy = add_spike_noise(noisy, ratio=0.001, amplitude=4)
    return noisy
