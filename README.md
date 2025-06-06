训练、辨别哭声制作手机APP项目
技 术 文 件
2025年6月

目录
简单背景	1
第一阶段工作	2
(一) 013版代码	2
第1单元	2
第2单元	6
第3单元	9
(二) 014版代码	20
第1单元	20
第2单元	24
第3单元	26
第二阶段工作	38
(一) export_pytorch_to_onnx代码	38
(二) convert_stage1代码	43
(三) convert_stage2代码	46
第三阶段工作	47



简单背景
我在中国大陆从事母婴护理行业十多年，我收集了一批婴儿在饥饿、生气、高兴、不舒服等各种生活场景下的哭声（22050Hz，每个音频大约4~7秒，统一截取6秒），我想训练这些哭声制作成模型，并最终生成手机APP，让爸爸妈妈通过手机上传自己宝宝的哭声，就可以知道宝宝哭声所表达的的状态和需求了。初期并不刻板追求辨别哭声的准确度（确保60%、争取70%），随着用户数据的越来越多，匹配度就会提高（要求80%）。
现在已经完成了两个阶段的工作：
第一阶段：在kaggle上借助其免费的gpu生成“best_cry_classifier_v13_fold_3.pth”文件，原始数据（婴儿哭声音频文件）路径：
'/kaggle/input/baby-kusen/Raw data':
belly_pain  discomfort	hungry	psychological_needs  tired。
第二阶段：在本地powershell中搭建虚拟环境，将上述“.pth”转换为“.tflite”文件。

项目的整体指导思想是——
	四个结合：智能预测哭声类型与人工辨听相结合、线上APP与线下培训相结合、APP软件与自媒体直播相结合、APP下载量与创收盈利相结合；
	三个平衡：把握好“工程量”与“准确率”的平衡、“质量”与“成本”的平衡、“效率”与“效果”的平衡；
	两个尽量：尽量利用网络免费资源(特别在初期)、尽量收集丰富原始数据(特别是高质量数据)；
	一个追求：追求稳健，由于编程、代码、AI、前后端等等距离自己的专业(母婴护理、育婴员讲师)相去甚远，所以追求稳扎稳打、一步一个脚印往前推。


第一阶段工作
(一) 013版代码
第1单元

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import warnings
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder # StandardScaler 不再使用
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR # CosineAnnealingLR 暂不使用
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
import joblib 
import time 

warnings.filterwarnings('ignore')

# --- 1. 配置参数段落 ---
class Config:
    # --- 路径配置 ---
    RAW_DATA_PATH = '/kaggle/input/baby-kusen/Raw data' 
    # 处理后的特征数据保存路径，方便 K-Fold 共享和快速加载
    PROCESSED_DATA_DIR = '/kaggle/working/processed_features' 
    PROCESSED_DATA_FILENAME = 'baby_cry_features_v13_ultimate.pkl' 
    MODEL_SAVE_DIR = '/kaggle/working/output' # 模型保存目录
    BEST_MODEL_FILENAME = 'best_cry_classifier_v13_ultimate_fold_{}.pth' # K-Fold 最佳模型文件名

    # --- 音频特征提取参数 ---
    SR = 22050           # 采样率 (Default: 22050)
    AUDIO_TARGET_LENGTH_SEC = 6 # 音频目标长度 (秒)
    N_MELS = 128         # 梅尔频谱图的梅尔带数量 (Default: 128)
    N_MFCC = 40          # 基础MFCC特征数量 (模型实际输入是 N_MFCC * 3, 因为包含 delta 和 delta-delta)
    N_FRAMES = 259       # 统一的特征时间帧数 (确保这个值能覆盖 6秒音频的特征帧数)
                         # 计算示例: 6秒 * 22050Hz / 512(hop_length) = 258.98，取 259 帧

    # --- 数据增强参数 ---
    # 频谱增强 (SpecAugment)
    SPEC_AUGMENT_TIME_MASK_PARAM = 20 # 默认参数，可调 (范围: 10-40)
    SPEC_AUGMENT_FREQ_MASK_PARAM = 10 # 默认参数，可调 (范围: 5-20)
    
    # 时域数据增强 (AdvancedAudioAugmenter)
    APPLY_AUDIO_AUGMENT = True       # 是否应用时域增强 (True/False)
    AUGMENT_PROB_TIME_DOMAIN = 0.5   # 单个样本应用时域数据增强的概率 (Default: 0.5)
    PITCH_SHIFT_STEPS_RANGE = (-1.5, 1.5) # 音高变换步长范围 (更温和)
    TIME_STRETCH_RATE_RANGE = (0.95, 1.05) # 时间拉伸比率范围 (更温和)
    NOISE_LEVEL = 0.0008 # 添加噪声的强度 (Default: 0.001, 稍低)

    # Mixup 数据增强 (在训练循环中应用)
    APPLY_MIXUP = True               # 是否应用 Mixup (True/False)
    MIXUP_ALPHA = 0.3                # Mixup 的 Alpha 参数 (Default: 0.3, 范围: 0.1-0.4)

    # --- 训练参数 ---
    BATCH_SIZE = 32      # 批处理大小 (Default: 32)
    NUM_EPOCHS = 100     # 总训练轮次 (Default: 100, 增加以应对更多增强和 K-Fold)
    INITIAL_LR = 0.0005  # AdamW 初始学习率 (OneCycleLR 会覆盖，但作为参考)
    MAX_LR = 0.004       # OneCycleLR 的最大学习率 (Default: 0.004, 范围: 0.003-0.01)
    WEIGHT_DECAY = 0.01  # 优化器权重衰减 (Default: 0.01)
    LABEL_SMOOTHING = 0.1 # 标签平滑强度 (Default: 0.1, 范围: 0.05-0.2)
    PATIENCE = 20        # 早停耐心值 (Default: 20)

    # --- 模型参数 ---
    DROPOUT_RATE_RESBLOCK = 0.15 # 残差块内的Dropout率 (Default: 0.15)
    DROPOUT_RATE_FUSION = 0.3    # 融合层内的Dropout率 (Default: 0.3)

    # --- K-Fold Cross-Validation 参数 ---
    N_SPLITS = 5         # K-Fold 的折数 (Default: 5)

    # --- 其他配置 ---
    RANDOM_SEED = 42     # 随机种子 (Default: 42)
    MIN_SAMPLES_PER_CLASS = 50 # 跳过类别所需的最小样本数 (Default: 50)


# --- 2. 通用模块与核心组件 ---

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用的计算设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")
else:
    print("未检测到GPU，将使用CPU进行训练。")

# Label Smoothing损失函数
class LabelSmoothingLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=Config.LABEL_SMOOTHING):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def reduce_loss(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def smooth_loss(self, log_prob, targets):
        with torch.no_grad():
            targets = torch.zeros_like(log_prob).scatter_(1, targets.unsqueeze(1), 1)
            targets = (1 - self.smoothing) * targets + self.smoothing / log_prob.size(-1)
        return torch.sum(-targets * log_prob, dim=-1)

    def forward(self, inputs, targets):
        log_prob = F.log_softmax(inputs, dim=-1)
        loss = self.smooth_loss(log_prob, targets)
        return self.reduce_loss(loss)

# 频谱增强
class SpecAugment:
    def __init__(self, time_mask_param=Config.SPEC_AUGMENT_TIME_MASK_PARAM, 
                 freq_mask_param=Config.SPEC_AUGMENT_FREQ_MASK_PARAM):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
    
    def time_mask(self, spec):
        # spec 预期维度: (..., freq_bins, time_frames)
        # 支持 (batch_size, channels, freq_bins, time_frames) 或 (channels, freq_bins, time_frames)
        if spec.dim() < 3: return spec # 确保至少有 freq 和 time 维度
        
        # 找出时间维度
        time_dim_idx = spec.dim() - 1 
        v = spec.size(time_dim_idx)
        if v == 0: return spec 

        t = np.random.randint(0, self.time_mask_param)
        t = min(t, v - 1) 
        t0 = np.random.randint(0, v - t) if v > t else 0
        
        # 构建切片元组以适应不同维度
        slices = [slice(None)] * spec.dim()
        slices[time_dim_idx] = slice(t0, t0 + t)
        spec[tuple(slices)] = 0
        return spec
    
    def freq_mask(self, spec):
        # spec 预期维度: (..., freq_bins, time_frames)
        # 支持 (batch_size, channels, freq_bins, time_frames) 或 (channels, freq_bins, time_frames)
        if spec.dim() < 2: return spec # 确保至少有 freq 维度

        # 找出频率维度
        freq_dim_idx = spec.dim() - 2 # 倒数第二个维度是频率维度
        v = spec.size(freq_dim_idx)
        if v == 0: return spec

        f = np.random.randint(0, self.freq_mask_param)
        f = min(f, v - 1)
        f0 = np.random.randint(0, v - f) if v > f else 0

        # 构建切片元组以适应不同维度
        slices = [slice(None)] * spec.dim()
        slices[freq_dim_idx] = slice(f0, f0 + f)
        spec[tuple(slices)] = 0
        return spec

    def augment(self, spec):
        augmented_spec = spec.clone() 
        augmented_spec = self.freq_mask(augmented_spec)
        augmented_spec = self.time_mask(augmented_spec)
        return augmented_spec

# 自适应特征融合
class AdaptiveFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 8, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        weights = self.attention(x)
        return x * weights 

# Mixup 数据增强函数
def mixup_data(x_mfcc, x_mel, x_chroma, y, alpha=Config.MIXUP_ALPHA, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x_mfcc.size()[0]
    index = torch.randperm(batch_size).to(device)

    # 对所有输入特征应用相同的 Mixup 混合
    mixed_mfcc = lam * x_mfcc + (1 - lam) * x_mfcc[index]
    mixed_mel = lam * x_mel + (1 - lam) * x_mel[index]
    mixed_chroma = lam * x_chroma + (1 - lam) * x_chroma[index]

    y_a, y_b = y, y[index]
    return mixed_mfcc, mixed_mel, mixed_chroma, y_a, y_b, lam

# Mixup 损失计算函数
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

第2单元

# --- 3. 模型定义 ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

# 改进的残差模块
class ImprovedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, 1),
            nn.Sigmoid()
        )
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.dropout = nn.Dropout2d(Config.DROPOUT_RATE_RESBLOCK) 
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) * out
        out = self.dropout(out)
        return F.relu(out + self.shortcut(x))

# 特征提取器基类
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.features = nn.Sequential(
            ImprovedResBlock(in_channels, 32), 
            ImprovedResBlock(32, 64, stride=2),
            ImprovedResBlock(64, 128, stride=2),
            ImprovedResBlock(128, 256, stride=2)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = self.features(x)
        return self.gap(x)

# 改进的CNN模型 (V13 Ultimate)
class CryCNN_V13_Ultimate(nn.Module): 
    def __init__(self, num_classes):
        super().__init__()
        
        # in_channels=1 是因为我们将每个特征图 (MFCC, Mel, Chroma) 视为一个单通道图像
        # 即使 MFCCs 是 120 维，它也是一个 120x259 的单通道特征图
        self.mfcc_branch = FeatureExtractor(1) 
        self.melspec_branch = FeatureExtractor(1)
        self.chroma_branch = FeatureExtractor(1)
        
        fusion_dim = 256 * 3  # 三个分支的特征维度之和 (每个 FeatureExtractor 输出 256 维)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE_FUSION), 
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE_FUSION) 
        )
        
        self.adaptive_fusion = AdaptiveFusion(fusion_dim // 4)
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim // 4, num_classes),
            nn.LogSoftmax(dim=1)
        )
        
        self._initialize_weights() 
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    @autocast()  # 启用自动混合精度
    def forward(self, mfcc, mel_spec, chroma):
        # 确保输入张量有正确的维度 (batch_size, channels, freq_bins, time_frames)
        # 例如，mfcc (N, 120, 259) -> mfcc.unsqueeze(1) (N, 1, 120, 259)
        # 因为 FeatureExtractor 期望输入是 (N, C, H, W)
        if mfcc.dim() == 3: 
            mfcc = mfcc.unsqueeze(1) 
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)
        if chroma.dim() == 3:
            chroma = chroma.unsqueeze(1)
        
        mfcc_features = self.mfcc_branch(mfcc).squeeze(-1).squeeze(-1) # 256维
        mel_features = self.melspec_branch(mel_spec).squeeze(-1).squeeze(-1) # 256维
        chroma_features = self.chroma_branch(chroma).squeeze(-1).squeeze(-1) # 256维
        
        combined = torch.cat([mfcc_features, mel_features, chroma_features], dim=1) # 256*3 维
        
        fused_mlp = self.fusion_mlp(combined) # 256*3 // 4 维
        
        fused_attended = self.adaptive_fusion(fused_mlp) # 256*3 // 4 维
        
        return self.classifier(fused_attended)

第3单元

# --- 4. 数据处理、训练流程与主函数 ---
import os
import numpy as np
import torch
import torch.optim as optim
import librosa
import warnings
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import GradScaler
import torch.multiprocessing as mp
import joblib 
import time 

warnings.filterwarnings('ignore')

# 改进的数据增强 (时域增强)
class AdvancedAudioAugmenter:
    def __init__(self, sr=Config.SR):
        self.sr = sr
        self.pitch_shift_steps_range = Config.PITCH_SHIFT_STEPS_RANGE
        self.time_stretch_rate_range = Config.TIME_STRETCH_RATE_RANGE
        self.noise_level = Config.NOISE_LEVEL
        
    def random_shift(self, y):
        # 确保有足够的长度进行移位
        shift_max = int(0.1 * self.sr) # 最大偏移 0.1 秒
        if len(y) <= shift_max * 2: 
            return y
        shift = np.random.randint(-shift_max, shift_max)
        return np.roll(y, shift)
    
    def pitch_time_augment(self, y):
        rate = np.random.uniform(*self.time_stretch_rate_range)
        pitch_steps = np.random.uniform(*self.pitch_shift_steps_range)
        
        y_stretched = librosa.effects.time_stretch(y, rate=rate)
        
        # 将拉伸后的音频统一回原始长度，避免维度不一致问题
        target_len = len(y)
        if len(y_stretched) > target_len:
            y_stretched = y_stretched[:target_len]
        else:
            y_stretched = np.pad(y_stretched, (0, target_len - len(y_stretched)))

        return librosa.effects.pitch_shift(y_stretched, sr=self.sr, n_steps=pitch_steps)
    
    def add_noise(self, y):
        noise = np.random.normal(0, self.noise_level, len(y))
        y_noisy = y + noise
        y_noisy = np.clip(y_noisy, -1.0, 1.0) # 音频通常在 -1 到 1 之间
        return y_noisy

class AudioFeatureExtractor:
    def __init__(self, sr=Config.SR, n_mels=Config.N_MELS, n_mfcc=Config.N_MFCC, n_frames=Config.N_FRAMES):
        self.sr = sr
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_frames = n_frames
        self.spec_augment_processor = SpecAugment() 
        self.audio_augmenter = AdvancedAudioAugmenter() 
        self.augment_prob_time_domain = Config.AUGMENT_PROB_TIME_DOMAIN 

    def load_and_preprocess(self, file_path):
        try:
            y, _ = librosa.load(file_path, sr=self.sr)
            target_length = int(self.sr * Config.AUDIO_TARGET_LENGTH_SEC) 
            if len(y) > target_length:
                y = y[:target_length]
            else:
                y = np.pad(y, (0, target_length - len(y)))

            y, _ = librosa.effects.trim(y, top_db=20)
            y = librosa.effects.preemphasis(y)
            y = librosa.util.normalize(y)
            return torch.from_numpy(y).float()
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            return None
            
    def extract_features(self, y_tensor, apply_spec_augment=True, apply_audio_augment=False):
        try:
            y_np = y_tensor.numpy()
            
            # --- 时域音频增强 (如果启用且随机概率命中) ---
            if apply_audio_augment and np.random.rand() < self.augment_prob_time_domain:
                aug_type = np.random.choice(['pitch_time', 'noise', 'shift'])
                if aug_type == 'pitch_time':
                    y_np = self.audio_augmenter.pitch_time_augment(y_np)
                elif aug_type == 'noise':
                    y_np = self.audio_augmenter.add_noise(y_np)
                elif aug_type == 'shift':
                    y_np = self.audio_augmenter.random_shift(y_np)
                y_np = librosa.util.normalize(y_np) # 增强后重新标准化

            # MFCC特征提取 (包括 delta 和 delta-delta)
            mfcc = librosa.feature.mfcc(y=y_np, sr=self.sr, n_mfcc=self.n_mfcc)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            mfcc_combined = np.vstack((mfcc, mfcc_delta, mfcc_delta2)) # 维度: (n_mfcc*3, n_frames)

            # 梅尔频谱图特征提取
            mel_spec = librosa.feature.melspectrogram(y=y_np, sr=self.sr, n_mels=self.n_mels)
            mel_spec = librosa.power_to_db(mel_spec) # 维度: (n_mels, n_frames)

            # 色度图特征提取
            chroma = librosa.feature.chroma_stft(y=y_np, sr=self.sr) # 维度: (12, n_frames)
            
            # 统一特征维度 (填充或截断到 n_frames)
            mfcc_padded = self._pad_or_truncate(mfcc_combined, target_rows=Config.N_MFCC * 3)
            mel_spec_padded = self._pad_or_truncate(mel_spec, target_rows=Config.N_MELS)
            chroma_padded = self._pad_or_truncate(chroma, target_rows=12) # Chroma 固定12维

            # 将特征转换为 PyTorch 张量并添加通道维度，以便模型处理
            # (freq_bins, time_frames) -> (1, freq_bins, time_frames)
            mfcc_tensor = torch.from_numpy(mfcc_padded).float() 
            mel_spec_tensor = torch.from_numpy(mel_spec_padded).float()
            chroma_tensor = torch.from_numpy(chroma_padded).float()
            
            # --- 频谱增强 (如果启用) ---
            if apply_spec_augment:
                # SpecAugment 期望输入 (batch_size, channels, freq_bins, time_frames) 或 (channels, freq_bins, time_frames)
                # 这里先 unsqueeze(0) 添加一个假的 batch 维度，再 unsqueeze(0) 添加 channel 维度
                mfcc_tensor = self.spec_augment_processor.augment(mfcc_tensor.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                mel_spec_tensor = self.spec_augment_processor.augment(mel_spec_tensor.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                # Chroma 通常不进行 SpecAugment，因为它不是基于能量的频谱图
                
            return {
                'mfcc': mfcc_tensor, 
                'mel_spec': mel_spec_tensor, 
                'chroma': chroma_tensor
            }
        except Exception as e:
            print(f"特征提取错误: {str(e)}")
            return None
    
    def _pad_or_truncate(self, feature, target_rows=None):
        # target_rows 用于验证特征的第一个维度是否正确
        if target_rows is not None and feature.shape[0] != target_rows:
            # 这是一个警告，通常不应该发生如果前处理正确
            print(f"警告: 特征 {feature.shape} 的行数与预期 {target_rows} 不符。")

        if feature.shape[1] > self.n_frames:
            return feature[:, :self.n_frames]
        else:
            return np.pad(feature, ((0, 0), (0, self.n_frames - feature.shape[1])))

class CryDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'mfcc': self.features[idx]['mfcc'],
            'mel_spec': self.features[idx]['mel_spec'],
            'chroma': self.features[idx]['chroma'],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, fold_idx=None):
    best_val_acc = 0.0
    patience_counter = 0
    scaler = GradScaler() 

    model = model.to(device)

    print("\n--- 开始训练 ---")
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            mfcc_orig = batch['mfcc'].to(device)
            mel_spec_orig = batch['mel_spec'].to(device)
            chroma_orig = batch['chroma'].to(device)
            labels_orig = batch['label'].to(device)

            optimizer.zero_grad()
            
            if Config.APPLY_MIXUP:
                mixed_mfcc, mixed_mel_spec, mixed_chroma, y_a, y_b, lam = mixup_data(
                    mfcc_orig, mel_spec_orig, chroma_orig, labels_orig, alpha=Config.MIXUP_ALPHA, device=device
                )
            else:
                mixed_mfcc, mixed_mel_spec, mixed_chroma, y_a, y_b, lam = \
                    mfcc_orig, mel_spec_orig, chroma_orig, labels_orig, labels_orig, 1.0

            with autocast(): # 启用混合精度
                outputs = model(mixed_mfcc, mixed_mel_spec, mixed_chroma)
                # Mixup 损失
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

            scaler.scale(loss).backward() # 缩放损失并反向传播
            scaler.step(optimizer)        # 更新优化器
            scaler.update()               # 更新缩放器

            scheduler.step() # OneCycleLR 在每个 batch 之后更新

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels_orig.size(0)
            
            # Mixup 后的训练准确率计算通常比较复杂，这里简化为加权准确率
            train_correct += (lam * predicted.eq(y_a).sum().item() + (1 - lam) * predicted.eq(y_b).sum().item())

            if (batch_idx + 1) % 10 == 0:
                print(f'Batch [{batch_idx + 1}/{len(train_loader)}] '
                      f'Loss: {train_loss/(batch_idx+1):.4f} '
                      f'Acc: {100.*train_correct/train_total:.2f}% '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # 验证阶段 (验证集不进行 Mixup)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                mfcc = batch['mfcc'].to(device)
                mel_spec = batch['mel_spec'].to(device)
                chroma = batch['chroma'].to(device)
                labels = batch['label'].to(device)
                
                with autocast(): # 验证阶段也使用混合精度
                    outputs = model(mfcc, mel_spec, chroma)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        print(f'\nTrain Loss: {train_loss/len(train_loader):.4f} Train Acc: {100.*train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f} Val Acc: {100.*val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            print(f'保存模型: 当前验证准确率 {val_acc:.4f} 优于历史最佳 {best_val_acc:.4f}')
            best_val_acc = val_acc
            os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
            model_filename = Config.BEST_MODEL_FILENAME.format(fold_idx) if fold_idx is not None else Config.BEST_MODEL_FILENAME.format('final')
            torch.save(model.state_dict(), os.path.join(Config.MODEL_SAVE_DIR, model_filename))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= Config.PATIENCE:
            print(f'Early stopping at epoch {epoch+1} due to no improvement for {Config.PATIENCE} epochs.')
            break
            
    return best_val_acc

# 独立函数用于处理原始音频数据，方便在主函数中调用和缓存
def process_raw_audio_data():
    processor = AudioFeatureExtractor() 
    features = []
    labels = []
    
    start_time = time.time()
    all_raw_files = []
    
    # 收集所有符合条件的原始文件路径和标签
    for label in os.listdir(Config.RAW_DATA_PATH):
        folder_path = os.path.join(Config.RAW_DATA_PATH, label)
        if not os.path.isdir(folder_path):
            continue
        
        current_label_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        if len(current_label_files) < Config.MIN_SAMPLES_PER_CLASS:
            print(f"跳过类别 '{label}' (样本数: {len(current_label_files)} < {Config.MIN_SAMPLES_PER_CLASS})")
            continue
            
        print(f"准备处理类别: '{label}' ({len(current_label_files)} 个文件)")
        for file_name in current_label_files:
            file_path = os.path.join(folder_path, file_name)
            all_raw_files.append((file_path, label))

    print(f"共发现 {len(all_raw_files)} 个符合条件的原始音频文件，开始特征提取...")

    # 循环处理所有文件
    for i, (file_path, label) in enumerate(all_raw_files):
        if (i + 1) % 50 == 0:
            print(f"处理进度: {i+1}/{len(all_raw_files)}...")
        
        y_tensor = processor.load_and_preprocess(file_path)
        
        if y_tensor is not None:
            # 提取原始特征 (不进行增强)
            features_orig = processor.extract_features(y_tensor, apply_spec_augment=False, apply_audio_augment=False)
            if features_orig is not None:
                features.append(features_orig)
                labels.append(label)
            
            # 提取增强特征 (同时应用 SpecAugment 和 时域增强)
            # 确保增强后的样本量是原始样本量的两倍，或者根据 Config.AUGMENT_PROB_TIME_DOMAIN 进行控制
            features_aug = processor.extract_features(y_tensor, apply_spec_augment=True, apply_audio_augment=Config.APPLY_AUDIO_AUGMENT)
            if features_aug is not None:
                features.append(features_aug)
                labels.append(label)
    
    end_time = time.time()
    print(f"原始音频数据处理耗时: {end_time - start_time:.2f} 秒")

    # 提前编码标签以备用
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    return features, labels, label_encoder

# 独立函数用于保存处理后的数据
def save_processed_data(features, labels, label_encoder, filepath):
    print(f"\n处理完成，共收集 {len(features)} 个样本。")
    print(f"正在保存处理后的特征和标签到 {filepath}...")
    save_data = {
        'features': features,
        'labels': labels,
        'classes': label_encoder.classes_ 
    }
    joblib.dump(save_data, filepath)
    print("数据保存完成。您可以在后续运行中直接加载此文件。")


def main():
    print("初始化...")
    # 设置随机种子，确保可复现性
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # 禁用 cuDNN 自动调优，以保证可复现性

    # 创建必要的目录
    os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)

    processed_data_filepath = os.path.join(Config.PROCESSED_DATA_DIR, Config.PROCESSED_DATA_FILENAME)
    
    features = []
    labels = []
    label_encoder = LabelEncoder()
    
    # --- 数据加载或处理逻辑 ---
    if os.path.exists(processed_data_filepath):
        print(f"\n检测到已处理数据文件: {processed_data_filepath}，正在加载...")
        try:
            loaded_data = joblib.load(processed_data_filepath)
            features = loaded_data['features']
            labels = loaded_data['labels']
            label_encoder.classes_ = loaded_data['classes'] 
            print(f"数据加载完成，共 {len(features)} 个样本。")
        except Exception as e:
            print(f"加载数据文件时出错: {e}。将重新处理原始数据。")
            features, labels, label_encoder = process_raw_audio_data()
            save_processed_data(features, labels, label_encoder, processed_data_filepath)
    else:
        print("\n未检测到已处理数据文件，正在处理原始音频数据...")
        features, labels, label_encoder = process_raw_audio_data()
        save_processed_data(features, labels, label_encoder, processed_data_filepath)

    num_classes = len(label_encoder.classes_)
    print(f"最终类别数量: {num_classes}")
    
    # K-Fold Cross-Validation 设置
    skf = StratifiedKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.RANDOM_SEED)
    
    # 将 list of dict 转换为 np.array of object，以便 StratifiedKFold 正确索引
    X_data = np.array(features, dtype=object) 
    y_data = np.array(label_encoder.transform(labels)) 

    fold_accuracies = []

    print(f"\n--- 开始 {Config.N_SPLITS}-Fold 交叉验证 ---")
    for fold_idx, (train_index, val_index) in enumerate(skf.split(X_data, y_data)):
        print(f"\n======== Fold {fold_idx + 1}/{Config.N_SPLITS} ========")
        
        # 为当前折叠划分数据集
        # .tolist() 将 numpy object 数组转换回 list of dicts，以便 CryDataset 接受
        X_train_fold, X_val_fold = X_data[train_index].tolist(), X_data[val_index].tolist()
        y_train_fold, y_val_fold = y_data[train_index].tolist(), y_data[val_index].tolist()

        # 创建数据加载器
        train_dataset_fold = CryDataset(X_train_fold, y_train_fold) 
        val_dataset_fold = CryDataset(X_val_fold, y_val_fold)
        
        train_loader_fold = DataLoader(
            train_dataset_fold, 
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=0, # 保持为0以避免在交互式环境中出现 pickle 错误
            pin_memory=True # 在GPU可用时，将数据预加载到GPU显存
        )
        
        val_loader_fold = DataLoader(
            val_dataset_fold, 
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=0, # 保持为0
            pin_memory=True
        )
        
        # 每次折叠都重新初始化模型、优化器和调度器
        model = CryCNN_V13_Ultimate(num_classes).to(device) # 使用终极版模型类名
        criterion = LabelSmoothingLoss(smoothing=Config.LABEL_SMOOTHING)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.INITIAL_LR,
            weight_decay=Config.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=Config.MAX_LR,
            epochs=Config.NUM_EPOCHS,
            steps_per_epoch=len(train_loader_fold), 
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # 训练和验证当前折叠
        fold_best_acc = train_and_validate(
            model=model,
            train_loader=train_loader_fold,
            val_loader=val_loader_fold,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=Config.NUM_EPOCHS,
            device=device,
            fold_idx=fold_idx + 1 # 传递折叠索引给保存文件名
        )
        fold_accuracies.append(fold_best_acc)
        print(f"Fold {fold_idx + 1} 最佳验证准确率: {fold_best_acc:.4f}")
    
    # 打印 K-Fold 结果
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\n--- K-Fold 交叉验证结果 ({Config.N_SPLITS} 折) ---")
    print(f"各折叠最佳验证准确率: {fold_accuracies}")
    print(f"平均验证准确率: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"训练完成!")


if __name__ == '__main__':
    # 确保在多进程（如 DataLoader）中使用 'spawn' 启动方法
    # 这在某些操作系统（如 Windows）和环境中是必需的
    mp.set_start_method('spawn', force=True)
    main()

(二) 014版代码
第1单元

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import warnings
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold # 确保这里只导入KFold相关
from sklearn.preprocessing import StandardScaler, LabelEncoder # <-- 修正：StandardScaler 从这里导入
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR 
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
import joblib 
import time 

warnings.filterwarnings('ignore')

# --- 1. 配置参数段落 ---
class Config:
    # 数据路径配置
    RAW_DATA_PATH = '/kaggle/input/baby-kusen/Raw data'
    # 处理后的特征数据保存路径，建议在 /kaggle/working/ 下创建子目录
    PROCESSED_DATA_DIR = '/kaggle/working/processed_features'
    PROCESSED_DATA_FILENAME = 'baby_cry_features_v14_b.pkl' # 方便管理版本
    MODEL_SAVE_DIR = '/kaggle/working/output' # 模型保存目录
    BEST_MODEL_FILENAME = 'best_cry_classifier_v14_b_fold_{}.pth' # 最佳模型文件名 (K-Fold专用)

    # 音频特征提取参数
    SR = 22050           # 采样率 (Default: 22050)
    N_MELS = 128         # 梅尔频谱图的梅尔带数量 (Default: 128)
    N_MFCC = 40          # 基础MFCC特征数量 (Default: 40)
    N_FRAMES = 259       # 统一的特征时间帧数 (Default: 259)

    # 数据增强参数
    SPEC_AUGMENT_TIME_MASK_PARAM = 20 # SpecAugment时间遮蔽参数 (Default: 20, 范围: 10-40)
    SPEC_AUGMENT_FREQ_MASK_PARAM = 10 # SpecAugment频率遮蔽参数 (Default: 10, 范围: 5-20)
    
    # 时域数据增强参数 (适度激进的设置)
    AUGMENT_PROB_TIME_DOMAIN = 0.4   # 应用时域数据增强的概率 (Default: 0.3, 增加到 0.4)
    PITCH_SHIFT_STEPS_RANGE = (-2, 2) # 音高变换步长范围 (Default: (-1.5, 1.5), 稍大范围)
    TIME_STRETCH_RATE_RANGE = (0.95, 1.05) # 时间拉伸比率范围 (Default: (0.98, 1.02), 稍大范围)
    NOISE_LEVEL = 0.001  # 添加噪声的强度 (Default: 0.0005, 稍高)
    MIXUP_ALPHA = 0.4    # Mixup 数据增强的 Alpha 参数 (Default: 0.3, 增加到 0.4)
    
    # 背景噪音路径 (重要！需要确保此路径下有噪音文件)
    BACKGROUND_NOISE_PATH = '/kaggle/input/environmental-sound-dataset/audio_files/noise' # 请替换为实际噪音文件路径
    NOISE_AUGMENT_PROB = 0.3 # 添加背景噪音的概率 (Default: 0.3, 范围: 0.1-0.5)


    # 训练参数
    BATCH_SIZE = 32      # 批处理大小 (Default: 32, 范围: 16-64)
    NUM_EPOCHS = 120     # 总训练轮次 (Default: 100, 增加到 120)
    INITIAL_LR = 0.0005  # AdamW 初始学习率 (OneCycleLR 会覆盖，但作为参考)
    MAX_LR = 0.003       # OneCycleLR 的最大学习率 (Default: 0.004, 降低到 0.003)
    WEIGHT_DECAY = 0.01  # 优化器权重衰减 (Default: 0.01, 范围: 0.001-0.05)
    LABEL_SMOOTHING = 0.1 # 标签平滑强度 (Default: 0.1, 范围: 0.05-0.2)
    PATIENCE = 25        # 早停耐心值 (Default: 20, 增加到 25)

    # 模型参数
    DROPOUT_RATE_RESBLOCK = 0.15 # 残差块内的Dropout率 (Default: 0.15, 范围: 0.05-0.2)
    DROPOUT_RATE_FUSION = 0.3   # 融合层内的Dropout率 (Default: 0.3, 范围: 0.1-0.4)

    # K-Fold Cross-Validation 参数
    N_SPLITS = 5         # K-Fold 的折数 (Default: 5, 范围: 3-10)

    # 其他配置
    RANDOM_SEED = 42     # 随机种子 (Default: 42)
    MIN_SAMPLES_PER_CLASS = 50 # 跳过类别所需的最小样本数 (Default: 50)


# --- 2. 通用模块与核心组件 ---

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")

# Label Smoothing损失函数
class LabelSmoothingLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=Config.LABEL_SMOOTHING):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    def reduce_loss(self, loss):
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    def smooth_loss(self, log_prob, targets):
        with torch.no_grad():
            targets = torch.zeros_like(log_prob).scatter_(1, targets.unsqueeze(1), 1)
            targets = (1 - self.smoothing) * targets + self.smoothing / log_prob.size(-1)
        return torch.sum(-targets * log_prob, dim=-1)

    def forward(self, inputs, targets):
        log_prob = F.log_softmax(inputs, dim=-1)
        loss = self.smooth_loss(log_prob, targets)
        return self.reduce_loss(loss)

# 频谱增强
class SpecAugment:
    def __init__(self, time_mask_param=Config.SPEC_AUGMENT_TIME_MASK_PARAM, 
                 freq_mask_param=Config.SPEC_AUGMENT_FREQ_MASK_PARAM):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
    
    def time_mask(self, spec):
        if spec.dim() == 4:
            v = spec.size(3) 
            if v == 0: return spec 
            t = np.random.randint(0, self.time_mask_param)
            t = min(t, v - 1) 
            t0 = np.random.randint(0, v - t) if v > t else 0
            spec[:, :, :, t0:t0 + t] = 0
        elif spec.dim() == 3: 
            v = spec.size(2) 
            if v == 0: return spec
            t = np.random.randint(0, self.time_mask_param)
            t = min(t, v - 1)
            t0 = np.random.randint(0, v - t) if v > t else 0
            spec[:, :, t0:t0 + t] = 0
        return spec
    
    def freq_mask(self, spec):
        if spec.dim() == 4:
            v = spec.size(2)
            if v == 0: return spec
            f = np.random.randint(0, self.freq_mask_param)
            f = min(f, v - 1)
            f0 = np.random.randint(0, v - f) if v > f else 0
            spec[:, :, f0:f0 + f, :] = 0
        elif spec.dim() == 3: 
            v = spec.size(1)
            if v == 0: return spec
            f = np.random.randint(0, self.freq_mask_param)
            f = min(f, v - 1)
            f0 = np.random.randint(0, v - f) if v > f else 0
            spec[:, f0:f0 + f, :] = 0
        return spec

    def augment(self, spec):
        augmented_spec = spec.clone() 
        augmented_spec = self.freq_mask(augmented_spec)
        augmented_spec = self.time_mask(augmented_spec)
        return augmented_spec

# 自适应特征融合
class AdaptiveFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(channels, channels // 8),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 8, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        weights = self.attention(x)
        return x * weights 

# Mixup 数据增强函数
def mixup_data(x, y, alpha=Config.MIXUP_ALPHA, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# Mixup 损失计算函数
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

第2单元

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

# 改进的残差模块
class ImprovedResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 16, out_channels, 1),
            nn.Sigmoid()
        )
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.dropout = nn.Dropout2d(Config.DROPOUT_RATE_RESBLOCK) # 从Config获取
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out) * out
        out = self.dropout(out)
        return F.relu(out + self.shortcut(x))

# 特征提取器基类
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.features = nn.Sequential(
            ImprovedResBlock(in_channels, 32), 
            ImprovedResBlock(32, 64, stride=2),
            ImprovedResBlock(64, 128, stride=2),
            ImprovedResBlock(128, 256, stride=2)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = self.features(x)
        return self.gap(x)

# 改进的CNN模型 (V14)
class CryCNN_V14(nn.Module): 
    def __init__(self, num_classes):
        super().__init__()
        
        self.mfcc_branch = FeatureExtractor(1) 
        self.melspec_branch = FeatureExtractor(1)
        self.chroma_branch = FeatureExtractor(1)
        
        fusion_dim = 256 * 3  # 三个分支的特征维度之和
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE_FUSION), # 从Config获取
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(Config.DROPOUT_RATE_FUSION) # 从Config获取
        )
        
        self.adaptive_fusion = AdaptiveFusion(fusion_dim // 4)
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim // 4, num_classes),
            nn.LogSoftmax(dim=1)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    @autocast()  # 启用自动混合精度
    def forward(self, mfcc, mel_spec, chroma):
        if mfcc.dim() == 3: 
            mfcc = mfcc.unsqueeze(1) 
        if mel_spec.dim() == 3:
            mel_spec = mel_spec.unsqueeze(1)
        if chroma.dim() == 3:
            chroma = chroma.unsqueeze(1)
        
        mfcc_features = self.mfcc_branch(mfcc).squeeze(-1).squeeze(-1)
        mel_features = self.melspec_branch(mel_spec).squeeze(-1).squeeze(-1)
        chroma_features = self.chroma_branch(chroma).squeeze(-1).squeeze(-1)
        
        combined = torch.cat([mfcc_features, mel_features, chroma_features], dim=1)
        
        fused_mlp = self.fusion_mlp(combined)
        
        fused_attended = self.adaptive_fusion(fused_mlp)
        
        return self.classifier(fused_attended)

第3单元

import os
import numpy as np
import torch
import torch.optim as optim
import librosa
import warnings
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold # 导入 StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import GradScaler
import torch.multiprocessing as mp
import joblib 
import time 

warnings.filterwarnings('ignore')

# --- 3. 数据处理与训练流程 ---

# 改进的数据增强 (时域增强)
class AdvancedAudioAugmenter:
    def __init__(self, sr=Config.SR, 
                 pitch_shift_steps_range=Config.PITCH_SHIFT_STEPS_RANGE,
                 time_stretch_rate_range=Config.TIME_STRETCH_RATE_RANGE,
                 noise_level=Config.NOISE_LEVEL):
        self.sr = sr
        self.pitch_shift_steps_range = pitch_shift_steps_range
        self.time_stretch_rate_range = time_stretch_rate_range
        self.noise_level = noise_level
        self.noise_files = self._load_noise_files() # 加载背景噪音文件
        self.noise_augment_prob = Config.NOISE_AUGMENT_PROB # 从Config获取噪音增强概率

    def _load_noise_files(self):
        noise_files = []
        if os.path.exists(Config.BACKGROUND_NOISE_PATH):
            for f_name in os.listdir(Config.BACKGROUND_NOISE_PATH):
                if f_name.endswith('.wav'):
                    noise_files.append(os.path.join(Config.BACKGROUND_NOISE_PATH, f_name))
            print(f"检测到 {len(noise_files)} 个背景噪音文件。")
        else:
            print(f"警告: 未找到背景噪音路径: {Config.BACKGROUND_NOISE_PATH}。将跳过背景噪音增强。")
        return noise_files

    def random_shift(self, y):
        shift_max = int(0.1 * self.sr) # 最大偏移 0.1 秒
        if len(y) <= shift_max * 2: 
            return y
        shift = np.random.randint(-shift_max, shift_max)
        return np.roll(y, shift)
    
    def pitch_time_augment(self, y):
        rate = np.random.uniform(*self.time_stretch_rate_range)
        pitch_steps = np.random.uniform(*self.pitch_shift_steps_range)
        
        y_stretched = librosa.effects.time_stretch(y, rate=rate)
        if len(y_stretched) > len(y):
            y_stretched = y_stretched[:len(y)]
        else:
            y_stretched = np.pad(y_stretched, (0, len(y) - len(y_stretched)))

        return librosa.effects.pitch_shift(y_stretched, sr=self.sr, n_steps=pitch_steps)
    
    def add_noise(self, y):
        if not self.noise_files:
            return y # 如果没有噪音文件，直接返回原音频

        if np.random.rand() < self.noise_augment_prob:
            try:
                noise_file = np.random.choice(self.noise_files)
                noise_y, _ = librosa.load(noise_file, sr=self.sr)
                
                # 确保噪音和音频长度匹配
                if len(noise_y) > len(y):
                    start_idx = np.random.randint(0, len(noise_y) - len(y))
                    noise_y = noise_y[start_idx : start_idx + len(y)]
                elif len(noise_y) < len(y):
                    noise_y = np.pad(noise_y, (0, len(y) - len(noise_y)))
                
                # 调整噪音音量，并与原始音频混合
                amp_factor = np.random.uniform(0.01, 0.05) # 噪音强度因子，可以调优
                y_noisy = y + noise_y * amp_factor
                y_noisy = np.clip(y_noisy, -1.0, 1.0) # 音频通常在 -1 到 1 之间
                return y_noisy
            except Exception as e:
                print(f"添加背景噪音时出错: {e}。返回原始音频。")
                return y
        return y # 不应用噪音增强的概率

class AudioFeatureExtractor:
    def __init__(self, sr=Config.SR, n_mels=Config.N_MELS, n_mfcc=Config.N_MFCC, n_frames=Config.N_FRAMES):
        self.sr = sr
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_frames = n_frames
        self.spec_augment_processor = SpecAugment() 
        self.audio_augmenter = AdvancedAudioAugmenter() 
        self.augment_prob_time_domain = Config.AUGMENT_PROB_TIME_DOMAIN 

    def load_and_preprocess(self, file_path):
        try:
            y, _ = librosa.load(file_path, sr=self.sr)
            target_length = int(self.sr * 6) # 目标长度为6秒
            if len(y) > target_length:
                y = y[:target_length]
            else:
                y = np.pad(y, (0, target_length - len(y)))

            y, _ = librosa.effects.trim(y, top_db=20)
            y = librosa.effects.preemphasis(y)
            y = librosa.util.normalize(y)
            return torch.from_numpy(y).float()
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            return None
            
    def extract_features(self, y_tensor, apply_spec_augment=True, apply_audio_augment=False):
        try:
            y_np = y_tensor.numpy()
            
            # --- 时域音频增强 (14版：根据概率应用时域增强，包括新加的背景噪音) ---
            if apply_audio_augment and np.random.rand() < self.augment_prob_time_domain:
                aug_type = np.random.choice(['pitch_time', 'noise', 'shift', 'none']) # 'none' 选项，降低增强概率
                if aug_type == 'pitch_time':
                    y_np = self.audio_augmenter.pitch_time_augment(y_np)
                elif aug_type == 'noise':
                    y_np = self.audio_augmenter.add_noise(y_np)
                elif aug_type == 'shift':
                    y_np = self.audio_augmenter.random_shift(y_np)
                # else: 'none' 情况下不进行增强

                y_np = librosa.util.normalize(y_np) # 增强后重新标准化

            # MFCC特征提取
            mfcc = librosa.feature.mfcc(y=y_np, sr=self.sr, n_mfcc=self.n_mfcc)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            mfcc_combined = np.vstack((mfcc, mfcc_delta, mfcc_delta2))

            # 梅尔频谱图特征提取
            mel_spec = librosa.feature.melspectrogram(y=y_np, sr=self.sr, n_mels=self.n_mels)
            mel_spec = librosa.power_to_db(mel_spec)
            # 色度图特征提取
            chroma = librosa.feature.chroma_stft(y=y_np, sr=self.sr)
            
            # 统一特征维度
            mfcc_padded = self._pad_or_truncate(mfcc_combined) 
            mel_spec_padded = self._pad_or_truncate(mel_spec)
            chroma_padded = self._pad_or_truncate(chroma)

            # 将特征转换为 PyTorch 张量并添加通道维度，以便 SpecAugment 处理
            mfcc_tensor_spec = torch.from_numpy(mfcc_padded).float().unsqueeze(0).unsqueeze(0) 
            mel_spec_tensor_spec = torch.from_numpy(mel_spec_padded).float().unsqueeze(0).unsqueeze(0)
            chroma_tensor_spec = torch.from_numpy(chroma_padded).float().unsqueeze(0).unsqueeze(0) 

            # --- 频谱增强 (可选) ---
            if apply_spec_augment:
                mfcc_tensor_spec = self.spec_augment_processor.augment(mfcc_tensor_spec)
                mel_spec_tensor_spec = self.spec_augment_processor.augment(mel_spec_tensor_spec)
                
            # 移除 batch 和 channel 维度，返回适合模型输入的字典，期望是 (freq_bins, time_frames)
            return {
                'mfcc': mfcc_tensor_spec.squeeze(0).squeeze(0), 
                'mel_spec': mel_spec_tensor_spec.squeeze(0).squeeze(0), 
                'chroma': chroma_tensor_spec.squeeze(0).squeeze(0) 
            }
        except Exception as e:
            print(f"特征提取错误: {str(e)}")
            return None
    
    def _pad_or_truncate(self, feature):
        # 警告：特征维度检查。在 14 版中，mfcc_padded 的第一个维度是 N_MFCC * 3
        # 所以这里的检查逻辑需要考虑
        # if feature.shape[0] != Config.N_MFCC * 3 and feature.shape[0] != Config.N_MELS and feature.shape[0] != 12:
        #     print(f"警告: 特征维度 {feature.shape[0]} 与预期 {Config.N_MFCC*3}, {Config.N_MELS} 或 12 不符。")

        if feature.shape[1] > self.n_frames:
            return feature[:, :self.n_frames]
        else:
            return np.pad(feature, ((0, 0), (0, self.n_frames - feature.shape[1])))

class CryDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'mfcc': self.features[idx]['mfcc'],
            'mel_spec': self.features[idx]['mel_spec'],
            'chroma': self.features[idx]['chroma'],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, fold_idx=None):
    best_val_acc = 0.0
    patience_counter = 0
    scaler = GradScaler() 

    model = model.to(device)

    print("\n--- 开始训练 ---")
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            mfcc_orig = batch['mfcc'].to(device)
            mel_spec_orig = batch['mel_spec'].to(device)
            chroma_orig = batch['chroma'].to(device)
            labels_orig = batch['label'].to(device)

            optimizer.zero_grad()
            
            # Mixup 数据增强
            batch_size = mfcc_orig.size()[0]
            index = torch.randperm(batch_size).to(device)
            lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)
            
            mixed_mfcc = lam * mfcc_orig + (1 - lam) * mfcc_orig[index]
            mixed_mel_spec = lam * mel_spec_orig + (1 - lam) * mel_spec_orig[index]
            mixed_chroma = lam * chroma_orig + (1 - lam) * chroma_orig[index]
            y_a, y_b = labels_orig, labels_orig[index]


            with autocast(): # 启用混合精度
                outputs = model(mixed_mfcc, mixed_mel_spec, mixed_chroma)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

            scaler.scale(loss).backward() # 缩放损失并反向传播
            scaler.step(optimizer)        # 更新优化器
            scaler.update()               # 更新缩放器

            scheduler.step() # OneCycleLR 在每个 batch 之后更新

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels_orig.size(0)
            
            # Mixup 后的训练准确率计算通常比较复杂，这里简化为只看 y_a 的准确率
            # 实际训练通常更关注 loss 的下降
            train_correct += (lam * predicted.eq(y_a).sum().item() + (1 - lam) * predicted.eq(y_b).sum().item())

            if (batch_idx + 1) % 10 == 0:
                print(f'Batch [{batch_idx + 1}/{len(train_loader)}] '
                      f'Loss: {train_loss/(batch_idx+1):.4f} '
                      f'Acc: {100.*train_correct/train_total:.2f}% '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # 验证阶段 (验证集不进行 Mixup)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                mfcc = batch['mfcc'].to(device)
                mel_spec = batch['mel_spec'].to(device)
                chroma = batch['chroma'].to(device)
                labels = batch['label'].to(device)
                
                with autocast(): # 验证阶段也使用混合精度
                    outputs = model(mfcc, mel_spec, chroma)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        print(f'\nTrain Loss: {train_loss/len(train_loader):.4f} Train Acc: {100.*train_acc:.2f}%')
        print(f'Val Loss: {val_loss/len(val_loader):.4f} Val Acc: {100.*val_acc:.2f}%')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            print(f'保存模型: 当前验证准确率 {val_acc:.4f} 优于历史最佳 {best_val_acc:.4f}')
            best_val_acc = val_acc
            # 确保模型保存目录存在
            os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True)
            model_filename = Config.BEST_MODEL_FILENAME.format(fold_idx) if fold_idx is not None else Config.BEST_MODEL_FILENAME.format('final')
            torch.save(model.state_dict(), os.path.join(Config.MODEL_SAVE_DIR, model_filename))
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 早停检查
        if patience_counter >= Config.PATIENCE:
            print(f'Early stopping at epoch {epoch+1} due to no improvement for {Config.PATIENCE} epochs.')
            break
            
    return best_val_acc

def main():
    print("初始化...")
    torch.manual_seed(Config.RANDOM_SEED)
    np.random.seed(Config.RANDOM_SEED)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 

    os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_SAVE_DIR, exist_ok=True) # 确保模型保存目录存在

    processed_data_filepath = os.path.join(Config.PROCESSED_DATA_DIR, Config.PROCESSED_DATA_FILENAME)
    
    features = []
    labels = []
    label_encoder = LabelEncoder()
    
    # --- 数据加载或处理 ---
    if os.path.exists(processed_data_filepath):
        print(f"\n检测到已处理数据文件: {processed_data_filepath}，正在加载...")
        try:
            loaded_data = joblib.load(processed_data_filepath)
            features = loaded_data['features']
            labels = loaded_data['labels']
            label_encoder.classes_ = loaded_data['classes'] 
            print(f"数据加载完成，共 {len(features)} 个样本。")
        except Exception as e:
            print(f"加载数据文件时出错: {e}。将重新处理原始数据。")
            features, labels, label_encoder = process_raw_audio_data()
            save_processed_data(features, labels, label_encoder, processed_data_filepath)
    else:
        print("\n未检测到已处理数据文件，正在处理原始音频数据...")
        features, labels, label_encoder = process_raw_audio_data()
        save_processed_data(features, labels, label_encoder, processed_data_filepath)

    num_classes = len(label_encoder.classes_)
    print(f"类别数量: {num_classes}")
    
    # K-Fold Cross-Validation 设置
    skf = StratifiedKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.RANDOM_SEED)
    
    X_data = np.array(features, dtype=object) 
    y_data = np.array(label_encoder.transform(labels)) 

    fold_accuracies = []

    print(f"\n--- 开始 {Config.N_SPLITS}-Fold 交叉验证 ---")
    for fold_idx, (train_index, val_index) in enumerate(skf.split(X_data, y_data)):
        print(f"\n======== Fold {fold_idx + 1}/{Config.N_SPLITS} ========")
        
        # 为当前折叠划分数据集
        X_train_fold, X_val_fold = X_data[train_index], X_data[val_index]
        y_train_fold, y_val_fold = y_data[train_index], y_data[val_index]

        # 创建数据加载器
        train_dataset_fold = CryDataset(X_train_fold.tolist(), y_train_fold.tolist()) 
        val_dataset_fold = CryDataset(X_val_fold.tolist(), y_val_fold.tolist())
        
        train_loader_fold = DataLoader(
            train_dataset_fold, 
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=0, 
            pin_memory=True
        )
        
        val_loader_fold = DataLoader(
            val_dataset_fold, 
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=0, 
            pin_memory=True
        )
        
        # 每次折叠都重新初始化模型、优化器和调度器
        model = CryCNN_V14(num_classes).to(device) # 使用新的模型类名
        criterion = LabelSmoothingLoss(smoothing=Config.LABEL_SMOOTHING)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.INITIAL_LR,
            weight_decay=Config.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=Config.MAX_LR,
            epochs=Config.NUM_EPOCHS,
            steps_per_epoch=len(train_loader_fold), 
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # 训练和验证当前折叠
        fold_best_acc = train_and_validate(
            model=model,
            train_loader=train_loader_fold,
            val_loader=val_loader_fold,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=Config.NUM_EPOCHS,
            device=device,
            fold_idx=fold_idx + 1 
        )
        fold_accuracies.append(fold_best_acc)
        print(f"Fold {fold_idx + 1} 最佳验证准确率: {fold_best_acc:.4f}")
    
    # 打印 K-Fold 结果
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\n--- K-Fold 交叉验证结果 ({Config.N_SPLITS} 折) ---")
    print(f"各折叠最佳验证准确率: {fold_accuracies}")
    print(f"平均验证准确率: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"训练完成!")


# 独立函数用于处理原始音频数据，方便在主函数中调用和缓存
def process_raw_audio_data():
    processor = AudioFeatureExtractor() 
    features = []
    labels = []
    
    start_time = time.time()
    all_raw_files = []
    
    for label in os.listdir(Config.RAW_DATA_PATH):
        folder_path = os.path.join(Config.RAW_DATA_PATH, label)
        if not os.path.isdir(folder_path):
            continue
        
        current_label_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        if len(current_label_files) < Config.MIN_SAMPLES_PER_CLASS:
            print(f"跳过类别 {label} (样本数: {len(current_label_files)})")
            continue
            
        print(f"准备处理类别: {label}")
        for file_name in current_label_files:
            file_path = os.path.join(folder_path, file_name)
            all_raw_files.append((file_path, label))

    print(f"共发现 {len(all_raw_files)} 个符合条件的原始音频文件。")

    # 循环处理所有文件
    for i, (file_path, label) in enumerate(all_raw_files):
        if (i + 1) % 50 == 0:
            print(f"处理进度: {i+1}/{len(all_raw_files)}...")
        
        y_tensor = processor.load_and_preprocess(file_path)
        
        if y_tensor is not None:
            # 提取原始特征 (不进行任何增强)
            features_orig = processor.extract_features(y_tensor, apply_spec_augment=False, apply_audio_augment=False)
            if features_orig is not None:
                features.append(features_orig)
                labels.append(label)
            
            # 提取增强特征 (同时应用 SpecAugment 和 时域增强)
            features_aug = processor.extract_features(y_tensor, apply_spec_augment=True, apply_audio_augment=True)
            if features_aug is not None:
                features.append(features_aug)
                labels.append(label)
    
    end_time = time.time()
    print(f"原始音频数据处理耗时: {end_time - start_time:.2f} 秒")

    # 提前编码标签以备用
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)
    return features, labels, label_encoder

# 独立函数用于保存处理后的数据
def save_processed_data(features, labels, label_encoder, filepath):
    print(f"\n处理完成，共收集 {len(features)} 个样本。")
    print(f"正在保存处理后的特征和标签到 {filepath}...")
    save_data = {
        'features': features,
        'labels': labels,
        'classes': label_encoder.classes_ 
    }
    joblib.dump(save_data, filepath)
    print("数据保存完成。您可以下载此文件并在后续运行中直接加载。")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()



第二阶段工作
（以013版生成的best_cry_classifier_v13_fold_3.pth为例）

(一) export_pytorch_to_onnx代码

# --- 0. 导入必要的库 ---
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa # 用于 Config 中的常量，以及模型定义完整性
import warnings

warnings.filterwarnings('ignore')

# --- 1. 配置参数段落 (与训练时的 Config 类保持完全一致) ---
# 确保这里的参数与您训练 CryCNN_V13 时的 config.py 或 Config 类定义完全相同
class Config:
    SR = 22050           # 采样率
    N_MELS = 128         # 梅尔频谱图的梅尔带数量
    N_MFCC = 40          # 基础MFCC特征数量 (模型实际处理的是 N_MFCC * 3，即 120)
    N_FRAMES = 259       # 统一的特征时间帧数
    NUM_CLASSES = 4 # 确保这个数量是正确的

    # Dummy parameters for model definition consistency
    DROPOUT_RATE_RESBLOCK = 0.15
    DROPOUT_RATE_FUSION = 0.3
    LABEL_SMOOTHING = 0.1
    SPEC_AUGMENT_TIME_MASK_PARAM = 20
    SPEC_AUGMENT_FREQ_MASK_PARAM = 10
    MIXUP_ALPHA = 0.3

# --- 2. 通用模块与核心组件 (CryCNN_V13 模型定义 - 与您训练时的模型代码完全一致) ---

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out) # Apply dropout after first ReLU
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CryCNN_V13(nn.Module):
    def __init__(self, config):
        super(CryCNN_V13, self).__init__()
        self.config = config
        self.in_channels = 64 # Initial in_channels for _make_layer context

        # Feature-specific initial convolutions
        self.conv1_mfcc = nn.Conv2d(config.N_MFCC * 3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_mfcc = nn.BatchNorm2d(64)
        self.conv1_mel = nn.Conv2d(config.N_MELS, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_mel = nn.BatchNorm2d(64)
        self.conv1_chroma = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1, bias=False) # 12 is fixed for chroma
        self.bn1_chroma = nn.BatchNorm2d(64)

        # To handle the 'in_channels' state for _make_layer correctly across branches
        # We temporarily save and restore 'in_channels' for each branch's layer construction.
        # This is a slightly unusual pattern but necessary if 'in_channels' is a class member
        # and _make_layer is called multiple times for different sequential blocks.

        # MFCC blocks
        self._temp_in_channels = 64 # Initial value for MFCC branch
        self.mfcc_layer1 = self._make_layer_branch(BasicBlock, 64, 2, stride=1, dropout_rate=config.DROPOUT_RATE_RESBLOCK)
        self.mfcc_layer2 = self._make_layer_branch(BasicBlock, 128, 2, stride=2, dropout_rate=config.DROPOUT_RATE_RESBLOCK)
        self.mfcc_layer3 = self._make_layer_branch(BasicBlock, 256, 2, stride=2, dropout_rate=config.DROPOUT_RATE_RESBLOCK)

        # Mel-spectrogram blocks
        self._temp_in_channels = 64 # Initial value for Mel branch
        self.mel_layer1 = self._make_layer_branch(BasicBlock, 64, 2, stride=1, dropout_rate=config.DROPOUT_RATE_RESBLOCK)
        self.mel_layer2 = self._make_layer_branch(BasicBlock, 128, 2, stride=2, dropout_rate=config.DROPOUT_RATE_RESBLOCK)
        self.mel_layer3 = self._make_layer_branch(BasicBlock, 256, 2, stride=2, dropout_rate=config.DROPOUT_RATE_RESBLOCK)

        # Chroma blocks
        self._temp_in_channels = 64 # Initial value for Chroma branch
        self.chroma_layer1 = self._make_layer_branch(BasicBlock, 64, 2, stride=1, dropout_rate=config.DROPOUT_RATE_RESBLOCK)
        self.chroma_layer2 = self._make_layer_branch(BasicBlock, 128, 2, stride=2, dropout_rate=config.DROPOUT_RATE_RESBLOCK)
        self.chroma_layer3 = self._make_layer_branch(BasicBlock, 256, 2, stride=2, dropout_rate=config.DROPOUT_RATE_RESBLOCK)

        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fusion Layer
        self.fusion_fc1 = nn.Linear(256 * 3, 512) # 3 branches * 256 features each
        self.fusion_bn1 = nn.BatchNorm1d(512)
        self.fusion_dropout = nn.Dropout(config.DROPOUT_RATE_FUSION)
        self.fusion_fc2 = nn.Linear(512, config.NUM_CLASSES)

    def _make_layer_branch(self, block, out_channels, num_blocks, stride, dropout_rate):
        """Helper to create layers, managing _temp_in_channels for isolated branch construction."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self._temp_in_channels, out_channels, s, dropout_rate))
            self._temp_in_channels = out_channels * block.expansion # Update for next block in this branch
        return nn.Sequential(*layers)

    def forward(self, mfcc, mel_spec, chroma):
        # MFCC branch
        mfcc_out = F.relu(self.bn1_mfcc(self.conv1_mfcc(mfcc)))
        mfcc_out = self.mfcc_layer1(mfcc_out)
        mfcc_out = self.mfcc_layer2(mfcc_out)
        mfcc_out = self.mfcc_layer3(mfcc_out)
        mfcc_out = self.avg_pool(mfcc_out)
        mfcc_out = mfcc_out.view(mfcc_out.size(0), -1) # Flatten

        # Mel-spectrogram branch
        mel_out = F.relu(self.bn1_mel(self.conv1_mel(mel_spec)))
        mel_out = self.mel_layer1(mel_out)
        mel_out = self.mel_layer2(mel_out)
        mel_out = self.mel_layer3(mel_out)
        mel_out = self.avg_pool(mel_out)
        mel_out = mel_out.view(mel_out.size(0), -1) # Flatten

        # Chroma branch
        chroma_out = F.relu(self.bn1_chroma(self.conv1_chroma(chroma)))
        chroma_out = self.chroma_layer1(chroma_out)
        chroma_out = self.chroma_layer2(chroma_out)
        chroma_out = self.chroma_layer3(chroma_out)
        chroma_out = self.avg_pool(chroma_out)
        chroma_out = chroma_out.view(chroma_out.size(0), -1) # Flatten

        # Fusion
        fused_features = torch.cat((mfcc_out, mel_out, chroma_out), dim=1)
        fused_features = F.relu(self.fusion_bn1(self.fusion_fc1(fused_features)))
        fused_features = self.fusion_dropout(fused_features)
        output = self.fusion_fc2(fused_features)

        return output


# --- 3. PyTorch 模型导出为 ONNX 格式 ---

if __name__ == '__main__':
    print("--- PyTorch 模型到 ONNX 导出流程开始 ---")

    # 定义路径 (与您的本地环境和模型文件位置匹配)
    MODEL_DIR = 'D:/baby_kusen/' # 包含 .pth 文件的目录
    PTH_FILENAME = 'best_cry_classifier_v13_fold_3.pth' # 您的 PyTorch 模型文件名
    ONNX_OUTPUT_PATH = os.path.join(MODEL_DIR, 'best_cry_classifier_v13_fold_3.onnx')

    # 确保输出目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. 初始化模型并加载权重
    print(f"加载 PyTorch 模型权重: {os.path.join(MODEL_DIR, PTH_FILENAME)}")
    try:
        model = CryCNN_V13(Config())
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, PTH_FILENAME), map_location=torch.device('cpu')))
        model.eval() # 将模型设置为评估模式 (重要：关闭 dropout 和 BatchNorm 的训练行为)
        print("PyTorch 模型加载成功。")
    except Exception as e:
        print(f"加载 PyTorch 模型失败: {e}")
        print("请检查模型文件是否存在于指定路径，并确认文件未损坏。详细错误: {e}")
        exit()

    # 2. 准备假输入数据（用于 ONNX 导出）
    # 模型的 forward 方法期望三个输入：mfcc, mel_spec, chroma
    # 它们的维度分别是：
    # mfcc: (Batch, N_MFCC * 3, N_FRAMES) 即 (Batch, 120, 259)
    # mel_spec: (Batch, N_MELS, N_FRAMES) 即 (Batch, 128, 259)
    # chroma: (Batch, 12, N_FRAMES) 即 (Batch, 12, 259)
    dummy_mfcc = torch.randn(1, Config.N_MFCC * 3, Config.N_FRAMES, dtype=torch.float32)
    dummy_mel_spec = torch.randn(1, Config.N_MELS, Config.N_FRAMES, dtype=torch.float32)
    dummy_chroma = torch.randn(1, 12, Config.N_FRAMES, dtype=torch.float32)

    # 3. 导出 PyTorch 模型到 ONNX 格式
    print("\n将 PyTorch 模型导出为 ONNX 格式...")
    try:
        input_names = ["mfcc_input", "mel_spec_input", "chroma_input"]
        output_names = ["output"] # ONNX 模型可能只有一个输出

        torch.onnx.export(
            model,
            (dummy_mfcc, dummy_mel_spec, dummy_chroma), # 注意：这里是元组形式传递多个输入
            ONNX_OUTPUT_PATH,
            input_names=input_names,
            output_names=output_names,
            opset_version=11, # 推荐使用 opset_version=11，兼容性较好
            do_constant_folding=True, # 启用常量折叠优化
            dynamic_axes={'mfcc_input': {0: 'batch_size'}, # 定义动态批处理大小
                          'mel_spec_input': {0: 'batch_size'},
                          'chroma_input': {0: 'batch_size'},
                          'output': {0: 'batch_size'}}
        )
        print(f"ONNX 模型已成功保存到: {ONNX_OUTPUT_PATH}")
    except Exception as e:
        print(f"导出 ONNX 模型失败: {e}")
        print("请检查模型架构是否与 PyTorch 兼容 ONNX 导出。详细错误: {e}")
        exit()

    print("\n--- PyTorch 模型到 ONNX 导出流程完成！ ---")

(二) convert_stage1代码

# --- 0. 导入必要的库 ---
import os
import numpy as np
import tensorflow as tf
import onnx # ONNX 核心库
from onnx_tf.backend import prepare # 用于 ONNX 到 TensorFlow 的转换
import warnings

warnings.filterwarnings('ignore')

# --- 1. 配置参数段落 (与训练时的 Config 类保持一致，仅需模型相关部分) ---
class Config:
    SR = 22050
    N_MELS = 128
    N_MFCC = 40
    N_FRAMES = 259
    NUM_CLASSES = 4 # 确保这个数量是正确的

    # Dummy parameters for model definition consistency
    DROPOUT_RATE_RESBLOCK = 0.15
    DROPOUT_RATE_FUSION = 0.3
    LABEL_SMOOTHING = 0.1
    SPEC_AUGMENT_TIME_MASK_PARAM = 20
    SPEC_AUGMENT_FREQ_MASK_PARAM = 10
    MIXUP_ALPHA = 0.3


# --- 2. ONNX 到 TensorFlow SavedModel 转换流程 ---

if __name__ == '__main__':
    print("--- ONNX 到 TensorFlow SavedModel 转换流程开始 ---")

    # 定义路径 (与您的本地环境和 ONNX 文件位置匹配)
    MODEL_DIR = 'D:/baby_kusen/' # 包含 ONNX 文件的目录
    ONNX_FILENAME = 'best_cry_classifier_v13_fold_3.onnx' # 刚才生成的 ONNX 文件名
    TF_SAVED_MODEL_DIR = os.path.join(MODEL_DIR, 'best_cry_classifier_v13_fold_3_tf_model')

    # 确保输出目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(TF_SAVED_MODEL_DIR, exist_ok=True)

    # 1. 加载 ONNX 模型
    ONNX_MODEL_PATH = os.path.join(MODEL_DIR, ONNX_FILENAME)
    print(f"加载 ONNX 模型: {ONNX_MODEL_PATH}")
    try:
        onnx_model = onnx.load(ONNX_MODEL_PATH)
        print("ONNX 模型加载成功。")
    except Exception as e:
        print(f"加载 ONNX 模型失败: {e}")
        print("请检查 ONNX 文件是否存在于指定路径，并确认文件未损坏。详细错误: {e}")
        exit()

    # 2. 将 ONNX 模型转换为 TensorFlow SavedModel 格式
    print("\n将 ONNX 模型转换为 TensorFlow SavedModel 格式...")
    try:
        tf_rep = prepare(onnx_model)
        print(f"DEBUG: tf_rep 对象的类型: {type(tf_rep)}")
        print(f"DEBUG: tf_rep 对象的属性列表: {dir(tf_rep)}")

        # 尝试使用 tf_rep.tf_module 属性（这是 onnx-tf 推荐的方式）
        if hasattr(tf_rep, 'tf_module') and tf_rep.tf_module is not None:
            print("DEBUG: tf_rep 具有 'tf_module' 属性。")
            if isinstance(tf_rep.tf_module, tf.Module):
                print("尝试保存 tf_rep.tf_module (类型为 tf.Module) 并添加签名...")

                # 定义输入 TensorSpec，以匹配 ONNX 模型的输入名称、形状和数据类型
                # 批量大小 (Batch) 使用 None 表示可变大小
                # 将 TensorSpec 放入字典中，键是 ONNX 模型的输入名称
                input_tensor_specs_dict = {
                    'mfcc_input': tf.TensorSpec(shape=(None, Config.N_MFCC * 3, Config.N_FRAMES), dtype=tf.float32, name='mfcc_input'),
                    'mel_spec_input': tf.TensorSpec(shape=(None, Config.N_MELS, Config.N_FRAMES), dtype=tf.float32, name='mel_spec_input'),
                    'chroma_input': tf.TensorSpec(shape=(None, 12, Config.N_FRAMES), dtype=tf.float32, name='chroma_input')
                }

                try:
                    # 从 tf_module 的 __call__ 方法获取一个“具体函数”（ConcreteFunction）
                    # 传入字典形式的关键字参数
                    concrete_func = tf_rep.tf_module.__call__.get_concrete_function(**input_tensor_specs_dict)
                    print(f"DEBUG: 成功从 tf_rep.tf_module 获取 concrete_func。")

                    # 使用获取到的具体函数创建 SavedModel 的 'serving_default' 签名
                    signatures = {'serving_default': concrete_func}
                    tf.saved_model.save(tf_rep.tf_module, TF_SAVED_MODEL_DIR, signatures=signatures)
                    print(f"TensorFlow SavedModel (通过 tf_rep.tf_module 和显式签名) 已成功保存到: {TF_SAVED_MODEL_DIR}")

                except Exception as func_e:
                    print(f"ERROR: 无法从 tf_rep.tf_module 获取 concrete_func 或保存带有签名的模型: {func_e}")
                    print("请检查 Config 中的 N_MFCC, N_MELS, N_FRAMES 是否与 ONNX 模型输入匹配，以及 onnx-tf 是否能正确处理模型结构。")
                    # 如果此步骤失败，则尝试不带签名保存（但已知 TFLite 转换将失败）
                    tf.saved_model.save(tf_rep.tf_module, TF_SAVED_MODEL_DIR)
                    print(f"WARNING: SavedModel 未能包含显式签名。TFLite 转换很可能将失败。")

            else:
                raise TypeError(f"tf_rep.tf_module 存在但不是 tf.Module 类型 ({type(tf_rep.tf_module)})，无法直接保存。")
        elif hasattr(tf_rep, 'graph') and tf_rep.graph is not None:
            print("DEBUG: tf_rep 具有 'graph' 属性。")
            raise NotImplementedError("直接从 tf_rep.graph 转换为 SavedModel 较为复杂，目前不推荐。")
        elif hasattr(tf_rep, 'model') and tf_rep.model is not None:
            print("DEBUG: tf_rep 具有 'model' 属性。")
            if isinstance(tf_rep.model, tf.Module) or isinstance(tf_rep.model, tf.function):
                print("尝试保存 tf_rep.model (类型为 tf.Module 或 tf.function)...")
                # 对于 tf_rep.model 路径，如果也需要显式签名，此处逻辑与 tf_module 类似
                # 但目前 tf_rep.tf_module 是主要工作路径
                tf.saved_model.save(tf_rep.model, TF_SAVED_MODEL_DIR)
                print(f"TensorFlow SavedModel (通过 tf_rep.model) 已保存到: {TF_SAVED_MODEL_DIR}")
            else:
                raise TypeError(f"tf_rep.model 存在但不是 tf.Module 或 tf.function 类型 ({type(tf_rep.model)})，无法直接保存。")
        else:
            raise ValueError("tf_rep 对象中没有可用的 'tf_module', 'graph', 或 'model' 属性进行保存。")

    except Exception as e:
        print(f"转换 ONNX 到 TensorFlow SavedModel 失败: {e}")
        print("这通常是 onnx-tf 库版本问题或 ONNX 模型中存在 TensorFlow 不支持的操作。详细错误: {e}")
        print("建议尝试调整 tensorflow 和 tensorflow-probability 的版本以满足 onnx-tf 的兼容性要求。")
        exit()

    print("\n--- ONNX 到 TensorFlow SavedModel 转换流程完成！ ---")

(三) convert_stage2代码

# --- 0. 导入必要的库 ---
import os
import tensorflow as tf
import warnings

warnings.filterwarnings('ignore')

# --- 1. 配置参数段落 (与训练时的 Config 类保持一致，仅需模型相关部分) ---
class Config:
    SR = 22050
    N_MELS = 128
    N_MFCC = 40
    N_FRAMES = 259
    NUM_CLASSES = 4 # 确保这个数量是正确的

    # Dummy parameters for model definition consistency
    DROPOUT_RATE_RESBLOCK = 0.15
    DROPOUT_RATE_FUSION = 0.3
    LABEL_SMOOTHING = 0.1
    SPEC_AUGMENT_TIME_MASK_PARAM = 20
    SPEC_AUGMENT_FREQ_MASK_PARAM = 10
    MIXUP_ALPHA = 0.3


# --- 2. TensorFlow SavedModel 转换为 TensorFlow Lite (TFLite) 格式 ---

if __name__ == '__main__':
    print("--- TensorFlow SavedModel 到 TFLite 转换流程开始 ---")

    # 定义路径 (与您的本地环境和 SavedModel 文件夹位置匹配)
    MODEL_DIR = 'D:/baby_kusen/' # 包含 SavedModel 文件夹的目录
    TF_SAVED_MODEL_DIR = os.path.join(MODEL_DIR, 'best_cry_classifier_v13_fold_3_tf_model') # 第一阶段生成的 SavedModel 文件夹
    TFLITE_OUTPUT_PATH = os.path.join(MODEL_DIR, 'best_cry_classifier_v13_fold_3.tflite') # 最终 TFLite 模型的输出路径

    # 确保输出目录存在
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. 将 TensorFlow SavedModel 转换为 TensorFlow Lite (TFLite) 格式
    print(f"\n将 TensorFlow SavedModel ({TF_SAVED_MODEL_DIR}) 转换为 TensorFlow Lite (TFLite) 格式...")
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(TF_SAVED_MODEL_DIR)
        # 启用默认优化（例如，量化为 float16），这将减少模型大小并加速推理
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        tflite_model = converter.convert()
    
        with open(TFLITE_OUTPUT_PATH, 'wb') as f:
            f.write(tflite_model)
    
        print(f"TensorFlow Lite 模型已成功保存到: {TFLITE_OUTPUT_PATH}")
    except Exception as e:
        print(f"转换 TensorFlow SavedModel 到 TFLite 失败: {e}")
        print("可能原因：TensorFlow Lite 转换器不支持 SavedModel 中的某些操作，或 SavedModel 缺少签名。详细错误: {e}")
        exit()
print("\n--- TensorFlow SavedModel 到 TFLite 转换流程完成！ ---")



第三阶段工作
【即将展开……】










