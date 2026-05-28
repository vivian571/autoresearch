"""
Autoresearch pretraining script. Single-GPU, single-file.
Cherry-picked and simplified from nanochat.
Usage: uv run train.py
"""

import os  # 操作系统接口，用于设置环境变量
# 启用 PyTorch 可扩展内存分配
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# 禁用 Hugging Face Hub 进度条输出
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import gc  # 垃圾回收，用于手动管理内存
import math  # 数学函数库
import time  # 计时函数
from dataclasses import dataclass, asdict  # 数据类装饰器和转字典函数

import torch  # PyTorch 主库
import torch.nn as nn  # 神经网络模块
import torch.nn.functional as F  # 函数式 API

# 导入 Flash Attention 内核加载器
from kernels import get_kernel
# 获取 GPU 的计算能力（如 (9,0) 表示 Hopper GPU）
cap = torch.cuda.get_device_capability()
# 根据 GPU 类型选择 Flash Attention 3 后端（Hopper 用 varunneal，其他用 kernels-community）
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
# ---------------------------------------------------------------------------
# GPT Model  （GPT 模型定义部分）
# ---------------------------------------------------------------------------

# 使用 dataclass 定义 GPT 模型配置参数容器
@dataclass
class GPTConfig:
    # 序列长度（context 长度）
    sequence_len: int = 2048
    # 词表大小
    vocab_size: int = 32768
    # Transformer 层数
    n_layer: int = 12
    # 注意力头数
    n_head: int = 6
    # Key-Value 头数（用于多查询注意力）
    n_kv_head: int = 6
    # 嵌入维度
    n_embd: int = 768
    # 窗口模式（S=短窗口，L=长窗口）
    window_pattern: str = "SSSL"


# 使用 RMSNorm 对张量进行规范化
def norm(x):
    # F.rms_norm：沿最后一个维度的均方根规范化
    return F.rms_norm(x, (x.size(-1),))


# 判断某层是否启用 Value Embedding（采用交替模式，最后一层必须启用）
def has_ve(layer_idx, n_layer):
    """Returns True if layer should have Value Embedding (alternating, last always included)."""
    # 根据层索引和总层数的奇偶性判断
    return layer_idx % 2 == (n_layer - 1) % 2


# 对 query 和 key 应用旋转位置编码（Rotary Position Embedding）
def apply_rotary_emb(x, cos, sin):
    # x 的形状应该是 (B, T, n_head, head_dim)
    assert x.ndim == 4
    # 将最后一个维度分成两部分
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    # 应用旋转矩阵变换：y1 = x1*cos + x2*sin, y2 = -x1*sin + x2*cos
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    # 拼接两部分得到变换后的结果
    return torch.cat([y1, y2], 3)


# Causal Self-Attention 层（多查询注意力变体）
class CausalSelfAttention(nn.Module):
    # 初始化注意力层
    def __init__(self, config, layer_idx):
        super().__init__()
        # 从配置中获取注意力参数
        self.n_head = config.n_head  # 查询头数
        self.n_kv_head = config.n_kv_head  # Key-Value 头数
        self.n_embd = config.n_embd  # 嵌入维度
        # 计算每个头的维度
        self.head_dim = self.n_embd // self.n_head
        # 验证维度整除性
        assert self.n_embd % self.n_head == 0
        # 验证多查询注意力的约束（KV 头数不超过查询头数）
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        # Query 投影层：将嵌入投影到查询空间
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        # Key 投影层：将嵌入投影到键空间（头数可能更少）
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        # Value 投影层：将嵌入投影到值空间
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        # 输出投影层：将注意力输出投影回嵌入维度
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        # Value Embedding 门控层输入通道数
        self.ve_gate_channels = 32
        # 如果该层启用 Value Embedding，创建门控层；否则设为 None
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    # 前向传播
    def forward(self, x, ve, cos_sin, window_size):
        # x: (B, T, C) - 批次、序列长度、嵌入维度
        # ve: Value Embedding 张量或 None
        # cos_sin: (cos, sin) 旋转位置编码
        # window_size: 注意力窗口大小
        B, T, C = x.size()
        # 计算 Query：(B, T, n_head, head_dim)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        # 计算 Key：(B, T, n_kv_head, head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        # 计算 Value：(B, T, n_kv_head, head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value 残差连接（ResFormer）：用门控机制混合 Value Embedding
        if ve is not None:
            # ve: (B, T, n_kv_head, head_dim)
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            # 从输入的前 32 个通道计算门控值，sigmoid 输出在 [0, 1]，乘以 2 得 [0, 2]
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            # 将 Value Embedding 加到 Value 上，由门控参数控制混合比例
            v = v + gate.unsqueeze(-1) * ve

        # 解包旋转位置编码
        cos, sin = cos_sin
        # 对 Query 和 Key 应用旋转位置编码
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        # 规范化 Query 和 Key
        q, k = norm(q), norm(k)

        # 调用 FlashAttention 进行高效的 Causal 注意力计算
        # window_size 用于限制注意力计算的窗口大小
        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        # 将注意力输出形状从 (B, T, n_head, head_dim) 变为 (B, T, C)
        y = y.contiguous().view(B, T, -1)
        # 通过输出投影层
        y = self.c_proj(y)
        # 返回注意力输出
        return y


# MLP（前馈网络）层
class MLP(nn.Module):
    # 初始化 MLP
    def __init__(self, config):
        super().__init__()
        # 第一层：将嵌入维度扩展到 4 倍
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        # 第二层：将 4 倍嵌入维度投影回嵌入维度
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    # 前向传播
    def forward(self, x):
        # x: (B, T, C) - 输入
        # 第一层线性变换
        x = self.c_fc(x)
        # 激活函数：ReLU(x)^2（平方激活）
        x = F.relu(x).square()
        # 第二层投影回原维度
        x = self.c_proj(x)
        # 返回 MLP 输出
        return x


# Transformer Block（包含注意力和 MLP 的完整块）
class Block(nn.Module):
    # 初始化 Block
    def __init__(self, config, layer_idx):
        super().__init__()
        # 创建 Causal Self-Attention 层
        self.attn = CausalSelfAttention(config, layer_idx)
        # 创建 MLP 层
        self.mlp = MLP(config)

    # 前向传播：注意力残差 + MLP 残差
    def forward(self, x, ve, cos_sin, window_size):
        # x: (B, T, C) - 输入
        # 先规范化，送入注意力，再加上残差
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        # 再规范化，送入 MLP，再加上残差
        x = x + self.mlp(norm(x))
        # 返回处理后的张量
        return x


# GPT 模型主类
class GPT(nn.Module):
    # 初始化 GPT 模型
    def __init__(self, config):
        super().__init__()
        # 保存配置
        self.config = config
        # 计算每层的窗口大小（用于局部注意力）
        self.window_sizes = self._compute_window_sizes(config)
        # 创建 transformer 模块字典
        self.transformer = nn.ModuleDict({
            # Token 嵌入层
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            # 多个 Transformer Block
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        # LM Head：从嵌入维度投影到词表大小
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 每层的残差连接缩放参数
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        # 每层的初始状态混合参数
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Value embeddings（仅对启用 VE 的层创建）
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            # 为每层创建对应的词表大小的 Value Embedding
            str(i): nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
        # 预计算旋转位置编码，序列长度扩展 10 倍以增加泛化性
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        # 注册为不持久化的 buffer（不保存到检查点）
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    # 权重初始化
    @torch.no_grad()
    def init_weights(self):
        # 初始化 Token 嵌入层：标准正态分布
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        # 初始化 LM Head 权重：较小的标准差
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        # 计算初始化缩放因子：与模型维度有关
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        # 初始化所有 Transformer Block
        for block in self.transformer.h:
            # Query、Key、Value 投影：均匀分布
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            # 输出投影权重初始化为 0（近似于残差连接的"无操作"）
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            # MLP 第一层权重：均匀分布
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            # MLP 输出投影权重初始化为 0
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        # 初始化残差连接缩放参数为 1.0
        self.resid_lambdas.fill_(1.0)
        # 初始化初始状态混合参数为 0.1
        self.x0_lambdas.fill_(0.1)
        # 初始化 Value Embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        # 初始化 Value Embedding 门控层权重为 0（sigmoid(0)=0.5，乘以 2 得 1.0 = 中性）
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
        # 重新计算旋转位置编码（确保在目标设备上）
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        # 更新 buffer
        self.cos, self.sin = cos, sin
        # 将 Token 嵌入和 Value 嵌入转换为 bfloat16（用于推理效率）
        self.transformer.wte.to(dtype=torch.bfloat16)
        for ve in self.value_embeds.values():
            ve.to(dtype=torch.bfloat16)

    # 预计算旋转位置编码（RoPE）
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # 如果未指定设备，使用 Token 嵌入权重的设备
        if device is None:
            device = self.transformer.wte.weight.device
        # 计算频率范围：[0, 2, 4, ..., head_dim-2]
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        # 计算逆频率：1 / (base^(i / head_dim))
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # 位置序列：[0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # 计算频率矩阵：outer(t, inv_freq)
        freqs = torch.outer(t, inv_freq)
        # 计算 cos 和 sin
        cos, sin = freqs.cos(), freqs.sin()
        # 转换为 bfloat16 以节省内存
        cos, sin = cos.bfloat16(), sin.bfloat16()
        # 重新形状以匹配注意力计算：(1, seq_len, 1, head_dim//2)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        # 返回 cos 和 sin
        return cos, sin

    # 计算每层的注意力窗口大小
    def _compute_window_sizes(self, config):
        # 将模式转为大写（'S' 或 'L'）
        pattern = config.window_pattern.upper()
        # 验证模式只包含 'S' 和 'L'
        assert all(c in "SL" for c in pattern)
        # 长窗口 = 完整序列长度
        long_window = config.sequence_len
        # 短窗口 = 长窗口的一半
        short_window = long_window // 2
        # 字符映射到窗口大小（第二个数字是偏移，这里都是 0）
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
        # 存储每层的窗口大小
        window_sizes = []
        # 遍历所有层，根据模式分配窗口大小
        for layer_idx in range(config.n_layer):
            # 循环使用模式中的字符
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # 强制最后一层使用完整窗口（全局注意力）
        window_sizes[-1] = (long_window, 0)
        # 返回窗口大小列表
        return window_sizes

    # 估算每个 token 的 FLOPs（前向 + 反向）
    def estimate_flops(self):
        """Estimated FLOPs per token (forward + backward)."""
        # 计算总参数数
        nparams = sum(p.numel() for p in self.parameters())
        # 不计入嵌入参数的 FLOPs（嵌入只有查询/键操作）
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        # 计算需要排除的参数数（嵌入 + 标量参数）
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel())
        # 注意力头数
        h = self.config.n_head
        # 每个头的维度
        q = self.config.n_embd // self.config.n_head
        # 序列长度
        t = self.config.sequence_len
        # 初始化注意力 FLOPs
        attn_flops = 0
        # 遍历每层的窗口大小
        for window_size in self.window_sizes:
            # 获取实际窗口大小
            window = window_size[0]
            # 有效序列长度 = min(窗口大小, 序列长度)
            effective_seq = t if window < 0 else min(window, t)
            # 注意力 FLOPs = 12 * n_head * head_dim * effective_seq_len
            attn_flops += 12 * h * q * effective_seq
        # 总 FLOPs = 6 * 矩阵参数 FLOPs + 注意力 FLOPs
        return 6 * (nparams - nparams_exclude) + attn_flops

    # 统计参数数量（按类型分类）
    def num_scaling_params(self):
        # Token 嵌入参数数
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        # Value 嵌入参数数
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        # LM Head 参数数
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        # Transformer Block 中的矩阵参数数
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        # 标量参数（resid_lambdas + x0_lambdas）
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        # 总参数数
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        # 返回字典形式的参数统计
        return {
            'wte': wte, 'value_embeds': value_embeds, 'lm_head': lm_head,
            'transformer_matrices': transformer_matrices, 'scalars': scalars, 'total': total,
        }

    # 设置优化器（混合 AdamW 和 Muon 优化器）
    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02,
                        weight_decay=0.0, adam_betas=(0.8, 0.95), scalar_lr=0.5):
        # 获取模型维度
        model_dim = self.config.n_embd
        # 收集 Transformer Block 的所有参数（矩阵参数，用 Muon 优化）
        matrix_params = list(self.transformer.h.parameters())
        # Value 嵌入参数（用 AdamW 优化）
        value_embeds_params = list(self.value_embeds.parameters())
        # Token 嵌入参数（用 AdamW 优化）
        embedding_params = list(self.transformer.wte.parameters())
        # LM Head 参数（用 AdamW 优化）
        lm_head_params = list(self.lm_head.parameters())
        # 残差连接缩放参数（用 AdamW 优化）
        resid_params = [self.resid_lambdas]
        # 初始状态混合参数（用 AdamW 优化）
        x0_params = [self.x0_lambdas]
        # 验证所有参数都被分类
        assert len(list(self.parameters())) == (len(matrix_params) + len(embedding_params) +
            len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params))
        # 计算学习率缩放因子（与 768 维模型的超参数进行缩放）
        # 原理：学习率应该随着模型维度的平方根成反比缩放
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print(f"Scaling AdamW LRs by 1/sqrt({model_dim}/768) = {dmodel_lr_scale:.6f}")
        # 创建参数分组列表
        param_groups = [
            # LM Head：用 AdamW，低学习率（0.004）
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            # Token 嵌入：用 AdamW，高学习率（0.2）
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            # Value 嵌入：用 AdamW，与 Token 嵌入相同的学习率
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            # 残差连接缩放：用 AdamW，极小学习率（scalar_lr * 0.01）
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=adam_betas, eps=1e-10, weight_decay=0.0),
            # 初始状态混合参数：用 AdamW，不同的 beta 值
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # 根据参数形状将矩阵参数分组，每个形状组使用 Muon 优化器
        for shape in sorted({p.shape for p in matrix_params}):
            # 收集该形状的所有参数
            group_params = [p for p in matrix_params if p.shape == shape]
            # 为该形状组创建 Muon 参数分组
            param_groups.append(dict(
                kind='muon',  # 使用 Muon 优化器
                params=group_params,
                lr=matrix_lr,  # 矩阵学习率（0.02）
                momentum=0.95,  # Nesterov momentum
                ns_steps=5,  # Polar express 迭代步数
                beta2=0.95,  # NorMuon 第二矩估计系数
                weight_decay=weight_decay,  # 权重衰减
            ))
        # 创建 MuonAdamW 优化器（组合了 AdamW 和 Muon）
        optimizer = MuonAdamW(param_groups)
        # 为每个参数分组存储初始学习率（用于学习率调度）
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        # 返回优化器
        return optimizer

    # 前向传播（训练和推理）
    def forward(self, idx, targets=None, reduction='mean'):
        # idx: (B, T) - Token 索引
        # targets: (B, T) - 目标 Token 索引（如果为 None 则仅推理）
        # reduction: 损失聚合方式（'mean' 或 'none'）
        B, T = idx.size()  # 批次大小和序列长度
        # 验证序列长度不超过预计算的旋转编码长度
        assert T <= self.cos.size(1)
        # 截取对应长度的旋转编码
        cos_sin = self.cos[:, :T], self.sin[:, :T]

        # Token 嵌入
        x = self.transformer.wte(idx)  # (B, T, C)
        # 规范化
        x = norm(x)
        # 保存初始状态，用于 x0_lambda 混合
        x0 = x
        # 通过所有 Transformer Block
        for i, block in enumerate(self.transformer.h):
            # 残差连接的可学习混合：lambda_resid * x + lambda_x0 * x0
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            # 如果该层启用 Value Embedding，获取对应的 VE；否则为 None
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            # 通过 Block（注意力 + MLP）
            x = block(x, ve, cos_sin, self.window_sizes[i])
        # 最后规范化
        x = norm(x)

        # Softcap：对 logits 进行缩放和 tanh 压制，防止极端值
        softcap = 15
        # LM Head 投影到词表大小
        logits = self.lm_head(x)  # (B, T, vocab_size)
        # 转换为 float32 以提高数值稳定性
        logits = logits.float()
        # 应用 Softcap：logits = softcap * tanh(logits / softcap)
        logits = softcap * torch.tanh(logits / softcap)

        # 如果提供了目标，计算损失
        if targets is not None:
            # 计算交叉熵损失
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=reduction)
            return loss
        # 如果没有目标，仅返回 logits（推理模式）
        return logits

# ---------------------------------------------------------------------------
# Optimizer (MuonAdamW, single GPU only)  （优化器：混合 Muon 和 AdamW）
# ---------------------------------------------------------------------------

# Polar Express 正交化的系数
# 用于矩阵参数的迭代正交化过程
polar_express_coeffs = [
    # 每个元组对应一次迭代的系数 (a, b, c)
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

# 编译优化的 AdamW 步骤函数
@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):
    # 参数：p（参数）、grad（梯度）、exp_avg（一阶矩）、exp_avg_sq（二阶矩）
    #      step_t（步数）、lr_t（学习率）、beta1_t/beta2_t（指数衰减系数）、eps_t（数值稳定性）、wd_t（权重衰减）
    
    # 权重衰减：p = p * (1 - lr * wd)
    p.mul_(1 - lr_t * wd_t)
    
    # 更新一阶矩估计：exp_avg = beta1 * exp_avg + (1 - beta1) * grad
    exp_avg.lerp_(grad, 1 - beta1_t)
    
    # 更新二阶矩估计：exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    
    # 偏差修正项 1：1 - beta1^t
    bias1 = 1 - beta1_t ** step_t
    # 偏差修正项 2：1 - beta2^t
    bias2 = 1 - beta2_t ** step_t
    
    # 分母：sqrt(exp_avg_sq / bias2) + eps
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    # 步长：lr / bias1
    step_size = lr_t / bias1
    
    # 参数更新：p = p - step_size * exp_avg / denom
    p.add_(exp_avg / denom, alpha=-step_size)

@torch.compile(dynamic=False, fullgraph=True)

# 编译优化的 Muon 步骤函数（用于矩阵参数）
@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    # 参数：stacked_grads（堆叠的梯度）、stacked_params（堆叠的参数）
    #      momentum_buffer/second_momentum_buffer（动量缓冲区）
    #      momentum_t（动量）、lr_t（学习率）、wd_t（权重衰减）、beta2_t（二阶矩系数）
    #      ns_steps（正交化迭代数）、red_dim（方差归一化的约简维度）
    
    # ===== Nesterov 动量部分 =====
    # 转换动量到梯度的 dtype
    momentum = momentum_t.to(stacked_grads.dtype)
    # 更新动量缓冲区：momentum_buffer = momentum * momentum_buffer + (1 - momentum) * grads
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    # 应用 Nesterov 加速：g = grads + momentum * momentum_buffer
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    
    # ===== Polar Express 正交化部分 =====
    # 将梯度转换为 bfloat16
    X = g.bfloat16()
    # 归一化：X = X / (||X|| * 1.02 + eps)，1.02 是过度放松因子
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    
    # 根据矩阵形状选择不同的正交化方向
    if g.size(-2) > g.size(-1):
        # 高矩阵（行 > 列）：使用 A = X^T @ X 的方式
        for a, b, c in polar_express_coeffs[:ns_steps]:
            # 计算 Gram 矩阵
            A = X.mT @ X
            # 计算更新项
            B = b * A + c * (A @ A)
            # 更新 X
            X = a * X + X @ B
    else:
        # 宽矩阵（行 <= 列）：使用 A = X @ X^T 的方式
        for a, b, c in polar_express_coeffs[:ns_steps]:
            # 计算 Gram 矩阵
            A = X @ X.mT
            # 计算更新项
            B = b * A + c * (A @ A)
            # 更新 X
            X = a * X + B @ X
    # 更新梯度为正交化后的版本
    g = X
    
    # ===== NorMuon 方差归一化部分 =====
    # 转换 beta2 到梯度 dtype
    beta2 = beta2_t.to(g.dtype)
    # 计算梯度平方的平均值（沿约简维度）
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    # 约简维度大小
    red_dim_size = g.size(red_dim)
    # 计算二阶矩平方和
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    # 计算范数
    v_norm = v_norm_sq.sqrt()
    # 更新二阶矩缓冲区
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    # 计算步长（二阶矩的倒数根）
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    # 计算缩放后的平方和
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    # 计算新范数
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    # 计算最终缩放因子
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    # 应用缩放到梯度
    g = g * final_scale.to(g.dtype)
    
    # ===== "谨慎"权重衰减与参数更新 =====
    # 转换学习率和权重衰减到梯度 dtype
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    # 创建掩码：只有当梯度与参数同号时才应用权重衰减
    # 这实现了"谨慎"权重衰减，避免在梯度与参数反向时施加额外的衰减
    mask = (g * stacked_params) >= 0
    # 参数更新：p = p - lr * g - lr * wd * p * mask
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)


# 组合优化器：Muon（用于 2D 矩阵参数）+ AdamW（用于其他参数）
class MuonAdamW(torch.optim.Optimizer):
    """Combined optimizer: Muon for 2D matrix params, AdamW for others."""

    # 初始化优化器
    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})
        # 使用 0-D CPU 张量存储超参数，避免 torch.compile 重新编译
        # 这些是标量张量，在 compile 之外修改，不会触发重编译
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")  # AdamW 学习率
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")  # AdamW beta1
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")  # AdamW beta2
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")  # AdamW 数值稳定性
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")  # AdamW 权重衰减
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")  # Muon 动量
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")  # Muon 学习率
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")  # Muon 权重衰减
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")  # Muon 二阶矩系数

    # AdamW 步骤（用于非矩阵参数）
    def _step_adamw(self, group):
        # 遍历参数组中的所有参数
        for p in group['params']:
            # 跳过没有梯度的参数
            if p.grad is None:
                continue
            # 获取梯度
            grad = p.grad
            # 获取或初始化状态字典
            state = self.state[p]
            if not state:
                # 初始化：步数为 0，一阶矩和二阶矩为零
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            # 增加步数
            state['step'] += 1
            # 更新 CPU 张量以传递给编译函数
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            # 调用编译优化的 AdamW step 函数
            adamw_step_fused(p, grad, state['exp_avg'], state['exp_avg_sq'],
                            self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                            self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t)

    # Muon 步骤（用于 2D 矩阵参数）
    def _step_muon(self, group):
        # 获取参数列表
        params = group['params']
        # 如果参数列表为空，直接返回
        if not params:
            return
        # 获取第一个参数（用于确定形状和设备）
        p = params[0]
        # 获取或初始化状态字典
        state = self.state[p]
        # 参数数量
        num_params = len(params)
        # 形状、设备和数据类型
        shape, device, dtype = p.shape, p.device, p.dtype
        # 初始化动量缓冲区（形状: [num_params, *param_shape]）
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        # 初始化二阶矩缓冲区（形状取决于矩阵方向）
        if "second_momentum_buffer" not in state:
            # 对于高矩阵，约简维度是最后一个；对于宽矩阵，约简维度是倒数第二个
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        # 确定约简维度（用于方差归一化）
        red_dim = -1 if shape[-2] >= shape[-1] else -2
        # 堆叠所有参数的梯度
        stacked_grads = torch.stack([p.grad for p in params])
        # 堆叠所有参数
        stacked_params = torch.stack(params)
        # 更新 CPU 张量以传递给编译函数
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        # 根据矩阵宽高比缩放学习率
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])
        # 调用编译优化的 Muon step 函数
        muon_step_fused(stacked_grads, stacked_params,
                        state["momentum_buffer"], state["second_momentum_buffer"],
                        self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t,
                        self._muon_beta2_t, group["ns_steps"], red_dim)
        # 将更新后的参数从堆叠版本复制回原始参数列表
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    # 优化器步骤（无梯度，不计算梯度）
    @torch.no_grad()
    def step(self):
        # 遍历所有参数分组
        for group in self.param_groups:
            # 根据分组类型调用相应的步骤函数
            if group['kind'] == 'adamw':
                # 调用 AdamW 步骤
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                # 调用 Muon 步骤
                self._step_muon(group)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)  （超参数配置）
# ---------------------------------------------------------------------------

# ===== 模型架构相关 =====
ASPECT_RATIO = 64       # 宽高比：model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # 目标注意力头维度
WINDOW_PATTERN = "SSSL" # 滑动窗口模式：L=完整窗口，S=半窗口上下文

# ===== 优化相关 =====
TOTAL_BATCH_SIZE = 2**19  # ~524K tokens per optimizer step（每步优化器处理的总 tokens 数）
EMBEDDING_LR = 0.6      # Token 嵌入的学习率（使用 Adam）
UNEMBEDDING_LR = 0.004  # LM Head 的学习率（使用 Adam）
MATRIX_LR = 0.04        # 矩阵参数的学习率（使用 Muon）
SCALAR_LR = 0.5         # 每层标量参数的学习率（使用 Adam）
WEIGHT_DECAY = 0.2      # Muon 的谨慎权重衰减系数
ADAM_BETAS = (0.8, 0.95)  # Adam 的 beta1 和 beta2
WARMUP_RATIO = 0.0      # 预算中用于 LR warmup 的比例
WARMDOWN_RATIO = 0.5    # 预算中用于 LR warmdown 的比例
FINAL_LR_FRAC = 0.0     # 最终 LR 作为初始 LR 的百分比

# ===== 模型大小 =====
DEPTH = 8               # Transformer 层数
DEVICE_BATCH_SIZE = 128  # 单卡批次大小（如果 OOM 则减小）

# ---------------------------------------------------------------------------
# Setup: tokenizer, model, optimizer, dataloader  （初始化：分词器、模型、优化器、数据加载器）
# ---------------------------------------------------------------------------

# 记录启动时间
t_start = time.time()
# 固定随机种子以保证可重现性
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# 使用 TensorFloat32（TF32）加速矩阵乘法
torch.set_float32_matmul_precision("high")
# 设备为 CUDA（GPU）
device = torch.device("cuda")
# 自动混合精度上下文：使用 bfloat16 进行计算
autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
# H100 GPU 的 bfloat16 峰值 FLOPs
H100_BF16_PEAK_FLOPS = 989.5e12

# 从本地目录加载分词器
tokenizer = Tokenizer.from_directory()
# 获取词表大小
vocab_size = tokenizer.get_vocab_size()
print(f"Vocab size: {vocab_size:,}")

# 根据深度构建模型配置
def build_model_config(depth):
    # 计算基础维度
    base_dim = depth * ASPECT_RATIO
    # 对齐到 HEAD_DIM 的倍数
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    # 根据模型维度和头维度计算头数
    num_heads = model_dim // HEAD_DIM
    # 返回 GPT 配置对象
    return GPTConfig(
        sequence_len=MAX_SEQ_LEN, vocab_size=vocab_size,
        n_layer=depth, n_head=num_heads, n_kv_head=num_heads, n_embd=model_dim,
        window_pattern=WINDOW_PATTERN,
    )

# 构建模型配置
config = build_model_config(DEPTH)
# 打印配置
print(f"Model config: {asdict(config)}")

# 在元设备上创建模型（不分配实际内存）
with torch.device("meta"):
    model = GPT(config)
# 转移模型到实际设备（CUDA）
model.to_empty(device=device)
# 初始化权重
model.init_weights()

# 获取参数统计信息
param_counts = model.num_scaling_params()
print("Parameter counts:")
for key, value in param_counts.items():
    print(f"  {key:24s}: {value:,}")
# 获取总参数数
num_params = param_counts['total']
# 估算每个 token 的 FLOPs
num_flops_per_token = model.estimate_flops()
print(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# 计算梯度累积步数
tokens_per_fwdbwd = DEVICE_BATCH_SIZE * MAX_SEQ_LEN
# 验证总批次大小是否能整除单卡前向/反向的 tokens 数
assert TOTAL_BATCH_SIZE % tokens_per_fwdbwd == 0
# 计算需要累积多少个微批次
grad_accum_steps = TOTAL_BATCH_SIZE // tokens_per_fwdbwd

# 设置优化器
optimizer = model.setup_optimizer(
    unembedding_lr=UNEMBEDDING_LR,
    embedding_lr=EMBEDDING_LR,
    scalar_lr=SCALAR_LR,
    adam_betas=ADAM_BETAS,
    matrix_lr=MATRIX_LR,
    weight_decay=WEIGHT_DECAY,
)

# 编译模型以提高性能
model = torch.compile(model, dynamic=False)

# 创建训练数据加载器
train_loader = make_dataloader(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train")
# 预加载第一个批次
x, y, epoch = next(train_loader)

# 打印训练配置信息
print(f"Time budget: {TIME_BUDGET}s")
print(f"Gradient accumulation steps: {grad_accum_steps}")

# ===== 学习率和超参数调度（基于进度 progress = training_time / TIME_BUDGET）=====

# 学习率倍数调度：warmup + plateau + warmdown
def get_lr_multiplier(progress):
    # progress 从 0 到 1
    if progress < WARMUP_RATIO:
        # Warmup 阶段：从 0 线性增加到 1
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        # Plateau 阶段：保持 1.0（常数 LR）
        return 1.0
    else:
        # Warmdown 阶段：从 1.0 衰减到 FINAL_LR_FRAC
        cooldown = (1.0 - progress) / WARMDOWN_RATIO
        return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC

# Muon 动量调度：从 0.85 逐步增加到 0.95
def get_muon_momentum(step):
    # 在前 300 步内逐渐增加动量
    frac = min(step / 300, 1)
    # 线性插值：momentum = (1-frac)*0.85 + frac*0.95
    return (1 - frac) * 0.85 + frac * 0.95

# 权重衰减调度：从 WEIGHT_DECAY 线性衰减到 0
def get_weight_decay(progress):
    # weight_decay = WEIGHT_DECAY * (1 - progress)
    return WEIGHT_DECAY * (1 - progress)

# ---------------------------------------------------------------------------
# Training loop  （训练循环）
# ---------------------------------------------------------------------------

# 记录训练开始时间
t_start_training = time.time()
# 平滑后的训练损失（用于显示）
smooth_train_loss = 0
# 累积的总训练时间（不包含编译时间）
total_training_time = 0
# 优化器步数
step = 0

# 主训练循环
while True:
    # 同步 GPU，确保时间测量准确
    torch.cuda.synchronize()
    t0 = time.time()
    
    # ===== 梯度累积 =====
    # 在累积 grad_accum_steps 个微批次后进行一次优化器步骤
    for micro_step in range(grad_accum_steps):
        # 使用自动混合精度（bfloat16）
        with autocast_ctx:
            # 前向传播，计算损失
            loss = model(x, y)
        # 分离损失用于记录（不会参与梯度计算）
        train_loss = loss.detach()
        # 缩放损失以进行梯度累积
        loss = loss / grad_accum_steps
        # 反向传播，累积梯度
        loss.backward()
        # 预加载下一个批次（在反向传播时进行）
        x, y, epoch = next(train_loader)

    # ===== 学习率和动量调度 =====
    # 计算当前进度（0 到 1）
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    # 获取学习率倍数
    lrm = get_lr_multiplier(progress)
    # 获取 Muon 动量
    muon_momentum = get_muon_momentum(step)
    # 获取权重衰减
    muon_weight_decay = get_weight_decay(progress)
    
    # 更新所有参数分组的学习率和其他超参数
    for group in optimizer.param_groups:
        # 应用学习率倍数
        group["lr"] = group["initial_lr"] * lrm
        # 如果是 Muon 优化器，更新动量和权重衰减
        if group['kind'] == 'muon':
            group["momentum"] = muon_momentum
            group["weight_decay"] = muon_weight_decay
    
    # 执行优化器步骤
    optimizer.step()
    # 清空梯度（设置为 None 比 zero_() 更高效）
    model.zero_grad(set_to_none=True)

    # 获取当前批次的损失值
    train_loss_f = train_loss.item()

    # ===== 快速失败机制：损失爆炸或 NaN 时中止 =====
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        exit(1)

    # 同步 GPU 以获取准确的时间
    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    # 跳过前 10 步的时间（用于编译和预热）
    if step > 10:
        total_training_time += dt

    # ===== 日志记录 =====
    # 使用指数移动平均（EMA）平滑损失
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    # 去偏差的 EMA 损失（消除初始偏差）
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    # 计算训练进度百分比
    pct_done = 100 * progress
    # 计算吞吐量：tokens/秒
    tok_per_sec = int(TOTAL_BATCH_SIZE / dt)
    # 计算 Model FLOPs Utilization（MFU）：实际 FLOPs / 峰值 FLOPs
    mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE / dt / H100_BF16_PEAK_FLOPS
    # 计算剩余时间
    remaining = max(0, TIME_BUDGET - total_training_time)

    # 打印训练日志（\r 用于覆盖前一行）
    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.1f}% | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # ===== 垃圾回收（GC）管理 =====
    # Python 的自动 GC 会导致约 500ms 的停顿，影响训练速度
    if step == 0:
        # 第一步时进行一次完整的垃圾回收
        gc.collect()
        # 冻结 GC 以防止自动触发
        gc.freeze()
        # 禁用自动 GC
        gc.disable()
    elif (step + 1) % 5000 == 0:
        # 每 5000 步进行一次手动垃圾回收（避免频繁停顿）
        gc.collect()

    # 增加步数
    step += 1

    # ===== 停止条件：时间预算已用完 =====
    # 只在 warmup 后停止，避免统计编译时间
    if step > 10 and total_training_time >= TIME_BUDGET:
        break

# 打印换行符（结束 \r 的覆盖）
print()

# ===== 训练后评估 =====
# 计算训练期间处理的总 tokens 数
total_tokens = step * TOTAL_BATCH_SIZE

# 将模型设置为评估模式（禁用 dropout 等）
model.eval()
# 在验证集上计算 Bits Per Byte（BPB）指标
with autocast_ctx:
    val_bpb = evaluate_bpb(model, tokenizer, DEVICE_BATCH_SIZE)

# ===== 最终汇总 =====
# 记录训练结束时间
t_end = time.time()
# 计算启动时间（从脚本开始到训练开始）
startup_time = t_start_training - t_start
# 计算稳定态 MFU（排除前 10 个 warmup 步）
steady_state_mfu = 100 * num_flops_per_token * TOTAL_BATCH_SIZE * (step - 10) / total_training_time / H100_BF16_PEAK_FLOPS if total_training_time > 0 else 0
# 获取峰值显存使用量
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

# 打印分隔符
print("---")
# 打印最终训练结果
print(f"val_bpb:          {val_bpb:.6f}")  # 验证集 BPB 指标
print(f"training_seconds: {total_training_time:.1f}")  # 实际训练时间
print(f"total_seconds:    {t_end - t_start:.1f}")  # 总用时（包含启动）
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")  # 峰值显存
print(f"mfu_percent:      {steady_state_mfu:.2f}")  # 稳定态模型利用率
print(f"total_tokens_M:   {total_tokens / 1e6:.1f}")  # 总 tokens 数（百万）
print(f"num_steps:        {step}")  # 总优化器步数
print(f"num_params_M:     {num_params / 1e6:.1f}")  # 总参数数（百万）
print(f"depth:            {DEPTH}")  # 模型深度（层数）
