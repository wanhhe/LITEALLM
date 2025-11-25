# finetune_codes/low_rank_whisper_new.py
import torch
import torch.nn as nn
import tqdm

# ========== 低秩 Linear ==========
class LinearLowRank(nn.Module):
    def __init__(self, weight1: torch.Tensor, weight2: torch.Tensor, bias: torch.Tensor):
        """
        weight1: [in_features, k]
        weight2: [k, out_features]
        bias   : [out_features]
        """
        super().__init__()
        # 继承原权重的 dtype / device
        dtype = weight1.dtype
        device = weight1.device

        self.weight1 = nn.Parameter(weight1.to(device=device, dtype=dtype))
        self.weight2 = nn.Parameter(weight2.to(device=device, dtype=dtype))
        self.bias    = nn.Parameter(bias.to(device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]
        y = torch.matmul(x, self.weight1)      # [..., k]
        y = torch.matmul(y, self.weight2)      # [..., out_features]
        y = y + self.bias                      # broadcast 到前面的维度
        return y


# ========== θ 随层数线性递增 ==========
def build_layerwise_theta_fn(rank_threshold: str, num_layers: int):
    """
    构造一个函数 theta_fn(layer_idx, is_attn) -> 当前层要用的 θ

    rank_threshold:
        * "0.99:0.999"  -> attn_min=0.99, mlp_min=0.999
        * "0.995"       -> attn_min=mlp_min=0.995
    我们把 attn/MLP 的 θ 从各自的 min 线性插值到一个不超过 0.999 的 max，
    让前层压缩更激进，后层更保守。
    """
    if ":" in rank_threshold:
        attn_min_str, mlp_min_str = rank_threshold.split(":")
        attn_min = float(attn_min_str)
        mlp_min  = float(mlp_min_str)
    else:
        attn_min = mlp_min = float(rank_threshold)

    ATTEN_MAX_CAP = 0.999
    MLP_MAX_CAP   = 0.999

    # 最多增加 0.009，你可以之后自己调
    attn_max = max(attn_min, min(ATTEN_MAX_CAP, attn_min + 0.009))
    mlp_max  = max(mlp_min,  min(MLP_MAX_CAP,   mlp_min  + 0.009))

    def theta_fn(layer_idx: int, is_attn: bool) -> float:
        if num_layers <= 1:
            return attn_max if is_attn else mlp_max

        alpha = layer_idx / (num_layers - 1)   # 0 → 1

        if is_attn:
            return attn_min + (attn_max - attn_min) * alpha
        else:
            return mlp_min  + (mlp_max  - mlp_min)  * alpha

    return theta_fn


# ========== 在 HF WhisperEncoder 上挂 forward hook ==========

import torch
import torch.nn as nn
import tqdm

# 保留你原来的 LinearLowRank / build_layerwise_theta_fn 不变
# 这里只改 hook + apply_low_rank

def attach_calibration_hooks_to_whisper_encoder(encoder):
    """
    encoder: whisper.speech_encoder (WhisperEncoder)

    这里不再把所有 token 激活攒在 calibration_outputs 里，
    而是对每个 module 累积：
      - calib_count      : 总 token 数
      - calib_sum_y      : 所有 y 的和       [D]
      - calib_sum_yyT    : 所有 y^T y 的和   [D, D]

    这样样本数可以很大，内存只和 D 有关，不和 token 总数有关。
    """
    encoder.is_calibrating = False  # 标志位

    def make_hook(module_name):
        def hook(module, inputs, output):
            if not getattr(encoder, "is_calibrating", False):
                return

            out = output
            if isinstance(out, tuple):
                out = out[0]
            # out: [B, T, D]
            out = out.detach()
            B, T, D = out.shape
            y = out.reshape(-1, D).float().cpu()   # [N_tokens, D]

            # 初始化统计量
            if not hasattr(module, "calib_count"):
                module.calib_count = 0
                module.calib_sum_y = torch.zeros(D, dtype=torch.float32)
                module.calib_sum_yyT = torch.zeros(D, D, dtype=torch.float32)

            # 累积
            module.calib_count += y.shape[0]
            module.calib_sum_y += y.sum(dim=0)              # [D]
            # y.T @ y: [D, D]
            module.calib_sum_yyT += y.T @ y

        return hook

    for layer in encoder.layers:
        # self-attn 的四个 Linear
        attn = layer.self_attn
        for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            mod = getattr(attn, proj_name)
            mod.register_forward_hook(make_hook(f"self_attn.{proj_name}"))

        # MLP 两层
        for mlp_name in ["fc1", "fc2"]:
            mod = getattr(layer, mlp_name)
            mod.register_forward_hook(make_hook(mlp_name))


def apply_low_rank_to_whisper_encoder(
    encoder,
    rank_threshold: str = "0.99:0.999",
):
    """
    使用 attach_calibration_hooks_to_whisper_encoder 收集的
    calib_count / calib_sum_y / calib_sum_yyT 做低秩分解。
    """
    num_layers = len(encoder.layers)
    d_model    = encoder.config.d_model
    theta_fn   = build_layerwise_theta_fn(rank_threshold, num_layers)

    stats = []

    for i_layer, block in enumerate(tqdm.tqdm(encoder.layers, desc="Low-rank per layer")):
        attn = block.self_attn
        fc1  = block.fc1
        fc2  = block.fc2

        components = [
            ("q_proj",   attn.q_proj,   True),
            ("k_proj",   attn.k_proj,   True),
            ("v_proj",   attn.v_proj,   True),
            ("out_proj", attn.out_proj, True),
            ("fc1",      fc1,           False),
            ("fc2",      fc2,           False),
        ]

        for name, layer, is_attn in components:
            # 1) 没有校准统计，直接跳过（这里才是你原来想要的 continue）
            if not hasattr(layer, "calib_count") or layer.calib_count == 0:
                # 你想看的话可以加一行：
                # print(f"[HF-cov] layer {i_layer:02d} | {name:8s} | no calib stats, skip")
                continue

            count = layer.calib_count
            sum_y = layer.calib_sum_y      # [D]
            sum_yyT = layer.calib_sum_yyT  # [D, D]

            # 2) 计算均值和 E[yy^T]
            mean = (sum_y / count).to(torch.float32)          # [D]
            E_yyT = (sum_yyT / count).to(torch.float32)       # [D, D]

            # 协方差 Cov = E[yy^T] - μ μ^T，顺便对称化一下防止数值问题
            cov = E_yyT - torch.outer(mean, mean)
            cov = (cov + cov.T) * 0.5

            # 简单检查一下是否有 NaN / Inf
            if not torch.isfinite(cov).all():
                print(f"[HF-cov] layer {i_layer:02d} | {name:8s} | non-finite cov, skip")
                continue

            # 3) 特征分解
            try:
                evals, evecs = torch.linalg.eigh(cov)   # evals: [D], evecs: [D,D]
            except RuntimeError as e:
                print(f"[HF-cov] layer {i_layer:02d} | {name:8s} | eigh failed: {e}, skip")
                continue

            # 从小到大排序，取后面大的（方差大的主方向）
            evals = evals.clamp_min(0)
            total_energy = evals.sum()
            if total_energy <= 0:
                print(f"[HF-cov] layer {i_layer:02d} | {name:8s} | zero energy, skip")
                continue

            # theta = theta_fn(i_layer, is_attn)

            # # 按能量从大到小累积
            # evals_sorted, indices = torch.sort(evals, descending=True)
            # cumsum = torch.cumsum(evals_sorted, dim=0)

            # # k = -1
            # # for j in range(16, len(evals_sorted) + 1, 16):
            # #     if (cumsum[:j].sum() / total_energy) > theta:
            # #         k = j
            # #         break
            # k = -1
            # for j in range(16, len(evals_sorted) + 1):
            #     if (cumsum[:j].sum() / total_energy) > theta:
            #         k = j
            #         break

            evals_sorted, indices = torch.sort(evals, descending=True)
            cumsum = torch.cumsum(evals_sorted, dim=0)

            theta = theta_fn(i_layer, is_attn)

            # --- 1) 先按能量阈值找最小的 k ---
            k = int((cumsum / total_energy >= theta).nonzero(as_tuple=True)[0][0].item()) + 1
            # k 现在是满足阈值的最小 index + 1

            # --- 2) 再加效率约束 ---
            D_in  = layer.in_features
            D_out = layer.out_features
            k_eff_max = int(D_in * D_out / (D_in + D_out))
            k = min(k, k_eff_max)

            # --- 3) 对齐到 16 的倍数（四舍五入/向上取都可以）---
            k = ( (k + 15) // 16 ) * 16    # 向上取整到 16 的倍数

            # --- 4) 一些 sanity check ---
            if k <= 0 or k > len(evals_sorted):
                print("skip: invalid k", k)
                continue


            print(f"[HF-cov] layer {i_layer:02d} | {name:8s} | theta={theta:.6f} | k={k}/{len(evals_sorted)}")

            # 4) k 不合理就跳过（不会 append stats）
            if is_attn:
                if k < 0 or k > 0.5 * d_model:
                    continue
            else:
                if k < 0 or k > 0.8 * d_model:
                    continue

            # 5) 取前 k 个特征向量组成投影矩阵 V_k
            proj_indices = indices[-k:]  # 注意 indices 是升序 or 降序，按上面逻辑调整
            # 上面 sort(descending=True)，所以大的在前面，这里用前 k 个：
            proj_indices = indices[:k]
            V_k = evecs[:, proj_indices]      # [D, k]

            # ====== 构造低秩分解 ======
            device = layer.weight.device
            dtype  = layer.weight.dtype

            # HF Linear: weight shape = (out_features, in_features)
            W = layer.weight.T.detach().to(dtype=torch.float32, device="cpu")  # [D_in, D_out], D_out == D

            V_k_cpu = V_k.to(dtype=torch.float32, device="cpu")   # [D_out, k]
            w1 = W @ V_k_cpu           # [D_in, k]
            w2 = V_k_cpu.T             # [k, D_out]

            if layer.bias is None:
                bias = mean - mean @ V_k_cpu @ V_k_cpu.T
            else:
                bias0 = layer.bias.detach().to(dtype=torch.float32, device="cpu")
                bias = mean + (bias0 - mean) @ V_k_cpu @ V_k_cpu.T

            # cast 回原 dtype + device
            w1   = w1.to(device=device, dtype=dtype)
            w2   = w2.to(device=device, dtype=dtype)
            bias = bias.to(device=device, dtype=dtype)

            new_layer = LinearLowRank(w1, w2, bias)

            if name == "q_proj":
                block.self_attn.q_proj = new_layer
            elif name == "k_proj":
                block.self_attn.k_proj = new_layer
            elif name == "v_proj":
                block.self_attn.v_proj = new_layer
            elif name == "out_proj":
                block.self_attn.out_proj = new_layer
            elif name == "fc1":
                block.fc1 = new_layer
            elif name == "fc2":
                block.fc2 = new_layer

            stats.append({
                "layer_idx": int(i_layer),
                "name": name,
                "is_attn": bool(is_attn),
                "theta": float(theta),
                "k": int(k),
                "rank_dim": int(len(evals_sorted)),
            })

    return encoder, stats