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

def attach_calibration_hooks_to_whisper_encoder(encoder):
    encoder.is_calibrating = False

    def make_hook(module_name):
        def hook(module, inputs, output):
            if not getattr(encoder, "is_calibrating", False):
                return

            out = output
            if isinstance(out, tuple):
                out = out[0]
            out = out.detach()
            B, T, D = out.shape

            # 1. 移到 if 之外，并确保 y 总是 float64
            y = out.reshape(-1, D).double().cpu() # double is float64

            # 2. 初始化统计量（只在第一次）
            if not hasattr(module, "calib_count"):
                module.calib_count = 0
                # 初始化为 float64
                module.calib_sum_y = torch.zeros(D, dtype=torch.float64)
                module.calib_sum_yyT = torch.zeros(D, D, dtype=torch.float64)

            # 3. 累积 (现在 y 保证是 float64)
            module.calib_count += y.shape[0]
            module.calib_sum_y += y.sum(dim=0)
            # 这里的 y.T @ y 将在双精度下计算
            module.calib_sum_yyT += y.T @ y

        return hook

    for layer in encoder.layers:
        # ... (和原来一样)
        attn = layer.self_attn
        for proj_name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            mod = getattr(attn, proj_name)
            mod.register_forward_hook(make_hook(f"self_attn.{proj_name}"))

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
            if not hasattr(layer, "calib_count") or layer.calib_count == 0:
                continue

            count = layer.calib_count
            sum_y = layer.calib_sum_y
            sum_yyT = layer.calib_sum_yyT

            # 1. 在 float64 下计算协方差以保证数值稳定性
            mean = (sum_y / count).to(torch.float64)
            E_yyT = (sum_yyT / count).to(torch.float64)
            cov = E_yyT - torch.outer(mean, mean)
            cov = (cov + cov.T) * 0.5

            if not torch.isfinite(cov).all():
                print(f"[HF-cov] layer {i_layer:02d} | {name:8s} | non-finite cov, skip")
                continue

            # 2. 在 float64 下进行特征分解
            try:
                evals, evecs = torch.linalg.eigh(cov)
            except RuntimeError as e:
                print(f"[HF-cov] layer {i_layer:02d} | {name:8s} | eigh failed: {e}, skip")
                continue

            evals = evals.clamp_min(0)
            total_energy = evals.sum()
            if total_energy <= 0:
                print(f"[HF-cov] layer {i_layer:02d} | {name:8s} | zero energy, skip")
                continue

            evals_sorted, indices = torch.sort(evals, descending=True)
            cumsum = torch.cumsum(evals_sorted, dim=0)
            theta = theta_fn(i_layer, is_attn)

            # 查找k值
            found_k = (cumsum / total_energy >= theta).nonzero(as_tuple=True)[0]
            if len(found_k) == 0:
                print(f"[HF-cov] layer {i_layer:02d} | {name:8s} | theta={theta:.6f} too high, failed to find k, skip")
                continue
            k = int(found_k[0].item()) + 1

            # 移除效率约束
            # D_in  = layer.in_features
            # D_out = layer.out_features
            # k_eff_max = int(D_in * D_out / (D_in + D_out))
            # k = min(k, k_eff_max)

            k = ( (k + 15) // 16 ) * 16

            if k <= 0 or k > len(evals_sorted):
                print(f"skip: invalid k {k}")
                continue

            print(f"[HF-cov] layer {i_layer:02d} | {name:8s} | theta={theta:.6f} | k={k}/{len(evals_sorted)}")
            
            proj_indices = indices[:k]
            V_k = evecs[:, proj_indices] # V_k 依然是 float64

            # ====== 构造低秩分解 ======
            device = layer.weight.device
            dtype  = layer.weight.dtype

            # 3. ★★★★★ 新增的修复步骤 ★★★★★
            # 将用于构造的张量统一转换到 float32
            mean_32 = mean.to(torch.float32)
            V_k_cpu = V_k.to(dtype=torch.float32, device="cpu")
            # ★★★★★ 修复结束 ★★★★★

            W = layer.weight.T.detach().to(dtype=torch.float32, device="cpu")
            w1 = W @ V_k_cpu
            w2 = V_k_cpu.T

            if layer.bias is None:
                # 现在所有张量都是 float32
                bias = mean_32 - mean_32 @ V_k_cpu @ V_k_cpu.T
            else:
                bias0 = layer.bias.detach().to(dtype=torch.float32, device="cpu")
                # 现在所有张量都是 float32
                bias = mean_32 + (bias0 - mean_32) @ V_k_cpu @ V_k_cpu.T

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