# MODEL_UPVCM.md — kiến trúc & huấn luyện UP-VCM (mô hình MỚI của v7)

> **UP-VCM không phải là Zhao additive editor của lineage v1–v6.** Nó được
> thiết kế từ hướng khảo sát tài liệu đầu tiên của v7 (pre-processing phổ quát
> trước codec đóng băng) và từ chính bản đồ falsification của lineage: trục
> *spatial targeting* chưa từng đo + cơ chế *temporal* là mảnh ghép video mà
> toàn bộ họ image-preprocessor (Zhao, Yang, Lu) không có.

## 0. Ký hiệu

| Ký hiệu | Shape | Ý nghĩa |
|---|---|---|
| `x` | `[B,3,T,H,W]` ∈ [0,1] | clip nguồn |
| `cond` | `[B,1]` | QP chuẩn hoá ∈ [0,1] (1 = nén nặng) |
| `W` | `[B,1,T,H,W]` ∈ [0,1] | bản đồ importance tự dự đoán (S head) |
| `mask` | `[B,1,T,H,W]` | **chỉ lúc train**: distill target (teacher saliency + DINOv2) |
| `x_pre` | `[B,3,T,H,W]` | output — pixel đưa vào codec |

## 1. `UPVCMPreprocessor` (`src/models/upvcm.py`, ~44k params)

```
x ──► S: importance head (6ch: RGB + |Δt|) ──► W = σ(CNN(x, |x−xₚᵣₑᵥ|))
│                                                [tự chủ — KHÔNG cần analyzer khi deploy]
├─ M1 background decimation:
│     dec = blurY(σ=0.8) + blurCbCr(σ=2.0) qua YCbCr   (mô phỏng thiệt hại yuv420)
│     x₁ = x + g_dec·(1−W)·(dec − x)          g_dec zero-init
├─ M2 ROI structure editor (UNet 2 tầng, FiLM(cond) zero-init ở bottleneck):
│     x₂ = x₁ + g_edit·W·UNet(x₁, cond)       g_edit zero-init; out-conv zero-init
└─ M3 temporal background stabilisation (causal, từng frame):
      mot = |x₂ᵗ − ref| ; static = 1−clamp(mot/τ)
      gate = g_stab·(1−W)·static ; x₃ᵗ = gate·ref + (1−gate)·x₂ᵗ
      ref = x₃ᵗ.detach()                        (state causal, không BPTT)
      → nền tĩnh ⇒ residual liên khung ≈ 0 ⇒ motion compensation của codec
        tự tiết kiệm bit — CƠ CHẾ VIDEO-NATIVE, module MÀ MÔ HÌNH CŨ KHÔNG CÓ
```

**Identity tại init:** cả 3 gate `g_*` zero-init ⇒ `x₃ = x` chính xác; mọi
module chỉ "bật" khi gradient của loss đòi hỏi (kỷ luật identity-start của v6).

**Deploy:** chỉ cần `x` và `cond` (QP). Không analyzer, không teacher, không
DINOv2 — khác căn bản với D1-gate của v6 (lấy saliency TỪ analyzer eval).

## 2. Distill S head (term `rho`, `src/losses.py`)

```
W_target = (1−α)·teacher_saliency + α·DINOv2_energy     α = model.dino_weight = 0.5
L_W = MSE(S(x), W_target)         (target detached; gradient chỉ vào S)
```

- `teacher_saliency` = |∂L_task/∂x| của panel [r3d_18, mc3_18] sampled từng
  step (multi-teacher stochastic regularization, như v6).
- `DINOv2_energy` = L2-norm của patch tokens `dinov2_vits14` (≤4 frame/clip,
  no_grad, min-max về [0,1], upsample thời gian) — anchor **độc lập với
  analyzer**, chống teacher-overfit (bài học omega=1.0 của v6).
- DINO load thất bại (mạng/hub) ⇒ fallback teacher-only, run vẫn hợp lệ, log 1 dòng.

## 3. Loss tổng (`configs/upvcm_ar.yaml`)

```
L = 1.0·L_task + 3.0·L_D + 0.001·bpp + 0.05·L_temp + 10.0·L_dct(kappa) + 1.0·L_W(rho)
```

- `omega=0` (feature distill tắt — teacher-overfit), `delta/gamma/gamma_res=0`
  (M1/M3 làm công việc của chúng một cách có cấu trúc, không cần penalty toàn
  cục), `kappa=10` (bit-restrainer được đo là tốt nhất của lineage),
  `mu=3` (nhẹ hơn 10 của additive — để có chỗ cho M1 chỉnh nền).

## 4. Huấn luyện & đánh giá

- Proxy codec yuv420 + anneal (C1/A3), in-grid QP [30..50] (C2), 16 epoch,
  batch 8, 128px — **nguyên vẹn protocol lineage** để so sánh cùng instrument.
- Eval: held-out `r2plus1d_18`, x264+x265 thật preset medium, QP {30..50},
  per-sequence → merge shard → BD-Rate + bootstrap CI + gap rule.
- Gates trước eval (`ops/gates_upvcm.py`): G1 non-identity (gate mở), G2
  no-blow-up (RMS < 0.14), G3 W healthy (mean ∈ [0.05,0.95], std > 0.02),
  G4 deploy purity (mask/DINO không ảnh hưởng output eval).

## 5. Vì sao đây là mô hình khác (không phải tuning của cũ)

| | Zhao additive (v1–v6) | UP-VCM (v7) |
|---|---|---|
| Cơ chế | cộng residual 2-branch, thuần 2D | 3 module có điều kiện: trừ nền (M1) + cộng ROI (M2) + ổn định thời gian (M3) |
| Importance | không có (hoặc gate từ analyzer eval — cần analyzer) | S head distill (DINOv2 + teacher), **tự chủ khi deploy** |
| Temporal | chỉ qua motion cue vào SFT/gate | M3 thay nền tĩnh bằng khung trước ⇒ residual ≈ 0 |
| QP-conditioning | FiLM (round b, chưa chạy) | FiLM trong editor M2 |
| Tham số | 9.8k | ~44k |
