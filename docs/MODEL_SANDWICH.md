# MODEL_SANDWICH.md — v8: PRE + POST quanh codec đóng băng

## Ý tưởng

Pre-only (v7 và mọi tiền nhiệm) mua accuracy đúng theo tỉ giá của anchor —
nghĩa là bộ tiền xử lý khó thắng chính núm QP. **Sandwich** thêm một nửa
**POST** (restoration filter sau decode, trước analyzer) với chi phí bit
**bằng không**: phần accuracy codec phá đi có thể mua lại phía decoder, cho
phép PRE cắt bit mạnh hơn mức pre-only có thể.

```
            PRE (UP-VCM, v7)              POST (mới)               
 x ─► pre(x) ─► x264/x265 (đóng băng) ─► decode x̂ ─► post(x̂) ─► analyzer
     cắt bit nền/ổn định                khôi phục cấu trúc đã mất
```

Đây là thiết kế mạnh nhất đã biết cho pre/post với codec đóng băng (Google
Sandwiched Compression, arXiv:2402.05887 — 10–15% trên metric con người);
v8 là bản máy (machine-metric) với eval held-out + CI.

## Kiến trúc (`src/models/sandwich.py`)

- **PRE**: nguyên UP-VCM (S + M1 + M2 + M3, ~44k) — checkpoint v7
  strict-load qua `load_pre_state()` (warm-start được test).
- **POST**: restoration UNet 3 tầng (~577k, base 32), FiLM(QP) zero-init ở
  bottleneck, output conv zero-init, gate vô hướng `post_strength` zero-init
  → identity tại khởi tạo. Input = clip đã decode (không dùng W: post phải
  sửa TOÀN bộ frame, và S nhìn artifact codec sẽ nhiễu).
- `forward()` = nửa PRE (engine gọi như cũ); `post_restore()` = nửa POST,
  engine gọi ngay sau codec. `bypass_post` = công tắc tách arm eval.

## Huấn luyện

- Joint end-to-end qua proxy codec yuv420 (gradients đến cả PRE lẫn POST
  trong 1 backward): loss trên `post(codec(pre(x)))`.
- `mu=3` giữ vai trò mục tiêu khôi phục (MSE về pixel nguồn) — với sandwich
  đây là mục tiêu *đúng* của nửa POST.
- Stage-2 STE: codec thật trong vòng lặp → POST học đảo artifact x264/x265
  thật (giá trị thật, gradient qua proxy — công thức đã được đo là lever
  transfer đáng tin nhất: −20.3% vs −14.6%).

## Eval — 3 arm cùng 1 checkpoint (tách rời cơ chế)

| Arm | Đường | Ý nghĩa |
|---|---|---|
| anchor | codec(x) → analyzer | baseline chuẩn |
| prep+codec | pre(x) → codec → analyzer (bypass post) | đóng góp PRE |
| **sandwich+codec** | pre(x) → codec → **post** → analyzer | **claim chính** |

Cùng bpp giữa prep và sandwich (post không tốn bit) → BD-Rate của sandwich
arm đo trực tiếp giá trị POST cộng thêm. Protocol còn lại nguyên vẹn: held-out
`r2plus1d_18`, x264+x265 QP30–50, n=1159, bootstrap CI, gap rule.

## Kỳ vọng đã đăng ký (trước khi chạy)

- POST ≥ 0 accuracy ở mọi QP (nó chỉ có thể sửa, không thể làm hỏng nếu
  train ổn — gate zero-init bảo hiểm) → gap rule an toàn hơn pre-only.
- Sandwich BD tốt hơn prep-only cùng checkpoint (dấu bằng cơ chế, không cần
  magnitude lớn).
- Rủi ro chính: POST khôi phục cho TEACHER (overfit analyzer panel) — kiểm
  bằng held-out; nếu sandwich gain biến mất trên held-out thì claim chỉ là
  "post overfits teacher" — sẽ báo trung thực.
