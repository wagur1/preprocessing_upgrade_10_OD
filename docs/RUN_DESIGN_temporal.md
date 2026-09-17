# RUN_DESIGN_temporal.md — POST có ngữ cảnh thời gian

**Ngày:** 2026-09-16 · **Trạng thái:** code + test + pre-flight xong, chưa chạy GPU
**Config:** `configs/sandwich_temporal.yaml`

---

## 1. Vì sao

POST hiện **xử lý từng frame độc lập**: `post_restore` làm phẳng `(B,T) -> B*T` rồi cho
qua UNet 2-D. Nhưng artifact của codec video có cấu trúc thời gian rất mạnh:

- I-frame và P/B-frame có thống kê lỗi khác nhau (cùng một QP);
- lỗi **lan truyền dọc GOP** — frame sau thừa hưởng tổn thất của frame trước;
- flicker là hiện tượng *định nghĩa bằng thời gian*.

Frame lân cận mang chính nội dung đó nhưng được mã hoá ở **điểm khác của chuỗi dự đoán** —
thông tin mà một bộ lọc per-frame về mặt cấu trúc không thể thấy. Đây là inductive bias
video chuẩn mực và là thứ literature video restoration luôn dùng; trong khi đó mọi
publication của niche này (Zhao, Lu, Google Sandwiched) đều per-frame ở phần POST.

Chi phí: **0 bit** (decoder-side), **+3.520 tham số** (một conv 6→64 ở đầu vào, +0,15 % của
POST 2,29M). Nếu có lợi thì lợi ích là miễn phí về rate.

## 2. Thiết kế

Nhánh **cộng thêm, zero-init** vào trunk đang sống:

```
e0 = in_conv(x_t) + in_conv_t([x_{t-1}, x_{t+1}])
```

- `in_conv_t` zero-init ⇒ tại init module **đúng bằng** POST per-frame ⇒ warm-start từ
  checkpoint kỷ lục là behaviour-preserving, mọi thay đổi quy được cho nhánh thời gian.
- Cửa sổ ±1, biên lặp lại frame đầu/cuối (không cần tương lai xa, không đệ quy).
- **Không thể dead-saddle** như M2/POST-gate: nó cộng vào trunk đã sống, nên gradient khác 0
  ngay khi POST gate mở (đã kiểm bằng test).
- T=1 (ảnh / All-Intra) vẫn chạy: biên tự lặp.

## 3. Đã kiểm chứng cục bộ

| Kiểm tra | Kết quả |
|---|---|
| 5 test mới (`tests/test_post_temporal.py`) | PASS — identity tại init, gradient ≠ 0 ngay, nhánh thay output khi mở, T=1 chạy được, cờ mặc định tắt |
| Toàn bộ suite | **123/123** (trước 118) |
| Pre-flight đường ống thật (CPU, dữ liệu giả, 6 bước) | **PASS** — `post_net.temporal=True` từ config, nhánh 3.520 tham số, sau 6 bước `post_strength` −2,6e-4 (thoát zero), `\|in_conv_t\|` 2,0e-4 (sống), đường thời gian làm output khác đường per-frame |

## 4. Đăng ký trước

**Thay đổi so với công thức E4 là DUY NHẤT một biến**: `model.post_temporal: true`
(cùng `post_base: 64`, cùng loss, cùng 16 epoch, seed 0, `w_budget: 0.0` giữ nguyên PRE như E4).
Đối chứng: `outputs/eval_e4_full` — h264 −5.71 / h265 −4.22 (n=1159, cùng instrument).

**Gate cơ chế (trước khi nói về BD):** `|in_conv_t.weight| > 1e-3` và output khác đường
per-frame (nhánh thực sự được dùng, không chỉ "sống").

**Kỳ vọng:** tốt hơn ≥ 0,5 pp ở ít nhất một codec, gap rule PASS. Đây là chỗ tôi đặt kỳ vọng
bước nhảy lớn nhất trong bốn họ, vì nó thêm *thông tin* chứ không thêm *trọng số*.

**Band thất bại đăng ký trước:** BD phẳng hoặc xấu hơn ⇒ restoration per-frame là đủ ở
capacity này; báo cáo và quay về scale POST 4.6M (lever đã đo, đơn điệu).

**Rủi ro:** (a) vượt cap 12h/session — `resume: true` đã bật, chấp nhận chạy 2 phiên;
(b) nhánh có thể học được rất ít nếu artifact chủ yếu là không-gian (blocking/ringing) —
khi đó kết quả là null trung thực.

## 5. Lệnh chạy

```bash
KAGGLE_ACCOUNT=<acct> python ops/push_kernel.py train \
  --commit <sha> --config configs/sandwich_temporal.yaml \
  --overrides "train.epochs=16" --accelerator NvidiaTeslaT4 \
  --slug-suffix=-temporal
# khi xong: eval 3 shard rồi merge
python ops/push_kernel.py eval --commit <sha> --config configs/sandwich_temporal.yaml \
  --train-kernel <acct>/u9-train-temporal --shard-idx N --num-shards 3 \
  --accelerator NvidiaTeslaT4
python ops/merge_eval.py <s0> <s1> <s2> --out outputs/eval_temporal --bootstrap 10000
```
