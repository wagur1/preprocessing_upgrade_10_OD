# RUN_DESIGN_spatial.md — khôi phục đầu không gian (W) và mở khoá M2

**Ngày:** 2026-09-16 · **Trạng thái:** thiết kế đã đăng ký, chưa chạy GPU
**Repo:** `pre_processing_upgrade_9` (thay đổi chưa commit) · **Config:** `configs/sandwich_spatialprobe.yaml`

---

## 1. Bằng chứng (đo trên checkpoint thật, 2026-09-16)

Ba phép đo độc lập trên checkpoint kỷ lục E4 (`ds_u8-bigpost-s1-ckpt/preprocessor.pth`,
post_base=64, 2.33M tham số) và trên toàn bộ 12 checkpoint sandwich trên máy:

**(a) M2 — ROI editor chết vì dead saddle.**
`_EditorBlock.out` zero-init **và** `edit_strength` zero-init ⇒ cả hai thừa số bằng 0 tuyệt đối:

```
grad(edit_strength) = <dL/dframes2, W · edit>            = 0   vì edit ≡ 0
grad(editor params) = edit_strength · W · dL/dframes2    = 0   vì edit_strength ≡ 0
```

Đo bằng autograd: `|grad| = 0.000000e+00` cho cả `edit_strength`, `editor.out.weight` và
`editor.enc1[0].weight`. Mọi checkpoint đều có `edit_strength == +0.000000` và
`|W_editor_out| == 0` ⇒ **M2 chưa từng hoạt động trong bất kỳ kết quả nào đã ghi**.

**(b) S head sụp.**
`W = sigmoid(s_net(...))` trên mọi checkpoint: `mean(W) ≈ 1e-8 … 5e-9`, logits `≈ −59 … −61`
(activation trong `h1/h2` vẫn khoẻ, nên đây là sụp *có học*, không phải ReLU chết).
Hệ quả: gate của M1 là `(1−W) ≈ 1` khắp khung (edit toàn cục, không còn "background-only"),
gate M3 cũng `≈ 1` (freeze toàn cục), còn M2 bị gate bởi `W ≈ 0`.

**(c) Hệ quả trực tiếp: sửa init M2 là KHÔNG đủ.**

| Cấu hình | mean W | \|grad edit_strength\| | Kết luận |
|---|---|---|---|
| legacy (W tự do) | 6.9e-9 | 0.0 | chết |
| sửa init M2, W tự do | 9.5e-9 | 1.7e-14 | dưới eps của Adam (1e-8) → đóng băng |
| sửa init M2 + `w_budget` (S cũ) | 1.7e-3 (spike) | 1.4e-8 | biên |
| **sửa init M2 + reset S + `w_budget=0.25`** | **0.250** | **1.6e-6** | **dùng được (160× eps)** |

Bản thiết kế probe đầu tiên (chỉ sửa init M2) đã bị **chính phép đo này bác bỏ trước khi
tốn GPU** — minh chứng cho giá trị của bước kiểm tra cục bộ.

---

## 2. Thay đổi trong code

| File | Thay đổi |
|---|---|
| `src/models/upvcm.py` | `_EditorBlock.out` đổi từ zero-init sang `N(0, 1e-3)` (giữ bias = 0) — cùng cách sửa mà nửa POST cần ở v8 (`3a948521e5db`). Thêm `w_budget` + `_budget_normalise()` (chuẩn hoá theo **tổng**, nên không sụp được: map bão hoà trả về map đều đúng bằng budget). |
| `src/models/upvcm.py` (forward) | Khi `w_budget > 0`, chuẩn hoá luôn `_last_w_target` cùng công thức ⇒ `rho` học *hình dạng* saliency thay vì *biên độ*. |
| `src/models/{sandwich,dualpost_sandwich,percodec_sandwich}.py` | truyền `w_budget` xuống PRE |
| `src/engine.py` | `_optimizer` chỉ nhận tham số `requires_grad`; thêm `train.freeze_except` (khớp theo segment tên), `_repair_dead_editor()` (sửa conv ra all-zero sau khi nạp checkpoint), `_reinit_modules()` + `train.reinit_prefixes` (reset module đã bão hoà sau khi nạp) |
| `tests/test_upvcm.py` | 2 test hồi quy: gradient M2 **khác 0** (test cũ chỉ kiểm `is not None` — một tensor bằng 0 vẫn qua, nên nó báo "editor learns" trong khi editor chết), và `w_budget` chống sụp kể cả khi logits = −60 |

Toàn bộ suite: **118/118 pass** (trước: 116).

---

## 3. Hai giai đoạn chạy

### Giai đoạn 1 — probe đầu không gian (rẻ, ~2h GPU)

Warm-start E4, **đóng băng mọi thứ trừ S và M2** (`freeze_except: [s_net, editor, edit_strength]`),
reset S (`reinit_prefixes: [s_net]`), `w_budget: 0.25`, 3 epoch, lr 3e-4.
Điểm xuất phát **đúng bằng** model kỷ lục (edit_strength = 0 ⇒ editor là identity; S reset chỉ
đổi W chứ không đổi cấu trúc), nên mọi thay đổi BD đều quy được cho đầu không gian.

Kiểm tra cục bộ đã chạy: sau 10 bước synthetic, `edit_strength` đi từ 0 → −2.7e-3,
gradient tăng 1.2e-6 → 8.3e-5, **24 tensor của đầu không gian dịch chuyển, 0 tensor khác dịch
chuyển** (`tools/probe_m2_warmstart_check.py`).

### Giai đoạn 2 — full retrain (đắt, ~11h GPU)

Chỉ chạy nếu probe xanh: train lại từ đầu 16 epoch với `w_budget` + init đã sửa
(không freeze), cùng protocol bigpost (3 shard, held-out analyzer, bootstrap 10k) ⇒
đây mới là ứng viên headline.

---

## 4. Đăng ký trước (pre-registration)

**Gate cơ chế (phải qua TRƯỚC khi nói bất cứ điều gì về BD):**
- `mean(W)` trong ±20 % của `w_budget` và `std(W) > 0.05` (map có cấu trúc, không phải hằng số);
- `|edit_strength| > 0.01` (M2 thật sự mở);
- `corr(W, saliency_target) > 0.3` (đầu không gian học được đúng chỗ máy nhìn).

**Kỳ vọng BD (n=1159, exploratory):**
- Nếu cơ chế đúng: h264 tốt hơn **−5.71** và/hoặc h265 tốt hơn **−4.22** ít nhất **1 pp**;
  dự đoán nghiêng về h264 (nơi mọi lever không-gian đã đo đều dương tính) và về các clip
  có nền tĩnh/đối tượng lớn (đúng pattern per-class đã đo).
- **Band thất bại được đăng ký trước:** BD xấu hơn E4, hoặc tốt hơn < 0.3 pp (nằm trong
  nhiễu ~1 pp giữa các lần chạy) ⇒ trục không gian-vật lý không phải lever ở điểm vận hành
  này. Báo cáo trung thực, không đổi chỉ số để cứu.

**Luật gap vẫn giữ nguyên:** `prep − anchor ≥ −0.05` tại mọi QP, cả hai codec. Vi phạm = loại.

**Rủi ro đã biết (ghi trước để không đổi giọng sau):**
1. S reset làm W đổi khắp khung ⇒ gate của M1/M3 đổi *biên độ* so với lúc huấn luyện
   (dec/stab được tune khi W ≈ 0). Probe vì vậy đo "đầu không gian có giá trị không",
   không đo "hệ đã hội tụ". Đây là lý do giai đoạn 2 tồn tại.
2. `w_budget` là một hyperparameter mới; 0.25 là điểm khởi đầu hợp lý (ROI của clip
   Kinetics thường chiếm 10–30 % khung), chưa được tối ưu.
3. Nếu `rho` vẫn không thắng được áp lực rate, W có thể tái sụp *trong* map —
   gate `std(W) > 0.05` bắt được điều này.

---

## 5. Lệnh chạy (giai đoạn 1)

```bash
# 1) đặt checkpoint E4 vào đúng chỗ finetune đọc
#    (Kaggle: dataset chứa preprocessor.pth của bigpost-s1)
mkdir -p outputs/sandwich_spatialprobe/checkpoints
cp <e4_ckpt>/preprocessor.pth outputs/sandwich_spatialprobe/checkpoints/preprocessor.pth

# 2) train probe (T4), theo ops chuẩn của repo
python ops/push_kernel.py train --commit <sha> --config configs/sandwich_spatialprobe.yaml \
    --accelerator NvidiaTeslaT4

# 3) gate cơ chế trên checkpoint thu được
python ops/gates_sandwich.py --ckpt outputs/sandwich_spatialprobe/checkpoints/preprocessor.pth \
    --index data/index/kinetics_hash_split.json

# 4) eval 3 arm, 3 shard, held-out analyzer, rồi merge + bootstrap
python ops/push_kernel.py eval --commit <sha> --config configs/sandwich_spatialprobe.yaml \
    --train-kernel <acct>/u9-train-spatialprobe --shard-idx N --num-shards 3 \
    --accelerator NvidiaTeslaT4
python ops/merge_eval.py <shard0> <shard1> <shard2> --out outputs/eval_spatialprobe --bootstrap 10000
```

So sánh với `outputs/eval_e4_full` (h264 −5.71, h265 −4.22, n=1159) — cùng instrument.

---

## 6. Việc chưa làm (đề xuất thứ tự)

1. ĐÃ PORT sang **v8**: `upvcm.py`, `sandwich.py`, `tests/test_upvcm.py` (hai repo dùng chung
   file model; không port thì nhánh v8 vẫn sản sinh checkpoint chết-M2). Các helper engine
   (`freeze_except`, `_repair_dead_editor`, `reinit_prefixes`) **chỉ có ở v9** — vì vậy
   `configs/sandwich_spatialprobe.yaml` chỉ đặt ở v9; chạy nó trên v8 sẽ train toàn bộ model
   mà không báo lỗi.
2. POST 4.6M — lever đã đo là đơn điệu và chưa bão hoà; đây là fallback độ chắc chắn cao.
3. Temporal POST (3-D conv ở bottleneck) — POST hiện xử lý từng frame độc lập; đây là
   inductive bias video chưa dùng, nhưng là thay đổi kiến trúc lớn hơn nên xếp sau.
