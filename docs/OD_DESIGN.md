# Spec — preprocessing for Object Detection (upgrade 10)

**Ngày:** 2026-09-17 · **Trạng thái:** thiết kế đã chốt, chưa implement xong
**Mục tiêu:** tạo ra một con số BD-rate **âm** trên trục mAP cho task Object Detection
(ô còn trống của báo cáo MPEG w21834), trong ràng buộc: codec đóng băng (x264/x265),
không đổi bitstream, không đổi decoder, analyzer đóng băng + protocol held-out/CI.

---

## 1. Bài toán

Ảnh → **preprocessing** → codec chuẩn (All-Intra) → decode → **postprocessing (0 bit)** →
detector đóng băng → mAP.

Đại lượng đo: `BD-Rate(mAP)` của arm có tiền xử lý so với anchor `codec(x)`, hai codec,
QP 30–50, CI bootstrap theo ảnh, và luật gap (accuracy prep ≥ anchor − 0.05 tại mọi QP).

## 2. Bằng chứng đã đo (không suy diễn lại)

| Hướng | Kết quả |
|---|---|
| Checkpoint AR zero-shot trên ảnh | vô hại cho mAP (ratio 1,02–1,03) nhưng **+2,29 % bit** (h264, CI [+0,43,+4,86]) và **+6,71 %** cho sandwich — tức *có hại về rate* |
| Train PRE cho ảnh (proxy intra, loss detector) | **phá 25 % mAP** ngay trước codec (ratio 0,75) |
| Đầu không gian trên AR | −5 pp so với kỷ lục (đóng họ "phân bổ không gian" phía AR) |
| Temporal POST trên AR | −3 pp so với kỷ lục (đóng họ temporal) |

**Diễn giải then chốt:** cơ chế kiếm bit của thiết kế AR là **thời gian** (M3 đóng băng nền →
residual liên khung ≈ 0). Ở ảnh (T=1) M3 vô dụng; thứ còn lại (M1/M2 dạng học) chỉ *thêm hoặc
sửa* cấu trúc — mà mAP thì không thưởng cho việc đó. Ngược lại, toàn bộ literature detection
ra số lớn đều làm **một** việc: **bỏ nền, giữ vật** (ROI-Packing −44,1 %, dual-region JPEG
−26,2 %, Różek VCIP 2023) — không cần editor học được.

## 3. Thiết kế

### R0 — Suppression bằng mask từ chính detector (0 tham số, không train)

```
x ──► detector đóng băng (phân tích phía encoder) ──► boxes
   ──► dilate ──► mask bảo vệ (1 = giữ)
   ──► x' = x*mask + blur_sigma(x)*(1−mask)      ← chỉ phá NGOÀI mask
   ──► codec → decode → (POST) → detector → mAP
```

- Không có tham số, không có gì để overfit proxy, không có gì để chọn ⇒ **không thể** lặp lại
  kiểu thất bại "học xong phá 25 % mAP".
- Không cần side information: decoder chỉ decode; mask chỉ dùng ở phía encoder.
- Tham số duy nhất là σ (quét 4/8/16) và biên dilate (0,15).

### R1 — Gain bảo vệ học được (chỉ mở nếu R0 dương)

Một module ~1–2k tham số đọc `(mask, QP, x)` và xuất **hệ số bảo vệ per-pixel** trong [0,1],
nhân vào độ mạnh suppression. Độ tự do duy nhất: *phá bao nhiêu, ở đâu* — nó không thể thêm
cấu trúc, nên không thể tái hiện chế độ thất bại của editor.

Train: `L = λ_task·L_det(post(codec(x'))) + μ·L_D + β·bpp`, có **pha STE với codec thật** để
không đánh nhau với proxy (nguyên nhân nghi phạm của thất bại "phá 25 % mAP").

## 4. Interface

| Thành phần | Chữ ký | Ghi chú |
|---|---|---|
| `protect_mask(boxes, scores, labels, size, score_thresh, dilate)` | `→ [1,1,S,S]` | 1 = bảo vệ; box dưới ngưỡng điểm **không** được bảo vệ |
| `suppress(x, mask, sigma)` | `→ x'` cùng shape | vùng mask **không đổi một bit**; σ=0 là no-op |
| `GatedSuppress(nn.Module)` (R1) | `forward(x, mask, cond)` | identity tại init; gate per-pixel ∈ [0,1] |
| `ops/probe_background_suppression.py` | CLI | 2 arm (anchor vs suppressed) × σ × codec × QP, lưu per-image records để tính CI offline |
| `src/tasks/object_detection.py` | `TaskAnalyzer` | detector đóng băng; `L_Acc` = loss của detector; buffer không đổi |
| `src/data/coco_det.py` | dataset + collate | T=1, box gt scale theo squash-resize |

## 5. Tiêu chí thành công / phản chứng

- **R0 win** nếu BD(suppressed) < 0 ở ít nhất một codec với **CI loại 0**, và đường mAP không
  nằm dưới anchor ở mọi QP. Ghi lại `cover` (tỉ lệ ảnh được bảo vệ) — nó nói cơ chế có bao
  nhiêu đất để làm việc.
- **R0 null** nếu |BD| < 0,3 pp ⇒ đóng họ suppression; báo cáo trung thực.
- **R0 fail** nếu BD > 0 ⇒ mAP dùng ngữ cảnh nhiều hơn literature gợi ý; ghi lại như một phát
  hiện (đây là câu hỏi khoa học thật, không phải thất bại kỹ thuật).
- **R1** chỉ được mở khi R0 win, và phải thắng R0 ở cùng σ.

## 6. Không nằm trong phạm vi

- Đổi codec/bitstream/decoder (vi phạm định vị "retrofit").
- Feature coding (cần NN phía decoder — niche khác).
- Báo cáo số on-teacher: mọi số công bố phải ở protocol held-out + CI.

## 7. Rủi ro đã biết

1. Detector **dùng ngữ cảnh** → phá nền quá tay ở bitrate thấp có thể mất vật nhỏ. Đây chính
   là điều R0 đo.
2. `cover` lớn (nhiều box) ⇒ ít đất để phá ⇒ BD nhỏ. Đọc `cover` trước khi diễn giải BD.
3. Proxy vs codec thật ở chế độ intra — R1 bắt buộc có STE.
