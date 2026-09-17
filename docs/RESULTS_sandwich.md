# RESULTS — SANDWICH (điền khi có kết quả)

> Template tạo TRƯỚC khi chạy. Kỳ vọng đã đăng ký trong docs/MODEL_SANDWICH.md.

## Trạng thái

| Bước | Trạng thái |
|---|---|
| Tests + smoke (96/96) | ✅ |
| Probe Kaggle | ✅ (qua train kernel) |
| Train 16ep sandwich v1 | ✅ COMPLETE (best epoch 14/16, ~11.3h, `nguyenhoanglan1232/u8-train-sandwich`) |
| Gates trên v1 ckpt | ✅ đọc checkpoint: PRE gates ≈ v7 (dec=−0.6239, edit=0.0000, stab=−0.0785) — **POST strength = 0.0000: POST CHẾT (dead-saddle bug, giống M2 của v7)** |
| **Bug fix + v2 retrain** | ✅ commit `3a948521e5db`, kernel `hieusunday0412/u8-train-v2` đang chạy |
| Eval v1 | ⏭️ BỎ QUA (POST identity ⇒ arm sandwich ≡ arm prep — không tốn quota cho số trùng) |
| Eval v2 (3 arm) | sau khi v2 train xong |

**Phát hiện v1:** POST không bao giờ mở — cùng dead-saddle như M2 v7 (zero
out conv × zero gate ⇒ gradient kép ≡ 0). Nhân bản độc lập của cùng bug trên
2 repo củng cố chẩn đoán. Eval v1 bị bỏ có chủ đích; mọi hy vọng sandwich nằm
ở v2 (đã fix init).

## Comparators

| Arm/đối thủ | BD h264 [CI] | BD h265 [CI] | gap |
|---|---|---|---|
| anchor (không prep) | 0% | 0% | — |
| prep-only (cùng ckpt, bypass post) | −0.09% [−2.26,+2.14] | −1.40% [−3.01,+0.35] | PASS |
| **sandwich (claim chính)** | **−4.25% [−6.73,−1.58]** | −1.48% [−3.34,+0.50] | PASS |
| lineage best (Zhao kappa=10, v6) | −3.42% [−5.88,−0.88] | −2.63% [−4.49,−0.79] | PASS |

## E2: Frankenstein-STE (PRE-v1 + POST-STE-h265, FULL n=1159, 10k bootstrap) — SỐ TỐT NHẤT DỰ ÁN

| Arm | BD h264 | CI95 | BD h265 | CI95 | P(BD<0) |
|---|---|---|---|---|---|
| prep (PRE-v1) | −2.46% | [−4.47, −0.40] | −0.78% | [−2.22, +0.73] | 0.991 / 0.844 |
| **sandwich** | **−5.89%** | **[−7.93, −3.76]** | **−2.79%** | **[−4.24, −1.25]** | **1.000 / 1.000** |

So với frankenstein (không STE): h264 −4.82 → **−5.89 (+1.07pp từ STE-POST)**, h265
−2.38 → −2.79 (+0.41pp). Cộng tính giữ vững lần thứ 3. CI h264 upper (−3.76) đã
sâu hơn mean của lineage best (−3.42). Gap rule PASS cả hai.

## E1: frankenstein + STE-x264 (shards 0+1 = 770 seqs, 5k bootstrap)

| Arm | BD h264 | CI95 | BD h265 | CI95 |
|---|---|---|---|---|
| sandwich | −4.49% | [−7.06, −1.80] | −1.96% | [−3.87, +0.06] |

**Verdict: STE theo codec là TRADE-OFF, không cộng dồn.** POST fine-tune trên
x264 (E1): h264 −4.49 (thua E2 −5.89), h265 −1.96 (thua STE-h265 −3.56). So sánh
3 biến thể STE cho thấy POST mạnh nhất cho codec nó fine-tune, lùi trên codec
kia: h264 record = E2 (POST-h265, vành đai proxy), h265 record = STE (POST-h265).
Mô hình đúng cho paper: **per-codec POST heads** (hoặc FiLM theo codec), không
một POST duy nhất cho cả hai.

## E3: co-adaptation (2ep STE-x264 từ frankenstein; shards 0+1 = 770 seqs, 5k bootstrap)

| Arm | BD h264 | CI95 | BD h265 | CI95 |
|---|---|---|---|---|
| sandwich | −3.07% | [−5.45, −0.48] | −2.92% | [−4.75, −1.00] |

**Verdict: co-adaptation (lr 2e-5, 2ep) KHÔNG vượt E2** (−3.07 vs −5.89 h264;
−2.92 vs −2.79 h265 — ngang). POST strength đóng bớt (0.052→0.047) trong khi
PRE giữ nguyên: fine-tune kéo cả hai về điểm cân bằng giữa, mất sự chuyên biệt
của từng nửa. Shard 2 đang chạy cho merge full. Bài học: frankenstein giữ
nguyên trạng (không co-adapt nhẹ) là cấu hình tốt nhất.

## TTO screen (100 clip, steps=30, frankenstein-STE)
tto+h264 −3.79% (acc 0.508→0.568) / tto+h265 **+6.09%** (acc 0.520→0.546).
TTO residual giúp x264 nhưng PHÁ x265 — cùng bất đối xứng proxy-block-8/x264.
Cần steps/mục tiêu theo codec hoặc bỏ arm h265 nếu tiếp tục. 100-clip = diagnostic.

## STE stage-2 FULL (x265-in-loop, n=1159, 10k bootstrap)

| Arm | BD h264 | CI95 | BD h265 | CI95 | P(BD<0) |
|---|---|---|---|---|---|
| prep+codec | −1.60% | [−3.40, +0.21] | −1.60% | [−2.90, −0.22] | 0.957 / 0.988 |
| **sandwich+codec** | −4.13% | [−6.02, −2.18] | **−3.56%** | **[−4.96, −2.09]** | 1.000 / 1.000 |

**h265 −3.56% là số tốt nhất từng đo trên codec này** (STE-h265 fix asymmetry:
joint không-STE −2.05 → −3.56, +1.5pp). Per-codec record holders: h264 = E2
frankenstein-STE (−5.89), h265 = STE sandwich (−3.56). E1 (frankenstein + STE-x264,
đang eval) nhắm hợp nhất cả hai.

## STE stage-2 (x265-in-loop 400 bước; shards 0+1 = 770 seqs, 5k bootstrap)

| Arm | BD h264 | CI95 | BD h265 | CI95 | P(BD<0) |
|---|---|---|---|---|---|
| prep+codec | −0.56% | [−2.72, +1.67] | −1.36% | [−2.97, +0.31] | 0.690 / 0.943 |
| **sandwich+codec** | −3.40% | [−5.63, −0.96] | **−3.33%** | **[−5.04, −1.43]** | 0.996 / **1.000** |

**STE kéo h265 lên ~1.3pp** (−2.05 joint không-STE full-n → −3.33 STE 770-seq;
dung sai ±1pp vẫn để lại cải thiện thật) mà **không phá h264** (−3.40 vs −4.45
joint — hơi lùi trong khoảng noise, đánh đổi chấp nhận được vì h264 đã mạnh).
Điều này xác nhận giả thuyết bất đối xứng codec: POST học được artifact x265
khi thấy x265 thật trong vòng lặp. Full merge khi shard 2 xong.

## Frankenstein (PRE=v7-v1 best + POST=v8-v2, FULL n=1159, 10k bootstrap)

| Arm | BD h264 | CI95 | BD h265 | CI95 |
|---|---|---|---|---|
| prep (PRE-v1) | −2.46% | [−4.47, −0.40] | −0.78% | [−2.22, +0.73] |
| **sandwich (PRE-v1 + POST)** | **−4.82%** | **[−6.96, −2.60]** | **−2.38%** | **[−3.85, −0.82]** |

P(BD<0): 1.000 (h264) / 0.999 (h265); gap rule PASS cả hai.

**Verdict cộng tính:** POST ghép vào PRE chưa từng co-train vẫn cho **+2.36pp
(h264) / +1.60pp (h265) thuần** — POST là **add-on portable**, không cần
co-training với PRE cụ thể. Frankenstein thắng joint sandwich trên cả hai codec
(h264 −4.82 vs −4.45; h265 −2.38 vs −2.05, trong dung sai ±1pp) với sự khác biệt
chủ yếu đến từ PRE mạnh hơn (v1 −2.46 vs v8-PRE −0.85). Sanity: frank prep arm
đối chiếu chính xác số pre-only v1 (−2.46/−0.78) — instrumentation nhất quán.

**Hàm ý:** (1) hướng triển khai "POST add-on trên hạ tầng codec sẵn có" (không
đụng encoder) có bằng chứng; (2) co-adaptation fine-tune frankenstein (Tier 2)
là bước tự nhiên — nếu POST tận dụng được PRE mạnh, khoảng cách tới −6…−7% còn;
(3) bảng paper có dòng "portable POST" hoàn chỉnh so với joint training.

## Kết quả FULL (v2, n=1159, 10k bootstrap) — số chốt

| Arm | BD h264 | CI95 | BD h265 | CI95 |
|---|---|---|---|---|
| prep+codec | −0.85% | [−2.66, +0.99] | −1.92% | [−3.21, −0.54] |
| **sandwich+codec** | **−4.45%** | **[−6.57, −2.31]** | **−2.05%** | **[−3.55, −0.50]** |

P(BD<0): sandwich h264 **1.000** (10k/10k resamples âm), h265 0.995. Gap rule
PASS cả hai codec. POST pure value h264: +3.60pp (−4.45 vs −0.85).

## Kết quả preliminary (v2, shards 0+1 = 770/1159 seqs, 5k bootstrap)

| Arm | BD h264 | CI95 | BD h265 | CI95 | P(BD<0) |
|---|---|---|---|---|---|
| prep+codec | −0.09% | [−2.26, +2.14] | −1.40% | [−3.01, +0.35] | 0.512 / 0.942 |
| **sandwich+codec** | **−4.25%** | **[−6.73, −1.58]** | −1.48% | [−3.34, +0.50] | **0.998** / 0.927 |

### Gates (từ checkpoint, best epoch 13)
```
PRE: dec=−0.340 (M1 mở)  edit=0.000 (M2 vẫn đóng ở v8)  stab=−0.069 (M3 mở)
POST strength = +0.052  → MỞ (dead-saddle fix có tác dụng)
```

## Đọc kết quả

- **Giá trị thuần của POST trên h264: +4.16pp** (sandwich −4.25% vs prep −0.09%, cùng
  checkpoint, cùng bpp — phép tách cơ chế sạch nhất có thể). Held-out analyzer ⇒ KHÔNG
  phải teacher-overfit: POST khôi phục được accuracy cho analyzer chưa từng thấy.
- **Sandwich vượt lineage best trên h264** (−4.25% vs −3.42%, CI dịch trái toàn phần:
  upper −1.58% vs −0.88%). Trên h265 POST gần như không cộng (+0.08pp) — bất đối xứng
  codec: POST học đảo artifact khớp x264 hơn (proxy block-8 gần x264 4×4/8×8 hơn là
  x265 block lớn + SAO).
- PRE của v8 yếu hơn v7 trên h264 (−0.09 vs −2.46): POST "án ngữ" vai trò khôi phục
  trong joint training, làm PRE bớt dốc — chính là trade-off thiết kế của sandwich.
- Số này là 2/3 shards; merge full n=1159 khi shard 2 xong (dự kiến lệch ≤ ±0.5pp).
