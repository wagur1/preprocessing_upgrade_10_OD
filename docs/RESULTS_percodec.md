# RESULTS — v9 per-codec POST (shared trunk + codec FiLM)

## Kết quả (shards 0+1 = 770/1159 seqs, 5k bootstrap)

| Arm | BD h264 | CI95 | BD h265 | CI95 | P(BD<0) |
|---|---|---|---|---|---|
| prep (PRE-v1) | −1.51% | [−3.88, +0.96] | −1.53% | [−3.38, +0.33] | 0.892 / 0.945 |
| sandwich (per-codec POST) | −1.76% | [−4.30, +1.01] | −2.90% | [−4.70, −0.93] | 0.896 / 0.998 |

## Sửa đúng theo audit 2026-09-11 (finding #5): warm-start KHÔNG exact

Kiểm tra với FiLM đã train (net[2] non-zero, mô phỏng checkpoint thật):
max diff giữa v8.post_restore và v9.post_restore(codec=h264) sau load =
**0.0012 — KHÔNG phải exact**. `load_v8_sandwich` bỏ toàn bộ
`post_net.film.*` của v8 (cond width 1→9 không tương thích), nên v9-a khởi
đầu từ **v8-trừ-FiLM** chứ không phải v8. Kết luận "alternating STE phá
specialization từ cùng warm-start" cần nới thành: **v9-a bắt đầu từ một điểm
khởi tạo hơi khác v8 (mất affine đã học của FiLM) VÀ chịu alternating STE** —
hai nhân tố trộn nhau, không thể quy hết cho STE. Kết quả âm của v9-a vẫn
đúng về mặt đo lường; phân tích nguyên nhân bị hạ xuống "không chắc chắn".
(Test cũ dùng v8 fresh-init với FiLM zero — không chứng minh được gì về
checkpoint thật, đúng như audit chấm.)

## Verdict: KHÔNG giữ được kỷ lục nào (downside scenario ~25% đã đăng ký)

- h264 −1.76 so với kỷ lục E2 **−5.89** (cùng warm-start!) — POST sau 1079 bước
  alternating STE **mất** phần lớn giá trị h264 của chính nó
- h265 −2.90 so với kỷ lục STE **−3.56**
- Gates cho biết routing ĐÃ học (FiLM 0→0.151, embeddings phân biệt) nhưng
  trunk chia sẻ +continued STE kéo POST về điểm trung bình compromise —
  đúng rủi ro "FiLM conflict" đã đăng ký trong MODEL_PERCODEC.md

## Bài học (bổ sung vào bản đồ falsification)

1. **Specialization đã có là tài sản mong manh**: tiếp tục train trên nhiều
   codec với trunk chung PHÁ huỷ specialization hiện có nhanh hơn là học
   routing mới (−5.89 → −1.76 chỉ sau 1079 bước lr 3e-5).
2. Điều này + E1/E3 xác nhận một nguyên lý: trong regime dữ liệu/budget này,
   **per-codec specialization thắng mọi dạng sharing/co-training** — cấu hình
   tối ưu vẫn là ghép module đã chuyên biệt (frankenstein E2).
3. Nghi vấn mở: E1's POST (tuned x264) thua E2's POST (tuned h265) TRÊN h264 —
   dynamics của STE fine-tune không đơn giản là "chuyên về codec trong vòng lặp".

## Số liệu cuối của toàn chiến dịch (xem v8/docs/RESULTS_sandwich.md)

Best single model: **E2 frankenstein-STE — h264 −5.89% [−7.93,−3.76] /
h265 −2.79% [−4.24,−1.25]** (full n=1159, P=1.000/1.000).
Per-codec records: h264 E2 −5.89 / h265 STE-sandwich −3.56.


## v9-b DUALCODEC — per-codec PRE + per-codec POST (shards 0+1 = 770/1159, 5k boot)

| Arm | BD h264 | CI95 | BD h265 | CI95 | P(BD<0) |
|---|---|---|---|---|---|
| prep (per-codec PRE) | −2.66% | [−5.18, −0.21] | −1.36% | [−2.97, +0.31] | 0.982 / 0.943 |
| **sandwich (full union)** | **−6.96%** | **[−9.44, −4.32]** | **−3.33%** | **[−5.04, −1.43]** | **1.000 / 1.000** |

**VERDICT: union THÀNH CÔNG và VƯỢT kỳ vọng cấu trúc.** h264 −6.96 vượt kỷ lục
E2 (−5.89) thêm 1.07pp — per-codec PRE routing (PRE-v1 cho x264, STE-PRE cho
x265) cộng hưởng với POST thay vì chỉ cộng từng phần. h265 −3.33 giữ gần kỷ lục
STE (−3.56, trong dung sai ±1pp). Gap rule PASS cả hai codec.

### FULL n=1159 (10k bootstrap) — SỐ CHỐT

| Arm | BD h264 | CI95 | BD h265 | CI95 | P(BD<0) |
|---|---|---|---|---|---|
| **sandwich (full union)** | **−5.88%** | **[−7.93, −3.76]** | **−3.53%** | **[−4.96, −2.09]** | **1.000 / 1.000** |

**Union chính xác theo cấu trúc**: h264 −5.88 ≈ kỷ lục E2 (−5.89), h265 −3.53 ≈
kỷ lục STE (−3.56) — cùng một checkpoint, P(BD<0)=1.000 cả hai codec, gap PASS.
Số partial (−6.96/−3.33 trên 770 seqs) là sampling noise; full-n là chuẩn.

### CI ĐÃ SỬA (bootstrap multiplicity — audit #1) — số hợp lệ cuối cùng

| Arm | BD h264 | CI95 (hợp lệ) | BD h265 | CI95 (hợp lệ) |
|---|---|---|---|---|
| sandwich | −5.89% | [−8.54, −3.09] | −3.56% | [−5.39, −1.62] |

(CI pre-fix [−7.93,−3.76]/[−4.96,−2.09] giữ làm historical record — sai vì
dedup ~63% clip/resample.) Đối đầu E4 (bootstrap hợp lệ, full-n): h264 v9-b
thắng nhẹ (−5.89 vs −5.71, noise), h265 E4 thắng (−4.22 vs −3.56) — CI chồng
lấn: hai kiến trúc (per-codec specialization vs shared-capacity) ngang tầm,
mỗi bên thắng một codec.

**Kết luận cuối chiến dịch (sửa)**: v9-b và E4 chia ngôi theo codec trong
một hệ thống deployable — v9-b = router theo codec, E4 = một head lớn.
v9-b giữ đồng thời cả hai kỷ lục codec riêng của từng cấu hình đơn.
Bảng 3 tầng: sharing thất bại (v9-a −1.76) < ghép modul đơn codec (E2 −5.89) <
per-codec union (v9-b −5.88/−3.53, cả hai codec cùng lúc).
