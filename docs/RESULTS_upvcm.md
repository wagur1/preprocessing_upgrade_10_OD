# RESULTS — UP-VCM (điền khi có kết quả)

> Template tạo TRƯỚC khi chạy. Comparators đóng từ lineage (cùng instrument:
> protocol chuẩn n=1159, held-out r2plus1d_18, QP30–50, preset medium).

## Trạng thái chạy

| Bước | Trạng thái | Ghi chú |
|---|---|---|
| Tests + smoke local (88/88) | ☐ | full pipeline trên dữ liệu giả |
| Probe hạ tầng Kaggle | ☐ PASS | fingerprint 30f083f8520a, codec checks |
| Train 16ep UP-VCM | ☐ | kernel `u7-upvcm-train`, T4 |
| Gates G1–G4 | ☐ | `ops/gates_upvcm.py` |
| Eval sharded n=1159 | ☐ | kernels `u7-eval-shard*` |
| Merge + bootstrap CI + gap rule | ☐ | `ops/merge_eval.py` |

## Comparators (từ v6, cùng protocol)

| Biến thể | BD h264 [CI95] | BD h265 [CI95] | gap rule |
|---|---|---|---|
| (anchor) không pre-processing | 0% (định nghĩa) | 0% | — |
| kappa=10 @16ep (Zhao editor, lineage best) | −3.42% [−5.88, −0.88] | −2.63% [−4.49, −0.79] | PASS |
| kappa=10 @16ep rep1 | −2.52% [−4.92, +0.16] | −2.33% [−4.14, −0.57] | PASS |

Đọc với dung sai ±1pp (re-run noise tầng eval/codec).

## Kết quả UP-VCM

### Gates

```
[G1 non-identity] dec=____ edit=____ stab=____ (>0.01)
[G2 no-blow-up]   RMS ____ (<0.14)
[G3 W healthy]    mean ____ std ____
[G4 deploy purity] diff ____
```

### BD-Rate (protocol chuẩn)

| Codec | BD-Rate | CI95 (bootstrap) | P(BD<0) | gap rule |
|---|---|---|---|---|
| h264 | | | | |
| h265 | | | | |

### Per-QP

| QP | anchor h264 | prep h264 | Δbpp h264 | anchor h265 | prep h265 | Δbpp h265 |
|---|---|---|---|---|---|---|
| 30 | | | | | | |
| 35 | | | | | | |
| 40 | | | | | | |
| 45 | | | | | | |
| 50 | | | | | | |

### Module utilization (từ checkpoint)

| Gate | Giá trị | Đọc |
|---|---|---|
| dec_strength (M1) | | >0: nền bị decimate |
| edit_strength (M2) | | >0: ROI edit hoạt động |
| stab_strength (M3) | | >0: ổn định thời gian hoạt động |

## Đọc kết quả (viết sau khi có số)

- So anchor (no-prep) — claim chính của mô hình mới.
- So kappa=10 (mô hình cũ) cùng instrument, dung sai ±1pp: UP-VCM có qua mặt
  lineage best không, và trên codec nào.
- Cơ chế: Δbpp@QP30 (M1/M3 phải ép xuống so với +14% của additive), gap per
  QP (M2 phải giữ/leo ở QP nặng).
