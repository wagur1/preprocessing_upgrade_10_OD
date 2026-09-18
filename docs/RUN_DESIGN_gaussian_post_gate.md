# RUN_DESIGN — QP-gated Gaussian POST, pre-registered confirmation

**Ngày đăng ký:** 2026-09-18 · **Trạng thái:** đăng ký trước khi chạy
**Kết quả sẽ đọc:** `u10-gaussian-post-gate500` (500 ảnh **mới**, `configs/r0_gate500_ids.json`)

---

## 1. Đã đo được gì (n=500, `u10-gaussian-post-500`)

Probe 3 arm, `prep` và `post` chia sẻ **đúng một** tensor decode và **đúng một** bpp
(kiểm chứng 5000/5000 dòng `pairing.jsonl`: cùng SHA-256 decode, cùng bpp), nên chênh lệch
mAP theo từng QP cô lập đúng bộ lọc POST.

| codec | arm | BD vs anchor | gap | BD vs prep |
|---|---|---:|---|---:|
| h264 | prep (R0) | −15,689 | PASS | — |
| h264 | post (Gaussian toàn khung σ=1) | −20,524 | **FAIL** | −4,311 % |
| h265 | prep (R0) | −11,249 | PASS | — |
| h265 | post | −6,571 | **FAIL** | **+5,923 %** |

mAP(post) − mAP(prep) theo QP 30/35/40/45/50:

* h264: **−0,0402 −0,0241 +0,0058 +0,0288 +0,0204**
* h265: **−0,0481 −0,0354 −0,0158 +0,0117 +0,0150**

Cơ chế: Gaussian σ=1 đóng vai trò **khử nhiễu**. Khi mã hoá còn sạch (QP thấp) nó phá chi
tiết → mAP giảm; khi nhiễu nén chiếm ưu thế (QP cao) nó giúp detector → mAP tăng, ở **cùng bpp**.

⇒ POST toàn khung **bị loại** (vi phạm luật gap ở cả hai codec; trên h265 còn tệ hơn R0).

## 2. Giả thuyết được đăng ký

Chỉ bật POST ở nửa bitrate cao, nơi nó giúp:

```
arm gated:  QP >= T  -> post(codec(prep(x)))     (Gaussian σ=1 sau decode)
            QP <  T  -> prep(codec(prep(x)))     (không lọc)
```

Đây là quy tắc **phía decoder**, không cần side information: QP có sẵn trong bitstream.

**Ngưỡng chốt trước, không chọn lại trên tập mới:**

| codec | T |
|---|---|
| h264 | **40** |
| h265 | **45** |

## 3. Bằng chứng nội bộ trước khi đăng ký (không thay thế xác nhận)

Chia 500 ảnh cũ thành 2 nửa (theo SHA-256 của id), **chọn ngưỡng trên một nửa, chấm điểm
trên nửa kia** (`tools/gate_gaussian_post.py`; chọn theo luật của dự án: luật gap phải PASS
rồi lấy BD âm nhất):

| chọn trên | ngưỡng chọn được | BD trên nửa held-out | prep-only cùng nửa | lợi |
|---|---|---|---:|---:|---:|
| A (221) | h264 QP≥40 | −19,276 | −13,680 | **−5,60 pp** |
| B (279) | h264 QP≥40 | −25,690 | −19,688 | **−6,00 pp** |
| A (221) | h265 QP≥45 | −11,174 | −9,411 | **−1,76 pp** |
| B (279) | h265 QP≥45 | −13,847 | −11,782 | **−2,07 pp** |

Hai nửa **độc lập chọn cùng một ngưỡng** và nửa còn lại đều xác nhận, luật gap PASS cả bốn
lần. Nhưng: *họ quy tắc* (một ngưỡng trên lưới QP) được đề xuất **sau khi đã nhìn** 500 ảnh
này, nên đây chưa phải bằng chứng độc lập — đó là lý do tồn tại của run này.

Lưu ý độ nhiễu: BD của chính arm prep-only lệch **−19,69 (A) vs −13,68 (B)**, tức ±3 pp giữa
hai nửa. Lợi ích ~6 pp lớn hơn độ lệch đó, nhưng không lớn hơn nhiều — n=500 là biên.

## 4. Đọc kết quả thế nào (chốt trước)

Trên 500 ảnh **mới**, với T cố định ở bảng §2:

* **XÁC NHẬN** nếu BD(gated) tốt hơn BD(prep-only) **≥ 1,0 pp** trên ít nhất một codec,
  **và** luật gap PASS trên cả hai codec.
* **NULL** nếu lợi < 1,0 pp trên cả hai codec ⇒ đóng họ POST, giữ R0 làm kết quả OD.
* **PHẢN CHỨNG** nếu lợi âm (gated tệ hơn prep) hoặc luật gap FAIL ⇒ ngưỡng học từ 500 ảnh
  cũ không chuyển được; ghi nhận và đóng họ.

Mọi số vẫn là **exploratory** (`held_out: False`, `ctc_conformant: False`) cho tới khi có
một run CTC-conformant.

## 5. Không nằm trong phạm vi

* Quét thêm σ cho POST (giữ σ=1 đúng như đã đo).
* Chọn lại ngưỡng trên tập mới — làm vậy là quay lại đúng lỗi mà run này sinh ra để tránh.
* Thay đổi PRE (giữ nguyên R0: mask từ detector, σ=4, score 0.5, dilate 0.15).
