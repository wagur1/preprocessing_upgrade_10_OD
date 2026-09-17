# preprocessing_upgrade_10_OD — preprocessing cho Object Detection

Line riêng cho task **Object Detection** của VCM: ảnh → preprocessing → codec đóng băng
(All-Intra) → decode → detector đóng băng → mAP. Đích là một con số BD-rate âm trên **trục mAP**
để điền ô Object Detection còn trống của báo cáo MPEG w21834.

Repo này tách ra sau khi **bốn hướng học-máy liên tiếp đo ra âm** trên chế độ ảnh:

| Hướng | Kết quả đo |
|---|---|
| Checkpoint AR zero-shot trên ảnh | vô hại cho mAP (ratio 1,02–1,03) nhưng **+2,29 % bit** (CI [+0,43,+4,86]), sandwich **+6,71 %** |
| Train PRE cho ảnh (proxy intra + loss detector) | **phá 25 % mAP** trước cả codec (ratio 0,75) |
| Đầu không gian (AR) | −5 pp so với kỷ lục |
| Temporal POST (AR) | −3 pp so với kỷ lục |

Đọc chung: cơ chế kiếm bit của thiết kế AR là **thời gian**; ở ảnh (T=1) nó vô dụng, còn lại chỉ
là các module *thêm/sửa* cấu trúc — thứ mAP không thưởng. Nên hướng đi ở đây đảo ngược:
**bỏ nền, giữ vật**, không dùng editor học được.

## Tài liệu

- **Spec:** [`docs/OD_DESIGN.md`](docs/OD_DESIGN.md) — bài toán, bằng chứng, thiết kế R0 (0 tham số)
  và R1 (gate học được ~1–2k tham số), interface, tiêu chí thành công/phản chứng.
- **Kế hoạch implement:** [`docs/superpowers/plans/2026-09-17-od-preprocessing.md`](docs/superpowers/plans/2026-09-17-od-preprocessing.md)
  — 6 task theo TDD, có decision gate giữa R0 và R1.

## Trạng thái

| Bước | Trạng thái |
|---|---|
| Core detection (data, analyzer, probe, pusher) | ✅ đã port, test xanh |
| R0 — mask từ detector + suppression nền | ✅ implement + test (`tests/test_mask_suppress.py`) |
| Chạy R0 trên Kaggle (COCO val, 500 ảnh) | ⏳ đang chạy |
| R1 — gate học được | ⏸ chỉ mở nếu R0 dương |

## Chạy

```bash
pytest -q                                   # 134 test

# R0 trên Kaggle (eval-only, không train, ~20 phút)
python ops/push_detection_probe.py --commit <sha> --account <acct> \
    --script ops/probe_background_suppression.py \
    --extra-args "--sigmas 4,8,16 --score 0.5 --dilate 0.15" \
    --ckpt-dataset "" --n-images 500 --size 320 --slug u9-probe-bgsuppress
```

## Ràng buộc (áp cho mọi thí nghiệm ở đây)

- Codec/bitstream/decoder **đóng băng**; chỉ can thiệp ở miền pixel trước encode và sau decode.
- Mọi số công bố dùng **held-out analyzer + CI bootstrap** và luật gap (`≥ −0.05` mọi QP, cả hai codec).
  Không có số on-teacher.
- Ảnh là đơn khung: `T=1`, codec intra-only (`codec.inter: false`).
- Box của torchvision là **xyxy**, COCO cần **xywh** — luôn đi qua `_coco_box`.
- Mọi split trong index phải **khác rỗng**: val rỗng sẽ âm thầm tắt model selection và early stopping.
- Run dài trên Kaggle: ghi diagnostics ra **file** trong output dir (cell bị cap 12h mất sạch stdout).
https://vsllm.com
