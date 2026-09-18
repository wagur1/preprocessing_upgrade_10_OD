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
| Chạy R0 trên Kaggle (COCO val, 500 ảnh) | ✅ **thắng** — xem bảng dưới |
| POST Gaussian toàn khung sau R0 (0 bit) | ❌ **bị loại** — gap FAIL cả hai codec |
| Cổng theo QP cho POST (T=40 h264 / 45 h265) | ⏳ đăng ký trước, `u10-gaussian-post-gate500` |
| R1 — gate học được | ⏸ mở được (R0 đã dương), nhưng chưa có lý do để ưu tiên |

### Kết quả probe POST (2026-09-18, `htran123456/u10-gaussian-post-500`)

Ba arm dùng chung round trip: `prep` (R0) và `post` (decode của prep + Gaussian toàn khung σ=1,
không mã hoá lại) có **đúng** cùng bpp — kiểm chứng 5000/5000 ô của `pairing.jsonl` trùng
SHA-256 decode và bpp. Vì vậy chênh lệch mAP theo QP là của riêng bộ lọc.

| codec | arm | BD vs anchor | gap | BD post vs prep |
|---|---|---:|---|---:|
| h264 | prep (R0) | −15,689 | PASS | — |
| h264 | post | −20,524 | **FAIL** (−0,0632) | −4,311 % |
| h265 | prep (R0) | −11,249 | PASS | — |
| h265 | post | −6,571 | **FAIL** (−0,0651) | **+5,923 %** |

POST toàn khung **bị loại**: nó thắng ở QP cao nhưng mất 4,0 pp mAP ở QP30 (h264) và trên h265
còn tệ hơn R0. Delta mAP theo QP cho thấy nó là bộ khử nhiễu, chỉ có lợi khi nhiễu nén chiếm ưu
thế — nên biến thể **cổng theo QP** (bật POST khi QP ≥ T) mới là thứ đáng thử. Giả thuyết đó đã
đăng ký trước tại [`docs/RUN_DESIGN_gaussian_post_gate.md`](docs/RUN_DESIGN_gaussian_post_gate.md)
với T cố định 40/45, và đang được xác nhận trên 500 ảnh mới hoàn toàn
(`configs/r0_gate500_ids.json`).

### Kết quả R0 (2026-09-18, `htran123456/u10-gaussian-correct-500`)

BD-rate trên **trục mAP** so với anchor `codec(x)`; âm = ít bit hơn ở cùng mAP.
n=500 ảnh COCO val2017 (320px), QP 30–50, analyzer Faster R-CNN R50-FPN COCO_V1 đóng băng,
`cover` (tỉ lệ pixel được bảo vệ) = 0,588.

| codec | blur4 | blur8 | blur16 |
|---|---:|---:|---:|
| h264 | −15,69 | −16,69 | −16,89 |
| h265 | −11,25 | −11,44 | −12,10 |

Luật gap PASS ở mọi QP trên cả hai codec (mức xấu nhất −0,024 mAP tại QP30, ngưỡng −0,05).
Đây là **điểm**; CI bootstrap theo ảnh đang tính offline ở
`vcm_deepseek/r0_gaussian500_analysis/` — `status` giữ `running` cho tới khi đủ 1000 draw.
Artifact: `outputs/probe_bgsuppress/` của run corrected (`probe_bgsuppress.json` +
`per_image_records.npz`); hai run độc lập cho cùng kết luận (run cũ −14,83/−16,57/−16,12 trên h264,
anchor của hai run trùng khớp tuyệt đối).

Đọc đúng: mAP **thấp hơn** ở hai mức bitrate cao (h264 QP30: 0,2806 so với anchor 0,3036) và
**cao hơn** ở QP50 — thắng về rate, không phải về accuracy.

## Chạy

```bash
pytest -q                                   # 239 test

# R0 trên Kaggle (eval-only, không train)
python ops/push_detection_probe.py --commit <sha> --account <acct> \
    --script ops/probe_background_suppression.py \
    --ids-file configs/r0_500_ids.json --out-name probe_bgsuppress \
    --extra-args "--sigmas 4,8,16 --score 0.5 --dilate 0.15 --device cuda" \
    --n-images 500 --size 320 --slug u10-probe-bgsuppress

# POST Gaussian toàn khung sau R0 (3 arm: anchor / prep / post)
python ops/push_detection_probe.py --commit <sha> --account <acct> \
    --script ops/probe_gaussian_post.py \
    --out-name probe_gaussian_post --ids-file configs/r0_500_ids.json \
    --extra-args "--prep-sigma 4 --post-sigma 1 --score 0.5 --dilate 0.15 --device cuda" \
    --slug u10-gaussian-post-500
```

`--extra-args` phải nằm trên **một dòng**: một `\n` literal bị bash đọc thành đối số `n` và
probe chết với `unrecognized arguments: n n n n` (đã xảy ra một lần; xem commit `fcc2f19`).
Danh sách ID dài đi qua `--ids-file`, không nhồi vào chuỗi extra-args.

## Ràng buộc (áp cho mọi thí nghiệm ở đây)

- Codec/bitstream/decoder **đóng băng**; chỉ can thiệp ở miền pixel trước encode và sau decode.
- Mọi số công bố dùng **held-out analyzer + CI bootstrap** và luật gap (`≥ −0.05` mọi QP, cả hai codec).
  Không có số on-teacher.
- Ảnh là đơn khung: `T=1`, codec intra-only (`codec.inter: false`).
- Box của torchvision là **xyxy**, COCO cần **xywh** — luôn đi qua `_coco_box`.
- Mọi split trong index phải **khác rỗng**: val rỗng sẽ âm thầm tắt model selection và early stopping.
- Run dài trên Kaggle: ghi diagnostics ra **file** trong output dir (cell bị cap 12h mất sạch stdout).
https://vsllm.com
