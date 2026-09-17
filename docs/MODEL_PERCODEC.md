# MODEL_PERCODEC.md — v9 design note

Đầy đủ luận cứ thiết kế + bảng bằng chứng v8: xem README (bảng) và
v8/docs/RESULTS_sandwich.md (8 thí nghiệm).

Điểm thiết kế then chốt:
1. **Shared trunk + codec FiLM** (chọn thay vì 2 UNet riêng): basis phục hồi
   cấp thấp (deblock, deringing) là chung x264/x265 — chia sẻ giúp 1500 bước
   STE calibration đủ (2 UNet riêng cần train lại gần như từ đầu). Ablation
   tự nhiên cho paper: duplicated-UNet arm.
2. **Warm-start exactness**: FiLM codec zero-init ⇒ khởi đầu hành vi = v8 STE
   (chứng minh bằng test_v8_warmstart_exact_at_load). Mọi thay đổi sau STE là
   công của routing, không phải noise khởi tạo.
3. **Engine codec plumbing**: STECodec.codec → post_restore(codec=...) ở cả
   train/val/eval; eval loop đã biết codec (vòng lặp name in h264/h265).

Rủi ro đã đăng ký: FiLM một đầu không đủ capacity phân tách 2 chế độ artifact
(downside); STE 1500 bước có thể thiên lệch về codec cuối cùng nếu sampling
không cân bằng (engine samples QP mỗi bước; codec lấy từ config — kernel train
v9 sẽ override ste_codec theo lịch xen kẽ nếu cần, xem ops notes).
