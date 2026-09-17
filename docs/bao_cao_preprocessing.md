# Báo cáo: Adaptive Video Preprocessing cho VCM (bản 3GB)

*Task chính: nhận dạng hành động (Kinetics-400 5%). Ngày: 2026-08-21.*

## 1. Tổng quan hệ thống

Bài toán: **Video Coding for Machines (VCM)** — nén video sao cho **máy** (mô
hình thị giác) vẫn nhận dạng tốt, không phải cho mắt người. Đóng góp của ta nằm
ở **khâu tiền xử lý (preprocessing)**: học các chỉnh sửa pixel **TRƯỚC** một
codec tiêu chuẩn, không đụng vào encoder (không QP/ROI phía mã hoá).

Pipeline (chỉ preprocessor được huấn luyện, codec + analyzer **đóng băng**):

```
x  ──►  Preprocessor θ  ──►  x_pre  ──►  Codec (frozen)  ──►  x̂  ──►  Analyzer (frozen)  ──►  nhãn máy
        (U-Net + FiLM + SFT)            (CompressAI /                (r3d_18)
                                         x264 / x265)
```

- **Codec proxy khi train:** `bmshj2018-factorized` (CompressAI) — khả vi, cho
  ước lượng bpp để backprop.
- **Codec thật khi eval:** x264 / x265 (ffmpeg, constant-QP) — đo bpp thật từ
  file mã hoá, nên bpp trung thực và so được với proxy.
- **Analyzer:** `r3d_18` (video ResNet-18) đóng băng, cho `L_task` + đặc trưng
  để distill.

Ý tưởng cốt lõi: preprocessor **dọn** thông tin mà máy không cần và **giữ/làm
nổi** thông tin máy cần, để ở cùng một ngân sách bit thì độ chính xác cao hơn
(hoặc cùng độ chính xác thì tốn ít bit hơn).

## 2. Kiến trúc preprocessor (`src/models/preprocessor.py`)

U-Net 2D chạy **theo từng khung hình** (`[B·T, C, H, W]`), tín hiệu thời gian duy
nhất là *motion cue* đưa vào SFT — rẻ trên T4 nhưng vẫn "biết" chuyển động. Sự
mạch lạc thời gian của chỉnh sửa do **loss** `L_temp` ép, không dùng conv 3D.

| Thành phần | Vai trò |
|---|---|
| `_UNet` (3 tầng: full, /2, /4 + skip) | Bộ chỉnh sửa pixel chính. Học **residual** δ, không học lại ảnh. |
| `_ConvBlock` = conv→act→conv→**FiLM**→**SFT** | Khối cơ bản; mọi khối đều được điều biến bởi rate (FiLM) và motion (SFT). |
| `FiLM(cond)` | Affine **theo kênh, toàn cục** từ điều kiện rate (QP chuẩn hoá). Cho **một** model thích ứng cả dải bitrate thay vì học trung bình. |
| `SFT(cue)` | Affine **theo không gian** từ motion cue → dồn chỉnh sửa vào vùng **chuyển động / liên quan tác vụ**. Đây là điểm mới cho video. |
| `_motion_cue` | `|x_t − x_{t-1}|` trung bình kênh, chuẩn hoá theo clip → `[B,1,T,H,W]`. Khung 0 mượn diff của khung 1. |
| `tail` (Conv **zero-init**) | Khởi tạo 0 → mạng bắt đầu là **identity** (x_pre = x), train ổn định, chỉ "bật" chỉnh sửa khi học được. |
| `res_scale` | Hệ số biên độ residual: `x_pre = clamp(x + res_scale·δ, 0, 1)`. `<1` thu nhỏ chỉnh sửa. |

FiLM và SFT cũng **zero-init** (lớp cuối = 0) → lúc đầu điều biến = phép đồng
nhất. Toàn bộ đảm bảo điểm xuất phát là identity, không phá ảnh.

**Vì sao thiết kế này** (ghi trong docstring): baseline cũ dùng stack residual
hai nhánh không gian/thời gian, học ra một ảnh "mờ trung bình rate" và **thua**
đường RD của codec. U-Net + FiLM + SFT tách bạch: U-Net làm biên tập viên, FiLM
điều chỉnh theo điểm rate, SFT điều chỉnh theo vùng chuyển động.

## 3. Hàm mục tiêu (`src/losses.py`)

$$L = \lambda_{task}\,L_{task} + \omega\,L_{distill} + \beta\,L_{rate} + \tau\,L_{temp}\;(+\,\delta\,L_{\delta})\;(+\,\gamma\,L_{TV})$$

**Cố ý KHÔNG có MSE-to-source.** Term ép x̂ bám pixel gốc chính là thứ khiến
baseline không bao giờ đạt BD-Rate âm — nó trực tiếp chống lại nén. Thay vào đó
ta giữ *ngữ nghĩa* bằng distillation đặc trưng và để rate thật "cắn".

| Term | Công thức | Vai trò |
|---|---|---|
| `L_task` | CE của analyzer trên x̂ | Giữ độ chính xác máy — mục tiêu cuối. |
| `L_distill` | MSE đặc trưng frozen-analyzer(source) vs (x̂), chuẩn hoá theo scale | Giữ ngữ nghĩa codec hay phá; **mạnh nhất ở bitrate thấp**. Source là target detach, gradient chỉ chảy qua nhánh x̂. |
| `L_rate` | bpp từ entropy model của proxy codec | Áp lực nén. |
| `L_temp` | MSE giữa *delta liên khung* của x̂ và của source | Giữ chuyển động, chống flicker mà **không** ghim pixel tuyệt đối (điểm mới video). |
| `L_δ` (tùy chọn, mặc định 0) | `mean|x_pre − x|` | Đòn bẩy **thưa chỉnh sửa** trên *đầu vào* (không phải x̂). |
| `L_TV` (tùy chọn, mặc định 0) | `TV(x_pre)` = năng lượng tần số cao không gian | Proxy chi phí bit **không phụ thuộc codec**: mọi codec DCT/wavelet/block đều tốn bit cho high-freq → phạt TV giúp **transfer sang x264/x265** (thứ `β·bpp` chỉ-của-proxy không chạm tới). `L_task` giữ lại cạnh máy cần. |

Trọng số mặc định: `lam_task=1.0, omega=0.5, beta=0.1, tau=0.1, delta=0, gamma=0`.
Trong các run báo cáo dưới đây **β được nâng lên 2** để nén thực sự cắn.

## 4. Điều kiện rate & lỗi đã sửa

Preprocessor nhận `cond ∈ [0,1]` = QP chuẩn hoá qua `_qp_norm(qp)=(qp−20)/(51−20)`
(dải `qp_ref=[20,51]`), đưa vào FiLM. Một model phủ cả dải rate.

**Lỗi train/eval (đã sửa, commit 1784040):** trước đây eval nhánh
prep+compressai dùng chuẩn hoá theo **chỉ số quality** trong khi train dùng
chuẩn hoá theo **QP**. Ví dụ quality 5: train thấy cond 0.065 còn eval thấy
0.429 → prep+compressai **chưa bao giờ đúng miền** lúc đo, làm **mọi số BD-Rate
cũ vô hiệu**. Sửa bằng `_quality_conds()`: đảo bảng `qp_to_quality` của train →
QP đại diện mỗi quality → `_qp_norm`, nội suy tuyến tính cho quality chưa train.
Lỗi **chỉ ở khâu đo**; huấn luyện luôn đúng → checkpoint vẫn hợp lệ, chỉ số đo bị
sai và đã chạy lại.

## 5. Thiết lập huấn luyện (run báo cáo)

| Mục | Giá trị |
|---|---|
| Data | Kinetics-400 5%, **cap 3GB** (index cân bằng lớp, không copy) |
| Analyzer | `r3d_18` frozen; clip 112px |
| Preprocessor | U-Net base_ch=32, res_scale=1.0, cond_dim=1 |
| Proxy codec | `bmshj2018-factorized`, qualities [1,2,3,5,8] |
| Clip train | 16 khung, frame_size 128, stride 2 |
| QP train | [22,27,32,37,42] → quality {5,3,2,1,1} |
| Loss | β=2, ω=0.5, τ=0.1, λ=1, δ=0, γ=0 |
| Optim | Adam lr 1e-4, cosine decay, epochs 3, max_steps 300, batch 4 |
| Đa hạt giống | seed ∈ {0,1,2} (đổi cả subset+split lẫn RNG train) → CI theo phương sai dữ liệu |
| Eval | x264/x265 QP [30,35,40,45,50], preset medium |

## 6. Chỉ số đánh giá (`src/metrics/bd_rate.py`)

BD-Rate cổ điển so hai codec ở **cùng chất lượng** (PSNR). Ở đây trục "chất
lượng" là **độ chính xác máy** (top-1), tích phân log-rate như thường lệ:

- **BD-Rate** (%): chênh bitrate trung bình test vs anchor ở **cùng accuracy**.
  Âm = tiết kiệm bit = tốt.
- **BD-Accuracy**: chênh accuracy trung bình ở **cùng bitrate**.

Hai cách so:
- `bd_prep_gain` — **cùng codec, chỉ khác có/không preprocessor** (prep+h265 vs
  h265). Đây là **claim thật** (QP so được).
- `bd_vs_anchor` — chéo codec (prep+compressai vs h264/h265). **Vô nghĩa** vì QP
  không so được giữa các codec; chỉ để tham khảo.

## 7. Kết quả

### 7.1 Kết quả chính — 3-seed CI ở 3GB (β=2, γ=0)

`bd_prep_gain` (cùng codec; âm = tiết kiệm bit), tổng hợp trên seed {0,1,2},
t-test một phía H1: mean<0:

| Cặp | BD-Rate trung bình | p(<0) | Diễn giải |
|---|--:|--:|---|
| prep+compressai vs compressai | **−2.30%** | 0.188 | In-domain tiết kiệm nhẹ, **chưa** có ý nghĩa thống kê |
| prep+h264 vs h264 | +3.91% | 0.798 | Transfer break-even/hơi âm hại |
| prep+h265 vs h265 | +3.18% | 0.885 | Tương tự |

Kết luận: ở cap 3GB, hiệu ứng in-domain đúng chiều (−2.3%) nhưng **nằm dưới sàn
nhiễu** (std ~3.5%, CI ≈ [−6.3, +1.7]); transfer sang codec thật chưa đạt.

### 7.2 Ablation — núm γ (TV loss) điều khiển in-domain ↔ transfer

Sweep có kiểm soát tại seed=0 (anchor trùng khít giữa các run → cùng dữ liệu;
lưu ý các run này ở cap5):

| γ | compressai (in-domain) | h264 | h265 |
|--:|--:|--:|--:|
| 0.00 | −1.87 | +5.68 | +3.36 |
| 0.01 | −0.66 | +3.66 | +0.61 |
| 0.02 | +2.42 | +2.89 | **−3.55** |
| 0.03 | +1.90 | **−3.92** | −0.41 |

Xu hướng: γ tăng → in-domain xấu đi, transfer tốt lên. γ=0.03 là điểm đầu tiên
**cả hai codec thật cùng ≤0 BD-Rate** và cùng **+BD-Accuracy** (+0.014 h264,
+0.012 h265) — đổi lại in-domain mất. Đường cong không hoàn toàn đơn điệu ở
điểm-điểm → xác nhận dao động 1-seed cỡ 3–4%, phải multi-seed mới chốt được.

## 8. Nhận xét

1. **Cơ chế γ là đóng góp có giá trị.** Nó là một *núm* điều khiển đánh đổi
   in-domain↔transfer, có thể giải thích và tái lập — đây là "mechanism" chứ
   không chỉ một con số. Phù hợp làm một mục ablation mạnh.

2. **Đường cong cắt nhau là bình thường.** prep thắng ở low–mid bitrate (dọn
   nhiễu, tăng accuracy — đúng regime VCM) và thua nhẹ ở bitrate cao (accuracy
   đã bão hoà, chỉnh sửa chỉ thêm bit). BD-Rate đã tích phân cả hai; số âm nghĩa
   là phần thắng trội. Với VCM (bit thấp) đánh đổi này đúng hướng.

3. **Điểm mạnh thật nằm ở low-rate + accuracy**, không ở BD-Rate whole-curve. Ở
   bit cực thấp, prep tăng accuracy tương đối +50…+130% — headline đúng tinh
   thần VCM hơn là con số BD-Rate khiêm tốn.

4. **Bottleneck là loss/proxy + trần vật lý của preprocessing, không phải kiến
   trúc hay thiếu data.** Tăng data (5→30GB) chủ yếu mua **ý nghĩa thống kê**
   (CI hẹp, p<0.05) và bump mean nhẹ, **không nâng trần**. Preprocessing-only
   chỉ sửa pixel — không phân bổ bit, không đổi transform — nên whole-curve
   thực tế dừng ở một chữ số / thấp hai chữ số.

## 9. Hạn chế & bước tiếp theo

- **Hạn chế:** hiệu ứng nhỏ và (ở 3GB) chưa có ý nghĩa thống kê; transfer chỉ
  đạt khi hi sinh in-domain; h264 là codec "lì" nhất.
- **Tiếp theo:**
  1. Chọn điểm vận hành (γ≈0.02–0.03 cho transfer) rồi chạy **5-seed CI** để có
     p-value tin cậy cho paper.
  2. Cân nhắc nâng data lên ~30GB để đạt p<0.05 (điểm cân bằng compute; full
     Kinetics-400 không lọt Kaggle free và không nâng trần).
  3. Khung paper: **transfer + low-rate + cơ chế γ + hai task (AR + tracking)**,
     để in-domain là hỗ trợ chứ không phải headline đơn độc. Mức 10–15%
     in-domain **có ý nghĩa thống kê** là đủ tầm Q2 nếu định vị đúng.

---
*Nguồn số liệu: 3-seed CI (b2_s0/s1/s2, cap3) và γ-sweep (cap5, seed0) trong
`outputs/*/eval/results.json`; tổng hợp bằng `kaggle/report_ci.py`.*


