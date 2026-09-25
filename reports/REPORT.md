# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Cù Đức Quang

Công cụ gán nhãn đã dùng: CVAT chạy local bằng Docker (`http://localhost:8080`). Mình tạo task
`day8_round1` với nhãn `car` kiểu rectangle, import pre-label bằng định dạng *Ultralytics YOLO
Detection 1.0*, sửa trên job của task, rồi export lại đúng định dạng đó và đóng gói bằng
`tools/pack_labels.py to_label/round1 --yolo-dir <thư mục export>/labels/train`. Không sửa trực tiếp
file nhãn.

Nguồn số liệu dùng trong báo cáo: `reports/rounds_table.md`, `outputs/metrics_round0.json`,
`outputs/metrics_round1.json`, `outputs/selection_round1.csv`, `outputs/round1_diff.md/.json`,
`outputs/compare_round0.jpg`, `outputs/compare_round1.jpg`, `outputs/selection_round2.csv`, cùng log
huấn luyện trong `notebooks/day8_active_learning.ipynb` (chỉ để chẩn đoán, không dùng làm số test).
Nhãn test do model tạo, chưa được người rà, nên mọi số AP/P/R dưới đây là **mức khớp với bộ tham
chiếu đó**, không phải độ đúng tuyệt đối.

## 1. Dữ liệu và cách chia tập

Theo `data/DATA.md`, 400 frame lấy từ một video 160 giây, camera cố định trên cầu vượt, 2.5
frame/giây. Hai frame liên tiếp chỉ cách nhau 0.4 s, và một chiếc xe ở trong khung vài giây. Vì vậy
tập được chia theo trục thời gian: 20 ảnh test ở 4 đoạn (tâm 20, 60, 100, 140 s), 112 ảnh vùng đệm
bị bỏ (±4 s quanh mỗi đoạn test), còn lại 268 ảnh pool. Ảnh pool gần test nhất vẫn cách 4.4 s.

Nếu chia ngẫu nhiên, gần như mọi ảnh test sẽ có một ảnh "anh em" cách 0.4 s trong pool. Khi đó cùng
một chiếc xe, cùng vị trí và ánh sáng, xuất hiện cả lúc train lẫn lúc test (rò rỉ dữ liệu). Mô hình
được chấm trên những xe gần như đã thấy, nên AP50/recall trên test sẽ **cao hơn thực tế**. Mức tăng
sau fine-tune cũng bị phóng đại, vì model chỉ cần "nhớ" xe chứ không cần tổng quát hóa. Vùng đệm là
để khoảng cách thời gian đủ lớn, xe ở test đã rời khung trước khi ảnh pool gần nhất xuất hiện.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0 từ `reports/rounds_table.md`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Theo `metrics_round0.json`: 403 box tham chiếu (bỏ 14 box cao dưới 16 px), TP 197, FP 16, FN 206.
Model cold start **rất thận trọng**: độ chính xác (precision) 0.925, nhưng ở ngưỡng 0.25 bỏ sót hơn
nửa số xe (recall 0.489). AP50 0.771 cao hơn nhiều so với F1, vì AP50 tính trên mọi ngưỡng: nhiều xe
vẫn được model thấy, chỉ với độ tin cậy dưới 0.25.

Trên `compare_round0.jpg` (4 ảnh test 0050, 0150, 0250, 0350; xanh lá là TP, vàng là FN, đỏ là FP),
các loại xe không khớp tham chiếu:

- **Xe chiều ngược bật đèn pha, nhất là xe gần camera.** Ở frame_0250 (TP 6, FP 2, FN 9), xe lớn ở
  góc dưới trái và hai xe đèn pha ở làn giữa, làn trái bị bỏ sót. Ở frame_0350 (TP 9, FP 2, FN 14), hầu hết xe đèn pha ở
  nửa trái bị bỏ sót. Quầng sáng che thân xe, trong khi COCO chủ yếu là ảnh ban ngày.
- **Xe ở mép phải ảnh bị cắt hoặc nhoè** (frame_0250, hai box vàng sát mép phải).
- **Xe nhỏ ở xa**, chỉ còn cụm đèn hậu: recall small chỉ 0.182 trên 66 box.

Recall theo kích thước (`det_eval.py`: small < 32², medium < 96² px): small 0.182, medium 0.547,
large 0.561. Xe nhỏ gần như không được phát hiện. Đáng chú ý là xe lớn cũng chỉ đạt 0.561 (41 box):
khó khăn không chỉ do kích thước mà do điều kiện đêm và lóa đèn. Xe gần camera hướng về phía mình có
quầng pha lớn nhất.

Một trường hợp cần người rà lại tham chiếu trước khi kết luận model sai: ở frame_0150, cụm xe đèn hậu
đỏ bên phải (khoảng x 470–520 trên nửa ảnh thu nhỏ) có tham chiếu 3 box chồng nhau, còn model cho 2–3
box lệch và bị tính FP. Ở frame_0050, model có box đỏ (FP) trên xe nhỏ phía xa ở giữa trên, nơi tham
chiếu vẽ box lệch hoặc không có box. Vì tham chiếu do model tạo, các FP kiểu này có thể là xe thật mà
tham chiếu thiếu hoặc vẽ khác IoU. Cần mở ảnh gốc xem trước khi tính đó là lỗi của model.

## 3. Chiến lược chọn mẫu

`score = W_U·U + W_A·A + W_D·D`, với `W_U, W_A, W_D = 0.5, 0.3, 0.2` (ô cấu hình notebook):

- **U (độ bất định):** với mỗi box model đưa ra (ngưỡng thấp 0.05), tính `u = 1 − |2c − 1|`. Box có
  c ≈ 0.5 được u ≈ 1, box rất chắc (c ≈ 1) hoặc rất yếu (c ≈ 0) được u ≈ 0. U của ảnh là trung bình 5
  box bất định nhất, nên ảnh có vài xe khiến model "phân vân" được ưu tiên.
- **A (số box mơ hồ):** số box có 0.15 ≤ c < 0.5, chia cho giá trị lớn nhất trong pool. Ảnh có nhiều
  xe nằm dưới ngưỡng pre-label 0.25 hoặc sát ngưỡng thì A cao, tức nhiều chỗ người phải quyết định.
- **D (đa dạng thời gian):** thưởng ảnh xa các ảnh đã gán (trong phạm vi 10 s). Ở vòng 1 chưa có ảnh
  gán nào nên `D = 1` cho mọi ảnh.
- **`MIN_GAP_S = 2.0`:** chọn tham lam theo điểm, bỏ ứng viên cách ảnh đã chọn dưới 2 s để một lô 12
  ảnh không bị lấp bởi các frame gần trùng của cùng một đoạn video.

Dẫn chứng (chi tiết trong `reports/SELECTION.md`):

- frame_0182 (hạng 1, score 0.9591, U 0.9182, A 1.0): 18/28 box mơ hồ. Khi rà thêm 10 box, trong đó
  có SUV tối cắt mép dưới và xe nhoè góc phải, đúng như `BLIND_SCAN.md` dự đoán.
- frame_0369 (hạng 2, 0.9324): 43 box ở ngưỡng 0.05 nhưng chỉ 14 box ở ngưỡng 0.25. Khi rà thêm 13 box.
- frame_0392 (hạng 15, 0.8874): U 0.9747 cao nhất lô nhưng A chỉ 0.6667. Nó vào lô vì ba ảnh hạng 6,
  9, 12 bị loại do gần trùng.
- Frame khác: frame_0372 (hạng 6, 0.9101) bị `MIN_GAP_S` loại vì cách 0369 chỉ 1.2 s, dù điểm cao hơn
  7 ảnh được chọn. Với ngân sách năm ảnh, mình còn bỏ cả frame_0380 (hạng 3) và frame_0331 (hạng 5) vì
  cách 0369 là 4.4 s và cách 0326 là 2.0 s. Công gán ảnh thứ hai của cùng cụm xe cho ít thông tin mới.

Điểm bất định **không chứng minh** ảnh đó sẽ cải thiện mô hình. Nó đo model phân vân trên box đã có,
không đo model sai. Lỗi lớn nhất tìm thấy lại là bỏ sót hoàn toàn (85 box thêm so với 7 box xóa trong
`round1_diff.md`), loại lỗi không sinh box nên không vào U hay A. Hơn nữa, ảnh bất định có thể bất
định vì chính nó mơ hồ (xe chỉ còn chấm đèn): gán nhãn không nhất quán ở những ảnh đó còn có thể làm
model học nhiễu. Chỉ có đánh giá trên cùng tập test sau fine-tune mới trả lời được câu hỏi này, và
vòng này không có baseline chọn ngẫu nhiên để so.

## 4. Các vòng học chủ động (active learning)

Bảng từ `reports/rounds_table.md` (20 ảnh test, 403 box tham chiếu, IoU 0.5, P/R/F1 ở conf 0.25):

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 247 | 0.588 | -0.184 | 1.000 | 0.191 | 0.321 | 0.000 | 0.182 | 0.561 |

`metrics_round1.json` ghi `n_train_images = 12` và `n_train_boxes = 247`, khớp đúng `final_boxes = 247`
trong `round1_diff.json`. Vậy mô hình vòng 1 được train trên đúng lô đã sửa, và được đánh giá trên cùng
20 ảnh test với vòng 0 (`reference: data/test/labels`, 403 box, bỏ 14 box nhỏ ở cả hai vòng).

**Vòng 1: mức sửa nhãn gợi ý** (`outputs/round1_diff.md`). Model cold start đề xuất 169 box cho 12 ảnh,
sau khi sửa còn 247 box: **giữ nguyên 159, chỉnh 3, xóa 7, thêm 85** (accept rate 94%). Pre-label ít box
sai nhưng thiếu nhiều xe: 85/247 = 34% box cuối cùng là xe model bỏ sót. Hai ảnh phải thêm nhiều nhất là
frame_0369 (+13) và frame_0182 (+10).

**AP50 thay đổi:** 0.771 → 0.588, tức **−0.184 so với cold start**. Vòng trước chính là cold start nên
so với vòng trước cũng là −0.184. Theo `metrics_round*.json`: TP 197 → 77, FP 16 → 0, FN 206 → 326.

**Nhóm xe tốt lên hoặc xấu đi trên cùng tập test:**

- Small (66 box): recall 0.182 → 0.000 (12 → 0 xe). Mô hình vòng 1 không còn thấy xe nhỏ ở xa.
- Medium (296 box): recall 0.547 → 0.182 (162 → 54 xe). Nhóm này mất nhiều nhất: 108 trong 120 TP bị
  mất.
- Large (41 box): recall giữ 0.561 (23 → 23 xe). Tổng không đổi, nhưng trên `compare_round1.jpg` các xe
  lớn được bắt không hoàn toàn là cùng những xe (xem ca bên dưới).
- Precision 0.925 → 1.000, FP 0: model vòng 1 rất dè dặt. Chỉ vài box vượt conf 0.25, và các box đó
  đều đúng.

**Một ca đổi sau fine-tune, có lý do kiểm được** (`outputs/compare_round1.jpg`; cột giữa là cold start,
cột phải là vòng 1; xanh lá là TP, vàng là FN, đỏ là FP):

- *Xấu đi:* **xe chiều đi chỉ thấy đèn hậu đỏ ở nửa phải ảnh.** Ở frame_0050, cold start đạt TP 11, FP 2,
  FN 7, trong đó các xe đèn hậu cỡ vừa bên phải là ô xanh lá. Vòng 1 chỉ còn TP 5, FP 0, FN 13, và cả
  dãy xe đèn hậu bên phải thành ô vàng. frame_0150 cũng vậy: TP 10 → 4, FN 10 → 16, cụm xe đèn hậu bên
  phải mất hết. Chính dạng xe này kéo recall medium từ 0.547 xuống 0.182.
- *Tốt lên:* **xe chiều ngược bật đèn pha ở làn sát lề trái.** Ở frame_0050, frame_0250 và frame_0350,
  một xe đèn pha bên trái là ô vàng (FN) ở cold start nhưng thành xanh lá ở vòng 1. Đây
  đúng là dạng xe mình thêm nhiều nhất khi sửa nhãn (ví dụ dòng frame_0369 trong `REVIEW_LOG.csv`, 13
  xe đèn pha), nên model có học được từ nhãn đã sửa.

**Lý do có thể kiểm cho việc giảm** (log ô train trong notebook):

1. Log ghi `Overriding model.yaml nc=80 with nc=1` và `Transferred 322/355 items from pretrained
   weights`. Đầu phân lớp COCO (80 lớp, gồm car/bus/truck mà cold start dùng) bị thay bằng đầu 1 lớp
   khởi tạo mới, nên phần phân lớp mất hiểu biết "đây là xe" của COCO.
2. 12 ảnh với `batch=16` nên mỗi epoch chỉ có **1 bước cập nhật**. 50 epoch là **50 bước** (AdamW lr
   0.002), quá ít để học lại đầu phân lớp.
3. Kiểm trên chính 12 ảnh train cho P 0.881, R 0.348, mAP50 0.782. Đây là số train, **không phải số
   test**, chỉ dùng để chẩn đoán. Model chưa nhớ được cả những ảnh nó đã học (underfit), nên nguyên nhân
   không phải overfit vào 12 ảnh. Cũng không phải nhãn sai: nếu nhãn sai hàng loạt thì FP phải tăng,
   trong khi FP về 0.

Kết luận: AP50 giảm chủ yếu do **công thức fine-tune** (đầu phân lớp mới và chỉ 50 bước), nên kết quả này
chưa đo được giá trị của lô nhãn đã chọn. Mức giảm 0.184 lớn hơn nhiều so với dao động của tập 20 ảnh
test nên là thay đổi thật. Nhưng đó là thay đổi của cả quy trình train, không riêng của dữ liệu.

**Phân biệt ba lớp bằng chứng:**

| Lớp | Nguồn | Nội dung |
| --- | --- | --- |
| Quan sát độc lập (trước khi xem box AI) | `reports/BLIND_SCAN.md`, khóa bằng `blind_lock.json` | frame_0182: khoảng 25 xe; dự đoán hai chỗ AI dễ bỏ sót là xe nhoè góc dưới phải và SUV tối bị cắt mép dưới |
| Lỗi pre-label đã sửa | `reports/REVIEW_LOG.csv`, `outputs/round1_diff.md/.json` | 7 xóa (box gộp hai xe, box trùng), 3 chỉnh (box ăn sang xe khác), 85 thêm (xe tối, xe tải, xe đèn pha). Hai chỗ dự đoán ở frame_0182 đúng là hai trong 10 box thêm ở ảnh đó |
| Chất lượng model sau train | `outputs/metrics_round1.json`, `compare_round1.jpg`, `rounds_table.md` | AP50 0.588 (−0.184); recall medium và small giảm mạnh; xe đèn pha gần camera được bắt tốt hơn |

Lớp thứ hai chỉ nói về **nhãn gợi ý của cold start trên 12 ảnh pool**. Lớp thứ ba nói về **model mới
trên 20 ảnh test**. Pre-label được giữ 94% không có nghĩa model sau train đúng 94%, và nhãn đã sửa tốt
hơn cũng không bảo đảm AP50 tăng.

**Một ca khó theo guideline:**

- frame_0331, box P15 (x 442–558, y 392–451) ôm hai xe đi cạnh nhau như một vật. Guideline yêu cầu hai
  xe sát nhau phải vẽ hai box riêng. P14 và P18 đã ôm đúng từng xe, nên P15 bị xóa chứ không co lại, vì
  co lại sẽ trùng với một trong hai box đó.
- Tương tự, ở frame_0392, box P6 là xe bị xe P9 che một phần. Guideline chỉ vẽ phần nhìn thấy, nên cạnh
  trên của box được thu từ y 326 xuống y 355.
- Với **xe rất nhỏ**, mình giữ box AI nếu nó ôm đúng một xe, nhưng không thêm box mới cao dưới 16 px.
  Trong 247 box có 3 box cao dưới 16 px (frame_0107 hai box, frame_0182 một box), cả ba là pre-label giữ
  nguyên (IoU 1.00 với box gợi ý). Khi chấm, các box này bị bỏ qua theo luật `MIN_BOX_H_PX = 16`.

## 5. Kết luận và giới hạn

**So với cold start:** trên cùng tập test, vòng 1 kém hơn. AP50 0.771 → 0.588, recall 0.489 → 0.191,
F1 0.640 → 0.321. Chỉ precision tăng (0.925 → 1.000), và xe đèn pha gần camera được bắt tốt hơn. Với
12 ảnh nhãn và cách train hiện tại, fine-tune **chưa** tốt hơn mô hình COCO có sẵn.

**Quyết định: dừng gán nhãn vòng 2 theo lô `selection_round2.csv` hiện tại, và chỉ tiếp tục sau khi sửa
cách train.** Lý do:

1. AP50 giảm do công thức train (mục 4), không phải do thiếu dữ liệu. Thêm 12 ảnh mà giữ cách train này
   thì chỉ lên 2 bước mỗi epoch, nhiều khả năng vẫn underfit.
2. Lô vòng 2 do chính model yếu này chọn. Ở conf 0.05, model vòng 1 chỉ thấy trung bình 7.2 box mỗi ảnh
   pool (2–14 box, `selection_round2.csv`), trong khi cold start thấy 27.9 box (14–53,
   `selection_round1.csv`). Notebook báo lô vòng 2 chỉ có **59 box gợi ý cho 12 ảnh** (khoảng 5 box/ảnh), trong khi
   ở vòng 1 mỗi ảnh có khoảng 20.6 xe sau khi sửa (247/12). Một model bỏ sót 80% xe thì độ bất định của
   nó không đáng tin để xếp hạng ảnh. Pre-label sẽ thiếu khoảng 190 xe (12 × 20.6 − 59), trong khi vòng 1
   chỉ phải thêm 85, tức **công gán tăng khoảng 2 lần**.
3. **Ảnh gần trùng:**
   - 4 trong 12 ảnh vòng 2 cách một ảnh đã gán không quá 2.4 s: frame_0229 cách 0227 0.8 s (D 0.08),
     frame_0378 cách 0380 0.8 s (D 0.08), frame_0176 và frame_0193 cách 0182 và 0187 đúng 2.4 s. Ngoài
     ra frame_0307 cách 0312 2.0 s.
   - Xe trong các ảnh này phần lớn là xe đã gán, nên tốn công mà ít thông tin mới.
   - Các ảnh đáng giữ là frame_0005, 0016, 0021 (2–8 s, D 1.0) và frame_0075 (30 s, D 0.96). Chúng thuộc
     đoạn đầu video, nơi chưa có ảnh nào được gán.

Trước khi train tiếp, cần giữ lại đầu phân lớp COCO hoặc tăng số bước cập nhật:

- giữ đầu phân lớp COCO bằng cách map nhãn vào lớp car của COCO, hoặc khởi tạo từ model cold start;
- hoặc tăng số bước (epoch 200 trở lên, `batch=4`).

Sau đó chạy lại vòng 1 **trên đúng 247 box này**. Chỉ khi AP50 không thấp hơn cold start mới dùng model
mới để chọn lô vòng 2.

**Hai ca còn yếu cho vòng sau:**

1. **Xe chiều đi chỉ thấy đèn hậu đỏ, cỡ vừa, ở nửa phải ảnh** (frame_0050 và frame_0150 trên
   `compare_round1.jpg`). Recall medium giảm 0.547 → 0.182.
   - Nên ưu tiên ảnh có dòng xe đi đông ở làn phải, trong đoạn thời gian chưa gán (0–30 s).
   - Chi phí: mỗi ảnh khoảng 20 xe, nhiều xe sát nhau nên phải tách box (cùng loại lỗi P15 ở frame_0331).
   - Không chọn thêm ảnh trong cụm 124–131 s, vì đã có 0312, 0326 và 0331.
2. **Xe lớn gần camera bật đèn pha hoặc bị cắt ở mép dưới.** Recall large đứng yên ở 0.561 qua cả hai
   vòng.
   - Ở frame_0250, xe lớn góc dưới trái là FN ở cả cold start lẫn vòng 1, còn xe lớn ở giữa mép dưới
     chuyển từ TP thành FN. Ở frame_0050, xe bị cắt ở mép dưới cũng chuyển từ TP thành FN.
   - Nhãn vòng 1 chỉ có vài box dạng này (SUV ở frame_0182, hai xe tối ở frame_0392).
   - Chi phí thấp vì xe lớn, rõ và ít box mỗi ảnh. Nhưng các frame liên tiếp có cùng một xe lớn đi qua,
     nên phải giữ `MIN_GAP_S` từ 2 s trở lên để khỏi gán cùng một xe ba lần.

Xe nhỏ ở xa (recall small 0.182 → 0) để sau. Phần lớn chúng cao sát 16 px, là nơi nhãn tham chiếu kém tin
cậy nhất và vẽ tốn công nhất.

**Giới hạn ảnh hưởng đến kết luận:**

- **Tập test chỉ có 20 ảnh từ 4 đoạn 8 giây**, thực chất chỉ khoảng 4 cảnh độc lập. Chỉ cần một ảnh đổi
  vài box là AP50 có thể đổi khoảng 0.01, nên mọi chênh lệch dưới khoảng 0.01–0.02 không có ý nghĩa. Mức
  −0.184 vượt xa ngưỡng đó. Tuy vậy, số theo nhóm kích thước (41 box large, 66 box small) rất nhiễu: 23/41
  ở cả hai vòng không có nghĩa là cùng 23 xe.
- **Luật bỏ qua xe nhỏ:** 14 box cao dưới 16 px không được chấm, nên với xe rất xa, cả mô hình lẫn nhãn
  đều không được thưởng hay bị phạt. Kết luận chỉ áp dụng cho xe cao từ 16 px trở lên.
- **Nhãn tham chiếu do model tạo, chưa được rà tay:**
  - Tham chiếu có thể mang cùng thiên lệch với cold start (cùng nhìn xe theo kiểu COCO), nên cold start có
    lợi thế khi so.
  - Model vòng 1 học theo guideline (box ôm thân xe, không ôm quầng đèn pha, tách xe sát nhau). Vì vậy nó
    có thể bị tính FN hoặc FP do IoU dưới 0.5 dù vẽ đúng.
  - Ở vòng này FP = 0 nên thiên lệch đó không giải thích được mức giảm. Nhưng nó hạn chế mọi kết luận
    kiểu "tốt hơn một chút".
- Chỉ có một seed, một lần train và không có baseline `random`, nên chưa tách được hiệu quả của cách chọn
  mẫu khỏi hiệu quả của công thức train.

**Nếu AP50 giảm, kiểm tra các bước sau trước khi train thêm** (đúng thứ tự mình đã làm ở vòng này):

1. Dữ liệu:
   - `n_train_images` và `n_train_boxes` trong `metrics_round1.json` khớp với `round1_diff.json` (12 và
     247, đã khớp);
   - không có ảnh test nào trong `labels/round1/`;
   - nhãn test không bị đổi (hash khớp `ref_hashes`);
   - nhãn đúng lớp 0 và tọa độ nằm trong [0, 1] (`pack_labels` đã kiểm).
2. Nhãn: xem lại box vẽ trên ảnh train. Nếu nhãn sai thì FP thường tăng; ở vòng này FP về 0 nên nguyên
   nhân không phải nhãn sai.
3. Train: log có thay đầu phân lớp không, có bao nhiêu bước cập nhật, model có học được chính ảnh train
   không (ở đây R 0.348, tức là underfit).
4. Đánh giá: xem `compare_round1.jpg` để biết số giảm ở nhóm xe nào. Nếu FN dồn vào một dạng xe mà tham
   chiếu có vẻ sai, mở ảnh gốc rà lại tham chiếu trước khi kết luận.
5. Sau các bước trên mới đổi cách train, rồi so lại trên đúng 20 ảnh test.

**Tự QC:**

- Trước khi nộp đã chạy `python tools/check_submission.py` và `pytest`, sau đó clone lại repo từ GitHub để
  kiểm bản nộp, gồm hash nhãn test và `blind_lock.json`.
- `BLIND_SCAN.md` được viết và khóa trước khi mở pre-label. Hai vị trí rủi ro trong đó khớp hai box thêm
  ở frame_0182 (`REVIEW_LOG.csv`, dòng 1–2).
- Mỗi dòng trong `REVIEW_LOG.csv` khớp với số trong `round1_diff.json`. Ví dụ: frame_0331 có 2 deleted;
  frame_0227, 0326 và 0392 mỗi ảnh có 1 edited; frame_0099 có 13 accepted.
- Sau khi sửa trên CVAT, đã xem lại 12 ảnh bằng ảnh ghép box để tìm xe rõ còn sót và box gộp. Không có
  box mới nào cao dưới 16 px.
- Mọi số trong báo cáo lấy từ `outputs/` và `reports/rounds_table.md`. Số đo trên ảnh train chỉ dùng để
  chẩn đoán và đã ghi rõ là số train.
