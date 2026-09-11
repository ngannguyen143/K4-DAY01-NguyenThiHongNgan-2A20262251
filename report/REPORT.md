# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:11/09/2026**

**Runtime Colab: GPU** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`): 468, cab, 1, 0.510915, ImageNet-1K

- Record này mô tả toàn ảnh như thế nào?
Record này cho biết mô hình YOLO11n-cls dự đoán toàn bộ ảnh traffic thuộc lớp cab với thứ hạng 1 và model score là 0.510915.

class_id = 468: ID của lớp được dự đoán.
class_name = cab: tên lớp tương ứng với ID 468.
rank = 1: đây là dự đoán có score cao nhất trong Top-5.
score = 0.510915: mức độ tin cậy của mô hình đối với dự đoán này.
taxonomy_name = ImageNet-1K: lớp cab thuộc hệ thống nhãn ImageNet-1K mà checkpoint classification sử dụng.

- Ai định nghĩa class list mà checkpoint có thể dự đoán?
Dataset/taxonomy dùng để huấn luyện checkpoint → xác định danh sách class → checkpoint chỉ có thể dự đoán trong danh sách class đó.

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
Cần giữ cả ba để xác định lớp một cách rõ ràng và tránh nhầm lẫn.
    class_id: ID số, thuận tiện cho máy xử lý và lưu trữ.
    class_name: tên lớp, giúp con người đọc và hiểu prediction.
    taxonomy_name: cho biết ID và tên lớp đang thuộc hệ thống nhãn nào.

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
Vì classification chỉ trả về một lớp cho toàn bộ ảnh, guideline cần quy định cách chọn lớp đại diện cho toàn ảnh khi có nhiều chủ thể.
Nếu ảnh có nhiều chủ thể, người gán nhãn phải xác định một lớp đại diện theo tiêu chí đã thống nhất, ví dụ chủ thể chính/nổi bật nhất trong ảnh.
Quan trọng nhất là tất cả người gán nhãn phải áp dụng cùng một quy tắc, tránh trường hợp cùng một ảnh nhưng người này chọn cab, người khác lại chọn một lớp khác.

- Vì sao model score không phải ground truth?
Vì model score chỉ là kết quả dự đoán của mô hình, còn ground truth là nhãn chuẩn do con người xác định theo guideline.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
   
    "class_name": "person",
    "score": 0.912625,
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68
- Diễn giải vị trí box bằng lời:
    + tọa độ góc trên bên trái: xmin = 385.33, ymin = 69.24
    + tọa độ góc dưới bên phải: xmax = 498.92, ymin = 348.92

- So sánh số prediction ở hai threshold:
Ở ngưỡng cao (conf = 0.50 trở lên): Số lượng prediction ít (thường chỉ 3 – 5 vật thể lớn, rõ ràng và có độ tin cậy cao như person [score 0.91], oven, refrigerator). Hầu như không có box rác hoặc nhãn ảo.

Ở ngưỡng chuẩn/thấp (conf = 0.35 hoặc 0.20): Số lượng prediction tăng lên rõ rệt (có thể lên tới 8 – 14 vật thể). Mô hình nhận diện thêm các vật dụng nhỏ hoặc khuất hơn trên bàn bếp (như bottle, cup, bowl, sink), nhưng bắt đầu xuất hiện những nhãn có score thấp, dự đoán nhầm hoặc box chồng lấn nhẹ do nhiễu môi trường.

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
Độ bao phủ (Recall/Coverage):
    + Khi hạ threshold: Độ bao phủ tăng lên, giảm nguy cơ bỏ sót các vật thể nhỏ (chén, đĩa, chai lọ) hoặc các vật thể bị che khuất một phần trong căn bếp.
    + Khi tăng threshold: Độ bao phủ giảm, mô hình chỉ giữ lại các vật thể chắc chắn, dẫn đến nguy cơ bỏ sót cao (False Negatives).

Khối lượng việc của Reviewer (QA/Kiểm duyệt):
    + Hạ threshold: Khối lượng công việc tăng mạnh. Reviewer phải tốn thời gian kiểm tra từng box score thấp, xóa bỏ các box dự đoán sai (False Positives) hoặc tinh chỉnh các box nhiễu.
    + Tăng threshold: Reviewer duyệt nhanh hơn ở các nhãn có sẵn vì độ chính xác (Precision) cao, nhưng lại tốn công vẽ thủ công (manual labeling) thêm các vật thể bị bỏ sót nếu dự án yêu cầu gán nhãn triệt để toàn bộ đồ vật trong bếp.

- Đề xuất một quy tắc box chặt:
Nội dung quy tắc: "Bounding box phải là hình chữ nhật nhỏ nhất bao trọn toàn bộ các điểm ảnh nhìn thấy được của đối tượng."

Tiêu chí kiểm định:
    + 4 cạnh của box phải tiếp xúc sát nhất có thể với 4 điểm biên cực trị (trên, dưới, trái, phải) của người/vật thể.
    + Không để lọt viền nền trống vượt quá 5% diện tích của bounding box.
    + Không cắt lẹm vào các phần cơ thể hoặc chi tiết của đối tượng (trừ phần bóng đổ không gắn liền với cấu trúc vật thể).

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
Ngưỡng nhìn thấy: Cần quy định rõ tỷ lệ che khuất. Ví dụ: Nếu vật thể bị che lấp trên 70% hoặc 80% diện tích thì bỏ qua không gán nhãn để tránh tạo dữ liệu quá mơ hồ cho mô hình học.

Đối tượng bị chia cắt làm hai phần rời rạc: Cần guideline quy định rõ: Vẽ 1 box bao trùm cả phần bị che chắn (chấp nhận có khoảng trống ở giữa) hay vẽ thành 2 box riêng lẻ (hoặc chuyển sang định dạng phân vùng đa giác - Polygon/Segmentation).

Trường hợp cần Escalation (Báo cáo xin ý kiến Lead/Client):

Các cụm đối tượng xếp chồng lên nhau quá dày đặc (ví dụ: giàn dao cắm chung trong ống, chồng chén đĩa xếp khít nhau) mà guideline chưa nêu rõ gán nhãn từng chiếc (individual instances) hay gán nhãn cả cụm (group/cluster).

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):

    "instance_id": "kitchen-010",
    "class_name": "bowl",
    "score": 0.372406,
    "bbox_xyxy": [
      154.46,
      169.34,
      182.99,
      184.98
    ],
    "polygon_xy": 
      [
        155.0,
        170.0
      ],
      [
        155.0,
        177.0
      ],
      [
        156.0,
        178.0
      ],
      [
        156.0,
        179.0
      ],
      [
        161.0,
        184.0
      ],
- Polygon bổ sung chi tiết gì so với box?
Mô tả đường viền thực tế (True pixel boundary): Bounding box chỉ là một hình chữ nhật bao quanh vùng cực trị, nên luôn chứa nhiều điểm ảnh nền (background noise) thừa ở 4 góc đối với các vật thể tròn, elip hoặc méo mó. Ngược lại, đa giác (polygon_xy) bám sát mép cong vật lý thực tế của miệng và đáy chiếc bát (bowl).

Diện tích và hình dạng chuẩn xác: Cho biết hình học chính xác (orientation, độ lõm/lồi, góc nghiêng) và số lượng pixel thực sự thuộc về đối tượng, phục vụ các tác vụ đo đạc kích thước, robot gắp vật thể hoặc tính toán tỷ lệ che khuất mà bounding box không thể thể hiện được.

- `instance_id` dùng để làm gì và không phải loại ID nào?
Mô tả đường viền thực tế (True pixel boundary): Bounding box chỉ là một hình chữ nhật bao quanh vùng cực trị, nên luôn chứa nhiều điểm ảnh nền (background noise) thừa ở 4 góc đối với các vật thể tròn, elip hoặc méo mó. Ngược lại, đa giác (polygon_xy) bám sát mép cong vật lý thực tế của miệng và đáy chiếc bát (bowl).

Diện tích và hình dạng chuẩn xác: Cho biết hình học chính xác (orientation, độ lõm/lồi, góc nghiêng) và số lượng pixel thực sự thuộc về đối tượng, phục vụ các tác vụ đo đạc kích thước, robot gắp vật thể hoặc tính toán tỷ lệ che khuất mà bounding box không thể thể hiện được.

- Đề xuất một quy tắc biên mask:
Nội dung quy tắc: "Đường biên đa giác (polygon mask) phải bám sát ranh giới pixel nhìn thấy được của vật thể, sai số cho phép không vượt quá 1 – 2 pixel so với viền thực tế."

Tiêu chuẩn kiểm duyệt (QA checklist):
    + Không để sót chi tiết lồi/lõm đặc trưng của vật thể (không dùng quá ít điểm làm biến dạng hình tròn của bát thành hình vuông thô).
    + Không bao lấn sang vùng nền (over-segmentation) hoặc cắt lẹm vào thành bát (under-segmentation).
    + Mật độ điểm phải tối ưu: đủ để thể hiện độ cong mượt mà, nhưng không cắm điểm quá dày đặc trên một đoạn viền thẳng.

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
Vùng biên mờ do chuyển động hoặc mất nét (Motion blur / Out-of-focus):

Guideline quy định: Thống nhất nguyên tắc vẽ viền theo đường ranh giới trung bình (50% gradient độ tương phản) hay vẽ bọc lấy mép ngoài cùng của vùng mờ mờ.

Vật thể tiếp xúc/dính sát nhau (Touching objects):
Guideline quy định: Khi hai chiếc bát/đĩa xếp chồng hoặc để sát nhau, ranh giới giữa chúng phải vẽ tiếp xúc tuyệt đối (shared boundary) hay cho phép chồng lấn 1 pixel? Tuyệt đối không để xảy ra khe hở rỗng vô lý giữa hai vật đang chạm nhau.

Vật thể bị che khuất một phần (Occluded parts):
    Nguyên tắc Segmentation: Chỉ vẽ mask cho phần nhìn thấy được (visible mask), không vẽ đa giác phỏng đoán vùng bị che lấp bên dưới (trừ phi dự án yêu cầu Amodal Segmentation).
    Bị chia cắt đôi (chia thành nhiều mảnh rời rạc): Quy định rõ ràng liệu đối tượng này được biểu diễn bằng Multi-polygon (1 instance_id có nhiều cụm polygon) hay tách thành 2 nhãn riêng.

Trường hợp cần Escalation (Báo cáo xin chỉ đạo):
Thức ăn hoặc vật dụng bên trong bát: Khi bát chứa đầy thức ăn/chất lỏng lồi lên trên mép, guideline cần làm rõ nhãn bowl tính riêng phần vỏ sứ/nhựa hay gộp chung toàn bộ phần thức ăn bên trong vào làm một thể thống nhất.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Đơn vị/định dạng ground truth | Lỗi hoặc điểm mơ hồ quan sát được | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |  |  |  |  |
| Phát hiện vật thể |  |  |  |  |
| Instance segmentation |  |  |  |  |

### 1. Tác vụ Phân loại ảnh (Image Classification)

- **Đơn vị/định dạng ground truth:**
  - Nhãn cấp toàn bộ bức ảnh (Image-level label).
  - Lưu trữ dưới dạng mã số lớp (`class_id`) hoặc tên lớp (`class_name`) tương ứng với thư mục chứa ảnh hoặc metadata định dạng CSV/JSON/TXT.

- **Lỗi hoặc điểm mơ hồ quan sát được:**
  - Bức ảnh có nhiều đối tượng cùng xuất hiện (ví dụ: bối cảnh bếp vừa có người, vừa có lò nướng, tủ lạnh).
  - Sự mơ hồ giữa việc gắn nhãn theo chủ thể chính nổi bật hay gắn nhãn theo toàn bộ bối cảnh xung quanh.
  - Mô hình dự đoán sai lệch do học đặc trưng nền (background bias) thay vì tập trung vào thực thể trọng tâm.

- **Annotator làm gì?**
  - Lựa chọn duy nhất một nhãn đại diện cho chủ thể chính nổi trội nhất dựa trên quy tắc kích thước hoặc độ bao phủ quy định trong guideline.
  - Bám sát danh mục nhãn chuẩn (taxonomy) của dự án.
  - Đánh dấu cờ phân vân (Uncertain/Ambiguous) hoặc báo cáo cấp trên (escalate) khi chủ thể trong ảnh không rõ ràng.

- **Reviewer xem gì?**
  - Kiểm tra tính nhất quán giữa nhãn được gán với định nghĩa trong tài liệu guideline.
  - Đảm bảo tính chính xác giữa bài toán đơn nhãn (single-label) và đa nhãn (multi-label).
  - Đo lường và kiểm soát tỷ lệ gán nhãn sai loại (misclassification rate).

---

### 2. Tác vụ Phát hiện vật thể (Object Detection)

- **Đơn vị/định dạng ground truth:**
  - Hộp bao hình chữ nhật quanh từng vật thể (Bounding box).
  - Lưu trữ theo định dạng nhãn kèm tọa độ pixel dạng `xyxy` `[xmin, ymin, xmax, ymax]` hoặc định dạng chuẩn hóa YOLO `[class_id, x_center, y_center, width, height]`.

- **Lỗi hoặc điểm mơ hồ quan sát được:**
  - Bỏ sót các vật thể kích thước nhỏ hoặc bị khuất một phần (False Negatives).
  - Nhận diện nhầm các chi tiết nền thành đối tượng (False Positives).
  - Hộp bao vẽ quá rộng (nhiều diện tích thừa) hoặc cắt lẹm vào chi tiết vật thể.
  - Xuất hiện các hộp bao trùng lặp/chồng lấn khi các đối tượng đứng quá sát nhau.

- **Annotator làm gì?**
  - Vẽ hộp bao sát nhất có thể ôm trọn toàn bộ phần nhìn thấy của đối tượng (tight bounding box).
  - Gán nhãn chính xác cho từng hộp bao độc lập.
  - Kẹp sát cạnh hộp vào mép viền ảnh đối với vật thể bị cắt mép và gắn nhãn cờ thuộc tính phụ (`truncated`, `occluded`) theo yêu cầu.

- **Reviewer xem gì?**
  - Đo lường độ trùng khớp ranh giới (chỉ số IoU - Intersection over Union) theo tiêu chuẩn chất lượng.
  - Rà soát lỗi sót nhãn (undercounting) và lỗi gán thừa nhãn ảo (overcounting).
  - Kiểm tra độ chính xác của nhãn gán và loại bỏ các hộp bao vẽ quá lỏng (viền nền thừa vượt 5%).

---

### 3. Tác vụ Phân vùng cá thể (Instance Segmentation)

- **Đơn vị/định dạng ground truth:**
  - Mặt nạ điểm ảnh hoặc tọa độ đa giác khép kín cho từng cá thể riêng biệt.
  - Lưu trữ dưới dạng chuỗi tọa độ đa giác `polygon_xy` `[[x1, y1], [x2, y2], ...]`, mã hóa RLE (Run-Length Encoding) hoặc mặt nạ nhị phân (binary mask) đi kèm với mã định danh `instance_id`.

- **Lỗi hoặc điểm mơ hồ quan sát được:**
  - Viền mặt nạ dự đoán bị thô, làm méo mó các chi tiết cong hoặc lồi/lõm tự nhiên của đối tượng.
  - Viền mặt nạ bị dính liền vào nhau giữa hai cá thể cùng loại đặt sát cạnh.
  - Mặt nạ vẽ xuyên qua phần vật cản nằm phía trước (lỗi over-segmentation).
  - Điểm ranh giới mơ hồ tại những vùng viền bị nhòe nét, bóng đổ hoặc chuyển động.

- **Annotator làm gì?**
  - Đặt các điểm nút đa giác men sát theo đường bao thực tế của vật thể với sai số cho phép từ 1 đến 2 pixel.
  - Phân tách rõ ràng ranh giới tiếp xúc khi nhiều cá thể cùng loại đứng chạm nhau.
  - Chỉ tạo mặt nạ cho phần nhìn thấy được (visible pixels), không tự ý phỏng đoán phần bị che khuất bên dưới.

- **Reviewer xem gì?**
  - Kiểm tra độ chuẩn xác của đường viền đa giác (không bị cắt lẹm hay tràn viền ra ngoài nền).
  - Đảm bảo mỗi cá thể đều có `instance_id` độc lập và được phân loại đúng nhãn.
  - Rà soát các vùng trống, lỗ rỗng bên trong cấu trúc của vật thể xem đã được đục lỗ (khoét rỗng) đúng kỹ thuật hay chưa.

## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:

- **Một quy tắc bảo vệ dữ liệu:**
  - Tuyệt đối không tự ý tải lên (upload), sao chép, trích xuất hoặc chia sẻ hình ảnh và nhãn dữ liệu của dự án ra các nền tảng công cộng hoặc công cụ trực tuyến của bên thứ ba chưa được cấp phép (chỉ thao tác trên đúng môi trường lab/hạ tầng nội bộ được phân quyền và tuân thủ nguyên tắc ẩn danh thông tin nhận dạng cá nhân - PII).

- **Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:**
  - Dừng thao tác ngay lập tức và báo cáo trực tiếp cho **Mentor phụ trách / Giảng viên hướng dẫn** (hoặc Quản trị viên dự án / Data Lead)[cite: 1] để được xác minh và xử lý theo đúng quy trình leo thang (escalation).

## 6. Danh sách bằng chứng

- [x ] `classification_predictions.json`
- [x ] `detection_predictions.json`
- [x ] `segmentation_predictions.json`
- [x ] `IMAGE_ATTRIBUTION.md`
- [x ] `visuals/classification_top5.png`
- [x ] `visuals/detection_predictions.png`
- [x ] `visuals/segmentation_prediction.png`
- [x ] Ô validation cuối notebook báo `PASS`.
- [x ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
