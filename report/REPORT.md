# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6/11/2026

**Runtime Colab:** GPU (`Device: cuda`)

**Python / PyTorch / Ultralytics:** Python 3.13.16 / PyTorch 2.11.0+cu130 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt` (cả ba đều `PASS` kiểm tra SHA-256)

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (không đổi)

**Thay đổi so với notebook nguồn:** Không đổi checkpoint, threshold hay mã tạo prediction. Chỉ thêm một ô `drive.mount('/content/drive')` để lưu file lên Google Drive; ô này không ảnh hưởng tới kết quả JSON/PNG.

## 1. Phân loại ảnh

Nguồn: `classification_predictions.json`, `sample_id = traffic`, `rank = 1`.

1. **Record hạng 1:** `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
   (Top-5 còn lại: minibus 0.164, police_van 0.086, recreational_vehicle 0.054, streetcar 0.048.)

2. Record này nói về **cả bức ảnh**. Record không có `bbox_xyxy` hay polygon, chỉ có một danh sách top-5 cho mỗi `sample_id`, và tổng score của các lớp là phân phối xác suất trên toàn ảnh. Ảnh `traffic` có nhiều xe buýt và ô tô, nhưng model chỉ trả về **một nhãn cho cả cảnh**, không cho biết "cab" nằm ở đâu.

3. Danh sách lớp **đi kèm taxonomy/checkpoint**: `yolo11n-cls.pt` được huấn luyện trên ImageNet-1K nên chỉ chọn được trong 1000 lớp của ImageNet. Model không tự nghĩ ra tên lớp mới. Vì vậy với ảnh đường phố COCO, model ra "cab"/"streetcar": đó là những lớp gần nhất mà taxonomy ImageNet có, không phải nhãn "traffic scene".

4. Cùng một con số `class_id` mang nghĩa khác nhau ở mỗi taxonomy: `class_id=2` trong COCO-80 là `car`, còn trong ImageNet-1K lại là `great_white_shark`. Một chiếc xe taxi được gọi là `car` (id 2) ở COCO nhưng là `cab` (id 468) ở ImageNet. Tên lớp cũng chia khác nhau, vd. COCO chỉ có `bus`, còn ImageNet tách thành `minibus`, `trolleybus`, `school_bus`. Ghi đủ ba trường thì:
   - `class_id` cho máy đọc chính xác, không phụ thuộc chính tả/ngôn ngữ của tên;
   - `class_name` cho người đọc kiểm tra nhanh;
   - `taxonomy_name` cho biết id/tên đó thuộc bộ lớp nào, để không gộp nhầm dữ liệu từ hai taxonomy và để truy vết khi đổi phiên bản.

5. Guideline cần nói rõ:
   - **Tiêu chí chọn lớp chính**: vd. "chọn vật/chủ đề chiếm diện tích lớn nhất hoặc là chủ thể trung tâm của ảnh"; hoặc theo mục đích dự án (vd. ưu tiên phương tiện giao thông).
   - **Thứ tự ưu tiên** khi có nhiều ứng viên ngang nhau (vd. bus > car > cab).
   - **Ví dụ đúng/sai** cho cảnh nhiều vật (như ảnh `traffic`: nhiều xe buýt + ô tô).
   - **Khi nào escalation** (vd. không có lớp nào trong taxonomy phù hợp, hoặc hai chủ thể ngang nhau) thay vì tự đoán.

6. `score` chỉ là mức "tin" của model, không phải sự thật. Ở đây top-1 "cab" chỉ có 0.51, tức model gần như phân vân một nửa; và nhãn đúng theo nghĩa thông thường của cảnh có thể là "bus"/"streetcar" chứ không phải "cab". Model có thể **tự tin mà vẫn sai**, và score cao không có nghĩa nhãn đúng theo guideline. Nếu lấy prediction làm ground truth thì lỗi của model sẽ quay lại làm dữ liệu huấn luyện. Ground truth phải do người gán theo guideline và có reviewer kiểm tra.

## 2. Phát hiện vật thể

Nguồn: `detection_predictions.json` + `visuals/detection_predictions.png`, `sample_id = kitchen` (ảnh 640×427).

1. **Record được chọn** (người đứng nấu bên phải ảnh):
   `class_name`=person (`class_id`=0) ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

2. **Giải thích box bằng lời** (`[x_min, y_min, x_max, y_max]`, đơn vị pixel, gốc (0,0) ở góc trên-trái của ảnh 640×427):
   - Góc trên-trái ở điểm **(385.33, 69.24)**: cách mép trái ảnh ~385 px (khoảng 60% bề ngang) và cách mép trên ~69 px, tức ngay trên đỉnh đầu người, phía khuỷu tay trái.
   - Góc dưới-phải ở điểm **(498.92, 348.92)**: dưới gót chân, sát mép ngoài khuỷu tay phải; cách đáy ảnh còn ~78 px.
   - Hộp **rộng 113.58 px** (= 498.92 − 385.33) và **cao 279.68 px** (= 348.92 − 69.24). Hộp cao gần gấp 2.5 lần bề rộng, đúng dáng một người đứng thẳng, và chiếm khoảng 65% chiều cao ảnh.

3. **So sánh ngưỡng** (output ô so sánh trong notebook):
   - `0.20` → **17** prediction: person, bowl, bowl, oven, oven, person, bowl, bowl, cup, cup, bowl, spoon, potted plant, spoon, dining table, spoon, bottle
   - `0.35` → 11 prediction (ngưỡng dùng cho JSON)
   - `0.60` → **6** prediction: person, bowl, bowl, oven, oven, person

4. Khi hạ ngưỡng 0.60 → 0.20, số hộp tăng từ 6 lên 17. Reviewer phải:
   - kiểm tra từng hộp mới có phải vật thật không (false positive), vd. `dining table` cho bàn làm bếp, `potted plant` cho bó lá thơm treo khô, 3 `spoon` nhỏ khó nhìn;
   - phân biệt **hộp trùng** với **hai vật sát nhau**. Ví dụ: trên PNG ở ngưỡng 0.35, nhãn hai hộp `cup` in đè lên nhau thành "cucup 0.45", nhìn như trùng hộp. Nhưng JSON cho thấy đó là hai cốc riêng: `cup 0.451108` ở [143.52, 268.13, 172.93, 301.81] và `cup 0.381542` ở [119.95, 272.2, 141.78, 304.16], hai hộp không giao nhau. Ngược lại, `bowl 0.381215` [0.32, 305.02, 87.69, 371.32] và `bowl 0.719063` [32.65, 342.12, 100.16, 384.93] **có giao nhau** (IoU ≈ 0.23; phần giao chiếm ~56% hộp nhỏ hơn). Nhìn ảnh thì đây là hai đồ vật khác nhau (khay bánh phía sau và đĩa sứ trắng phía trước), nên reviewer cần mở ảnh xác nhận chứ không gộp hai hộp theo mức chồng lấn;
   - kiểm tra lại **lớp** có đúng taxonomy/guideline không (vd. hai vật nhỏ trên kệ `bowl 0.701282` [155.02, 168.67, 182.38, 184.18] và `bowl 0.464904` [155.76, 113.58, 173.98, 130.13] trông giống cốc/hũ hơn là bát);
   - kiểm tra độ khít của box. Tóm lại: nhiều hộp hơn thì ít bỏ sót hơn nhưng tốn thời gian review hơn và nhiều nhiễu hơn.

5. **Quy tắc box:**
   > Vẽ **một** hộp chữ nhật cho **mỗi** vật thể thuộc taxonomy, bao khít **phần nhìn thấy được** của vật (cạnh hộp chạm mép ngoài cùng của pixel thuộc vật, không chừa nền thừa quá vài pixel). Không vẽ hai hộp cho cùng một vật. Không ước đoán phần bị che khuất nằm ngoài khung hoặc sau vật khác.

6. **Vật bị cắt mép:** ở mép trái ảnh, giữa bàn, chỉ thấy **một cánh tay/bàn tay** của người thứ hai. Model gán `person 0.610931` với `bbox_xyxy`=[0.12, 263.18, 61.55, 310.78]: `x_min` ≈ 0 nghĩa là hộp chạm mép trái ảnh, và hộp chỉ rộng 61.44 × cao 47.59 px, chỉ bao cánh tay. Mình sẽ **viết thành quy tắc trong guideline**: tình huống "người bị cắt mép ảnh, chỉ thấy một phần cơ thể" lặp lại rất thường xuyên trong nhiều ảnh, nên cần một quy tắc thống nhất cho mọi annotator, vd. "vẫn gán `person` nếu nhận ra được là bộ phận người; box chỉ bao phần nhìn thấy". Chỉ **escalation** cho Lab Coach nếu phần nhìn thấy quá nhỏ/mờ đến mức không chắc là người, hoặc dự án có yêu cầu riêng (vd. bỏ qua người bị cắt quá X%) mà guideline chưa nói.

## 3. Phân đoạn theo từng đối tượng

Nguồn: `segmentation_predictions.json` + `visuals/segmentation_prediction.png`, `sample_id = kitchen`.

1. **Record được chọn** (người đứng bên phải):
   `instance_id`=kitchen-001 ; `class_name`=person (`class_id`=0) ; `score`=0.899318 ; `polygon_point_count`=348 ; 5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]
   (`bbox_xyxy` của mask này = [385.45, 66.44, 498.02, 348.58].)

2. **Polygon so với box:** box chỉ cho biết vật nằm trong hình chữ nhật [385.45, 66.44, 498.02, 348.58]. Polygon dùng **348 điểm** để vẽ **đường viền thật** của người. Ví dụ 5 điểm đầu (446,70) → (442,72) đi dần sang trái và xuống dưới, tức là đang men theo vòm đỉnh đầu, chứ không phải một cạnh thẳng như box. Nhờ đó polygon tách được khoảng trống giữa hai chân, giữa tay và thân, vốn vẫn nằm trong box. Polygon cũng cho biết diện tích thật và phần bị che. Ví dụ mask `dining table` (`kitchen-005`) có tới 598 điểm, vì biên bàn dài và bị nhiều đồ vật đặt trên bàn cắt ngang. Trong khi đó, bát nhỏ trên kệ `kitchen-010` chỉ cần 12 điểm.

3. **`instance_id`** là mã định danh **riêng cho từng vật trong một ảnh** của output lab, dùng để phân biệt hai vật **cùng lớp**. Ví dụ trong JSON: `kitchen-001` (person, 0.899318) và `kitchen-009` (person, 0.431438, cánh tay ở mép trái) có cùng `class_id`=0 nhưng `instance_id` và polygon khác nhau. Tương tự, 4 cái bát `kitchen-002/003/006/010` đều có `class_id`=45.
   - **Không phải class ID**: class ID cho biết vật *là gì* (0 = person, 45 = bowl); instance_id cho biết *cái nào*.
   - **Không phải tracking ID**: số chỉ được đánh theo thứ tự output trong một lần chạy trên một ảnh (ở đây theo score giảm dần: 001 = 0.899, 002 = 0.736, …). Chạy lại, đổi ngưỡng hay sang khung hình khác thì cùng một người có thể nhận số khác. Vì vậy không dùng instance_id để theo dõi một vật qua nhiều ảnh.

4. **Quy tắc vẽ biên mask:**
   > Đường biên polygon đi theo **mép ngoài cùng của phần nhìn thấy** của vật, mỗi instance một polygon riêng. Không tô lẹm sang nền hoặc vật bên cạnh. Lỗ hở nhìn xuyên được (giữa hai chân, giữa tay và thân) **không** thuộc mask. Phần bị vật khác che **không** vẽ; vật đứng trước giữ nguyên mask, vật phía sau chỉ lấy phần lộ ra. Hai mask không được chồng pixel lên nhau.

5. **Chỗ khó:** trên mặt bàn bên trái, **hai bát chạm/chồng lên nhau**: `kitchen-006` (bowl 0.528201, box [0.11, 302.85, 85.7, 370.15]) và `kitchen-002` (bowl 0.735744, box [31.44, 343.77, 100.89, 385.55]). Hai box giao nhau trong vùng x ≈ 31–86, y ≈ 344–370. Cả hai lại nằm trên mask `kitchen-005` (dining table 0.578416). Mask `kitchen-007` (oven 0.489667) cũng bị loang, có mảng tối lẹm ra ngoài lò. Với hai vật chạm nhau, mình dùng **guideline**: đây là tình huống lặp lại, có thể giải bằng quy tắc ở câu 4 (vật trước giữ mask, vật sau lấy phần lộ ra, không chồng pixel). Riêng việc bàn làm bếp bị gán `dining table` (COCO-80 không có lớp "bàn bếp") là **câu hỏi về taxonomy** nên mình **escalation** cho Lab Coach, không tự quyết định gán `dining table` hay bỏ qua.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | **1 nhãn cho cả ảnh**: `class_id` + `class_name` + `taxonomy_name` (vd. `468 / cab / ImageNet-1K`) | Ảnh `traffic` có nhiều bus + car nhưng top-1 là `cab` chỉ với score 0.51; taxonomy ImageNet không có lớp "cảnh giao thông" | Xem cả ảnh, chọn **một** lớp theo tiêu chí chủ thể chính trong guideline; escalation nếu không lớp nào hợp | Lớp chọn có đúng guideline/taxonomy không; các annotator có chọn thống nhất trên ảnh nhiều chủ thể không |
| Phát hiện vật thể | **1 box/vật**: `class_id`, `class_name`, `bbox_xyxy` (pixel, gốc trên-trái) | `kitchen`: nhãn hai `cup` in đè thành "cucup" dù JSON cho thấy là 2 hộp không giao nhau; `bowl 0.38` và `bowl 0.72` chồng nhau; vật trên kệ gán `bowl 0.46/0.70` có thể là cốc; nồi chảo treo không có lớp COCO nên bị bỏ; `person 0.61` (x_min=0.12) chỉ là cánh tay bị cắt mép; số hộp 17/11/6 tùy ngưỡng | Vẽ box khít phần nhìn thấy cho **mọi** vật thuộc taxonomy (không phụ thuộc ngưỡng), chọn đúng lớp, không vẽ trùng | Box thiếu (vật bị bỏ sót), box thừa/trùng, sai lớp, box lỏng/chật; vật bị che/cắt mép có làm đúng quy tắc không |
| Instance segmentation | **1 polygon/instance**: `instance_id`, `class_id`, `class_name`, `polygon_xy` (pixel) | `kitchen`: hai bát `kitchen-006` và `kitchen-002` chạm nhau; mask `oven` (`kitchen-007`, 0.49) loang ra ngoài; bàn bếp bị gán `dining table` (`kitchen-005`, 0.58); `person` `kitchen-009` (0.43) chỉ là cánh tay; model seg ra 11 instance, khác tập lớp của model detection ở cùng ngưỡng 0.35 (có thêm spoon, potted plant, dining table; không có cup) | Vẽ polygon theo biên phần nhìn thấy, tách riêng từng instance cùng lớp, không chồng pixel | Biên có bám sát vật không; có lẹm sang nền/vật khác không; mỗi vật có đúng một instance; các ca biên mờ/che khuất có đúng guideline/escalation không |

## 5. An toàn dữ liệu

1. Chỉ dùng ba ảnh COCO công khai mà notebook đã cố định (kiểm tra bằng SHA-256, ghi nguồn/giấy phép trong `IMAGE_ATTRIBUTION.md`); không tải lên ảnh cá nhân, dữ liệu nội bộ hay thông tin cá nhân (họ tên, MSSV, email) vào notebook, báo cáo hoặc output.
2. Nếu gặp ảnh/dữ liệu không đúng phạm vi (ảnh cá nhân, nhạy cảm, không rõ giấy phép), mình **dừng lại, không xử lý hay commit**, và báo ngay cho **Lab Coach / giảng viên phụ trách**.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
