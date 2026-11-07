# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 06/11/2026

**Runtime Colab:** GPU (cuda)

**Python / PyTorch / Ultralytics:** 3.13.16 / 2.11.0+cu130 / 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không đổi checkpoint, threshold hay code chạy model. Chỉ thêm một ô `drive.mount` để lưu file ra Google Drive, ô này không ảnh hưởng đến JSON/PNG.

## 1. Phân loại ảnh

1. `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K

   Các hạng sau lần lượt là minibus (0.164), police_van (0.086), recreational_vehicle (0.054), streetcar (0.048).

2. Record này nói về cả bức ảnh. Trong record không có tọa độ box hay polygon nào, mỗi ảnh chỉ có 5 dòng xếp hạng lớp. Ảnh `traffic` có rất nhiều xe buýt và ô tô nhưng model chỉ đưa ra một nhãn chung, không chỉ ra "cab" nằm ở đâu.

3. Danh sách lớp đi kèm checkpoint. `yolo11n-cls.pt` được train trên ImageNet-1K nên chỉ chọn được trong 1000 lớp của bộ này. Ảnh đường phố mà ra "cab" hay "streetcar" là vì ImageNet không có lớp nào kiểu "cảnh giao thông", model chỉ lấy lớp gần nhất nó có.

4. Vì cùng một con số có thể mang nghĩa khác nhau ở taxonomy khác nhau. Ví dụ `class_id`=2 bên COCO-80 là `car`, còn bên ImageNet-1K là `great_white_shark`. Tên lớp cũng chia khác nhau: COCO chỉ có `bus`, ImageNet thì tách ra `minibus`, `trolleybus`, `school_bus`. Nếu chỉ ghi tên hoặc chỉ ghi id thì khi gộp dữ liệu từ nhiều nguồn rất dễ nhầm. Có đủ id, tên và taxonomy thì máy đọc được, người đọc cũng kiểm tra được.

5. Guideline cần ghi rõ cách chọn lớp chính khi ảnh có nhiều vật, ví dụ chọn vật chiếm diện tích lớn nhất hoặc vật ở trung tâm, hoặc ưu tiên theo mục đích dự án. Nên có thứ tự ưu tiên khi hai lớp ngang nhau (ví dụ bus trước car), kèm vài ảnh mẫu làm đúng/làm sai. Cũng cần nói khi nào phải hỏi Lab Coach, ví dụ không có lớp nào trong taxonomy hợp với ảnh.

6. Score chỉ cho biết model tin vào đáp án của nó đến đâu, không cho biết đáp án đó đúng. Ở ảnh này top-1 chỉ được 0.51, trong khi nhìn ảnh thì xe buýt mới là thứ chiếm nhiều diện tích nhất, không phải taxi. Model hoàn toàn có thể tự tin mà vẫn sai. Nếu lấy prediction làm ground truth thì lỗi của model sẽ bị đưa ngược lại vào dữ liệu train.

## 2. Phát hiện vật thể

Record được chọn: người đầu bếp đứng bên phải ảnh `kitchen` (ảnh 640×427).

1. `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

2. Gốc tọa độ (0, 0) là góc trên-trái ảnh, đơn vị pixel. Góc trên-trái của box ở điểm (385.33, 69.24), tức cách mép trái khoảng 385 px và cách mép trên khoảng 69 px, ngay trên đầu người. Góc dưới-phải ở (498.92, 348.92), ngang gót chân, còn cách đáy ảnh khoảng 78 px. Box rộng 113.58 px và cao 279.68 px, cao gấp khoảng 2.5 lần chiều rộng, đúng với dáng người đứng.

3. So sánh ngưỡng 0.20 và 0.60:
   - 0.20: 17 prediction (person, bowl, bowl, oven, oven, person, bowl, bowl, cup, cup, bowl, spoon, potted plant, spoon, dining table, spoon, bottle)
   - 0.60: 6 prediction (person, bowl, bowl, oven, oven, person)

   Ngưỡng 0.35 dùng cho JSON thì có 11 prediction.

4. Hạ ngưỡng thì có thêm nhiều box, nên reviewer phải soi kỹ hơn:
   - Kiểm tra box mới có đúng là vật thật không. Từ 0.35 xuống 0.20 có thêm 3 `spoon`, `potted plant`, `dining table` và `bottle`, toàn là prediction score thấp. Notebook không lưu ảnh ở ngưỡng 0.20 nên reviewer phải mở ảnh gốc để xác nhận từng cái. Đối chiếu với ảnh segmentation thì `potted plant` nhiều khả năng là bó lá khô treo ở góc trên bên trái, không phải chậu cây.
   - Phân biệt box trùng với hai vật đứng sát nhau. Trên PNG, nhãn của hai box `cup` bị in đè lên nhau thành "cucup 0.45", nhìn tưởng trùng. Mở JSON ra thì đó là hai cốc riêng: `cup 0.451108` [143.52, 268.13, 172.93, 301.81] và `cup 0.381542` [119.95, 272.2, 141.78, 304.16], hai box không giao nhau. Còn `bowl 0.381215` [0.32, 305.02, 87.69, 371.32] và `bowl 0.719063` [32.65, 342.12, 100.16, 384.93] thì có giao nhau (IoU khoảng 0.23), nhưng nhìn ảnh là hai đồ khác nhau (khay bánh phía sau và đĩa sứ trắng phía trước).
   - Kiểm tra lớp có đúng không. Hai vật nhỏ trên kệ được gán `bowl 0.701282` và `bowl 0.464904` nhưng nhìn giống cốc hoặc hũ hơn.
   - Kiểm tra box có ôm khít vật không.

5. Quy tắc box: mỗi vật thuộc taxonomy vẽ đúng một box, box ôm khít phần nhìn thấy được của vật, không chừa nền thừa và không đoán thêm phần bị che hoặc nằm ngoài khung ảnh.

6. Ở mép trái ảnh, chỗ cái bàn, chỉ thấy một cánh tay của người thứ hai, phần còn lại nằm ngoài khung. Model gán `person 0.610931` với box [0.12, 263.18, 61.55, 310.78]. `x_min` gần bằng 0 cho thấy box chạm mép ảnh, và box chỉ rộng khoảng 61 px, cao 48 px.

   Trường hợp này nên viết thành quy tắc trong guideline, vì người bị cắt mép ảnh gặp rất thường xuyên, cần mọi người làm giống nhau: vẫn gán `person` nếu nhận ra được đó là người, box chỉ bao phần nhìn thấy. Chỉ hỏi Lab Coach khi phần nhìn thấy quá nhỏ hoặc quá mờ, không chắc có phải người không.

## 3. Phân đoạn theo từng đối tượng

Record được chọn: cùng người đầu bếp như ở mục 2.

1. `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; `polygon_point_count`=348 ; 5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]

2. Box chỉ là một hình chữ nhật [385.45, 66.44, 498.02, 348.58] quanh người. Polygon thì dùng 348 điểm đi theo viền thật của người. Ví dụ 5 điểm đầu đi từ (446, 70) sang (442, 72), tức là đang men theo đỉnh đầu chứ không phải một cạnh thẳng. Trên `segmentation_prediction.png`, box của người này còn chứa cả phần tường, mấy cây cán bột treo sau đầu và khoảng trống hai bên hông, nhưng mask chỉ tô phần thân người. Vật càng to và viền càng phức tạp thì càng cần nhiều điểm: mask `dining table` (kitchen-005) có 598 điểm, còn cái bát nhỏ trên kệ (kitchen-010) chỉ có 12 điểm.

3. `instance_id` dùng để phân biệt từng vật trong một ảnh, kể cả khi chúng cùng lớp. Ví dụ ảnh `kitchen` có hai person là `kitchen-001` (người đầu bếp, 0.899) và `kitchen-009` (cánh tay ở mép trái, 0.431). Cả hai đều có `class_id`=0 nhưng mỗi cái một `instance_id` và một polygon riêng. Bốn cái bát `kitchen-002/003/006/010` cũng vậy.

   Nó không phải class ID vì class ID cho biết vật đó là gì, còn `instance_id` cho biết là cái nào. Nó cũng không phải tracking ID vì số này chỉ đánh theo thứ tự output trong một lần chạy (ở đây xếp theo score giảm dần: 001 là 0.899, 002 là 0.736…). Chạy lại, đổi ngưỡng hoặc sang ảnh khác thì cùng một người có thể bị đánh số khác, nên không dùng nó để theo dõi một vật qua nhiều frame được.

4. Quy tắc vẽ mask: polygon đi theo mép ngoài của phần nhìn thấy được, mỗi vật một polygon riêng. Không tô lấn sang nền hay vật bên cạnh. Khe hở nhìn xuyên qua được (giữa hai chân, giữa tay và người) thì không tính vào mask. Chỗ bị vật khác che thì không vẽ, vật ở trước giữ nguyên mask, vật ở sau chỉ lấy phần lộ ra, hai mask không đè lên nhau.

5. Trên bàn bên trái có hai cái bát chạm nhau: `kitchen-006` (bowl 0.528201, box [0.11, 302.85, 85.7, 370.15]) và `kitchen-002` (bowl 0.735744, box [31.44, 343.77, 100.89, 385.55]). Hai box chồng nhau ở khoảng x 31–86, y 344–370. Trường hợp này dùng guideline, vì hai vật chạm nhau gặp rất nhiều và quy tắc ở câu 4 đã giải quyết được (vật trước giữ mask, vật sau lấy phần lộ ra).

   Mask xanh lam lớn ghi `dining table 0.58` ở nửa dưới bên trái `segmentation_prediction.png` (kitchen-005, box [0.97, 234.74, 345.86, 423.87]) đang phủ lên cái bàn gỗ để nhào bột, trên bàn có cây cán bột, khay bánh, hũ mứt. Đây không phải bàn ăn, nhưng COCO-80 không có lớp "bàn bếp" nên `dining table` là lớp gần nhất. Không rõ nên gán `dining table` hay để trống vì vật này không có trong taxonomy. Mask này còn lem xuống cả cái thùng kim loại dưới gầm bàn. Phần lem thì sửa được theo quy tắc ở câu 4, nhưng chuyện có gán lớp `dining table` hay không thì phải escalation cho Lab Coach. Lý do là chọn cách nào cũng phải áp dụng cho mọi ảnh có bàn bếp, đó là quyết định về taxonomy của cả dự án, annotator không tự chọn được.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho cả ảnh, gồm `class_id`, `class_name`, `taxonomy_name` | Ảnh `traffic` toàn bus và car nhưng top-1 lại là `cab` (0.51). ImageNet không có lớp nào cho cảnh giao thông | Nhìn cả ảnh, chọn một lớp theo quy tắc chọn chủ thể chính; không có lớp phù hợp thì hỏi Lab Coach | Lớp đã chọn có đúng guideline không, các annotator có chọn giống nhau ở ảnh nhiều vật không |
| Phát hiện vật thể | Mỗi vật một box: `class_id`, `class_name`, `bbox_xyxy` (pixel) | Nhãn hai `cup` in đè thành "cucup" dù là hai box riêng; vật trên kệ bị gán `bowl` nhưng giống cốc; nồi chảo treo trên tường không có lớp trong COCO nên bị bỏ; `person 0.61` chỉ là cánh tay ở mép ảnh; số box là 17/11/6 tùy ngưỡng | Vẽ box khít cho mọi vật thuộc taxonomy, không phụ thuộc ngưỡng của model, chọn đúng lớp, không vẽ trùng | Có vật bị bỏ sót không, có box thừa/trùng không, lớp đúng chưa, box có khít không, vật bị che/cắt mép làm đúng quy tắc chưa |
| Instance segmentation | Mỗi vật một polygon: `instance_id`, `class_id`, `class_name`, `polygon_xy` (pixel) | Hai bát kitchen-006 và kitchen-002 chạm nhau; mask `oven` (kitchen-007) bị hổng một mảng tối ở góc trên-trái và tràn xuống mép bàn; bàn bếp bị gán `dining table` và mask lem xuống cả thùng kim loại dưới gầm bàn; kitchen-009 chỉ là cánh tay; cùng ngưỡng 0.35 nhưng model seg ra lớp khác model detection (có spoon, potted plant, dining table, không có cup) | Vẽ polygon theo viền phần nhìn thấy, tách riêng từng vật cùng lớp, các mask không đè lên nhau | Viền có bám sát vật không, có lem sang nền/vật khác không, mỗi vật có đúng một instance không, chỗ khó đã làm theo guideline hoặc đã hỏi Lab Coach chưa |

## 5. An toàn dữ liệu

1. Chỉ dùng ba ảnh COCO công khai có sẵn trong notebook, không đưa ảnh cá nhân, dữ liệu nội bộ hay thông tin cá nhân (họ tên, MSSV, email) vào notebook, báo cáo hoặc output.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi thì dừng lại, không xử lý hay commit, và báo cho Lab Coach.

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
