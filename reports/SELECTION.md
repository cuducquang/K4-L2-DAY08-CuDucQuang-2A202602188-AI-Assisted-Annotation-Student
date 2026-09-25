# Vì sao chọn lô này?

Nguồn số liệu: `outputs/selection_round1.csv` (268 ảnh pool, cột `score = 0.5·U + 0.3·A + 0.2·D`),
`outputs/selection_round1.jpg` (contact sheet 12 ảnh được chọn) và `outputs/round1_diff.md` (kết quả
sửa nhãn sau khi đã chọn). Thời điểm `t` tính bằng giây trong video (2.5 frame/giây).

## Top 5 nếu chỉ có ngân sách rà năm ảnh

Nhận xét chung về 50 dòng đầu: điểm chỉ trải từ 0.8203 (hạng 50) đến 0.9591 (hạng 1), cả 50 dòng
đều có `D = 1.0` và `empty = False`. Nghĩa là thành phần đa dạng thời gian không phân biệt được
ứng viên nào ở vòng 1, và không có ảnh nào model "không đoán được box" (cả 268 ảnh pool đều
`empty = False`, nên `EMPTY_BONUS` không được dùng). Thứ tự vì vậy do `U` và `A` quyết định, và nhiều
ứng viên dồn vào vài cụm thời gian: quanh 147–153 s (0369, 0372, 0368, 0374, 0380, 0383, 0384),
quanh 130–132 s (0326, 0328, 0329, 0330, 0331), quanh 124–125 s (0309, 0310, 0312, 0313, 0314).
Với ngân sách chỉ năm ảnh, mình ưu tiên mỗi cụm một ảnh thay vì lấy đúng năm hạng đầu:

| Thứ tự | Frame | Hạng CSV | score | t (s) | Lý do |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | frame_0182.jpg | 1 | 0.9591 | 72.8 | Điểm cao nhất; `A = 1.0` (18/28 box ở vùng tin cậy 0.15–0.5), cụm 72–75 s không trùng cụm nào khác. |
| 2 | frame_0369.jpg | 2 | 0.9324 | 147.6 | Đại diện cụm 147–153 s. Chọn 0369 thay vì 0372 (hạng 6, cách 1.2 s) và 0368 (hạng 9, cách 0.4 s): hai ảnh này gần trùng, cùng dòng xe. |
| 3 | frame_0326.jpg | 4 | 0.9155 | 130.4 | Đại diện cụm 130–132 s. Bỏ 0331 (hạng 5, 0.9154) dù điểm gần bằng: cách 0326 chỉ 2.0 s, nhiều xe còn trong khung. 0330 (hạng 12) cách 0331 0.4 s. |
| 4 | frame_0312.jpg | 7 | 0.9100 | 124.8 | Cụm 124–125 s, cách 0326 5.6 s; `A = 1.0` (18 box mơ hồ trên 37). |
| 5 | frame_0099.jpg | 8 | 0.9063 | 39.6 | `U = 0.946`, ảnh duy nhất ở nửa đầu video trong 8 hạng đầu. Không có nó, cả 5 ảnh đều nằm ở 72–153 s. |

Quyết định về ảnh gần trùng: bỏ frame_0380.jpg (hạng 3, 0.917) khỏi top 5 dù điểm cao hơn 0326.
Nó cách 0369 4.4 s, vượt `MIN_GAP_S = 2.0` nên vẫn vào lô 12. Nhưng xét công rà, hai ảnh cùng kiểu
dòng xe chiều ngược đông đúc: sau khi sửa, 0369 thêm 13 box và 0380 thêm 9 box (`round1_diff.md`),
lỗi cùng một dạng. Với ngân sách năm ảnh, một ảnh đại diện là đủ.

## Ba frame thuộc lô 12 ảnh model chọn, kèm bằng chứng

- **frame_0182.jpg** (hạng 1, score 0.9591, U 0.9182, A 1.0, n_boxes 28, n_ambiguous 18). Ảnh có
  nhiều box nhất trong vùng tin cậy mơ hồ. Khi rà: 13 box gợi ý, giữ 12, xóa 1, thêm 10
  (`round1_diff.md`). Model bỏ sót đúng hai chỗ đã ghi trong `BLIND_SCAN.md`: SUV tối bị cắt ở mép
  dưới và xe nhoè góc dưới phải. Điểm cao ở đây đúng là dấu hiệu ảnh khó.
- **frame_0369.jpg** (hạng 2, score 0.9324, U 0.9315, A 0.8889, n_boxes 43, n_ambiguous 16). Ở ngưỡng
  pre-label 0.25 chỉ còn 14 box, nhưng sau khi sửa có 27 box: thêm 13, nhiều nhất lô. Phần lớn là xe
  chiều ngược chỉ còn cụm đèn pha. Chênh lệch 43 box ở ngưỡng 0.05 so với 14 box ở 0.25 cho thấy model
  "thấy" nhiều xe nhưng không đủ tự tin, đúng loại ảnh mà điểm bất định nhắm tới.
- **frame_0392.jpg** (hạng 15, score 0.8874, U 0.9747, A 0.6667, n_boxes 35, n_ambiguous 12). `U` cao
  nhất lô nhưng `A` thấp nhất nên chỉ đứng hạng 15. Khi rà: thêm 7 box (hai xe tối bị cắt ở mép dưới,
  một xe tải thùng trắng phía xa), sửa 1 box gộp hai xe (P6). Ảnh vào lô được là nhờ các hạng 6, 9,
  12 bị `MIN_GAP_S` loại, đẩy nó lên thành ảnh thứ 12.

Trên contact sheet `selection_round1.jpg`, cả 12 ảnh đều là cảnh đêm cùng góc camera, chỉ khác mật độ
xe. Đây là giới hạn của một video cố định: đa dạng chỉ nằm ở trục thời gian.

## Một frame điểm cao nhưng không chọn, và một frame điểm thấp vẫn nên xem

- **frame_0372.jpg** (hạng 6, score 0.9101, t = 148.8 s): điểm cao hơn 7 ảnh được chọn nhưng bị loại
  vì cách 0369 chỉ 1.2 s (< `MIN_GAP_S`). Đây là quyết định đúng: xe ở 0372 là xe ở 0369 dịch đi vài
  chục pixel. Gán cả hai tốn gấp đôi công mà gần như không thêm thông tin.
- **frame_0195.jpg** (hạng 268, thấp nhất, score 0.5721, U 0.6442, A 0.1667, n_boxes 20) nên được xem
  như ảnh đối chứng. Điểm thấp chỉ nói model tự tin, không nói model đúng. Trong lô đã sửa, lỗi chủ yếu
  là **bỏ sót** (85 box thêm so với 7 box xóa), và một xe bị bỏ sót hoàn toàn không sinh box nào để
  tính U. Nếu 0195 cũng thiếu xe tối ở mép dưới hay xe tải lớn, đó là lỗ hổng mà điểm bất định không
  bắt được.

## Điều phép chọn này chưa chứng minh về chất lượng mô hình

- Điểm bất định cao chỉ nói model phân vân ở ảnh đó. Nó không chứng minh model sai ở đó, và càng không
  chứng minh gán ảnh đó sẽ làm AP50 trên test tăng. Muốn kết luận phải xem `metrics_round1.json` trên
  cùng 20 ảnh test.
- Công thức chỉ nhìn box model đã đề xuất. False negative (xe không có box nào, như xe tải ở
  frame_0270 hay SUV tối ở frame_0182) không đóng góp vào U hay A.
- `D = 1.0` cho mọi ứng viên, nên vòng này thực chất chỉ xếp theo U và A. Ràng buộc đa dạng thật sự
  duy nhất là `MIN_GAP_S`.
- Vòng này không chạy chiến lược `random` để đối chứng, nên chưa chứng minh được uncertainty sampling
  tốt hơn chọn ngẫu nhiên trên dữ liệu này.
