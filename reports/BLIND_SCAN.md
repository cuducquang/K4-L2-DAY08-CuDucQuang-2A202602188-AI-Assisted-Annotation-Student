# Quét độc lập trước khi xem pre-label

Frame: frame_0182.jpg (hạng 1 trong `outputs/selection_round1.csv`), mở file `.jpg` gốc, chưa mở `.txt`, `.json` hay task CVAT.

Số xe nhìn thấy bằng mắt: khoảng 25 xe. Trong đó 17 xe rõ thân (9 xe chiều ngược bên trái bật đèn pha trắng, 8 xe chiều đi bên phải thấy đèn hậu đỏ) và khoảng 8 xe ở xa sát chân trời (tọa độ y khoảng 275–325 px) chỉ còn cụm đèn, box cao dưới khoảng 16–20 px.

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe:

1. Góc dưới bên phải (khoảng x 950–1135, y 620–720): xe chiều đi bị nhoè do chuyển động, bị cắt ở mép dưới ảnh và bị vệt lóa xanh của ống kính phủ lên. AI dễ bỏ sót hoặc vẽ box ôm cả quầng đèn thay vì thân xe.
2. Giữa mép dưới (khoảng x 480–620, y 630–720): xe SUV màu tối, chỉ thấy nóc và kính, đèn pha nằm ngoài khung hình nên không có điểm sáng. AI dễ bỏ sót vì thân tối lẫn vào mặt đường.

Ghi chú thêm: cụm xe xa bên trái (x 215–335, y 275–295) và dãy đèn hậu giữa đường (x 465–640, y 305–325) nhỏ, dễ bị bỏ sót nhưng phần lớn dưới ngưỡng 16 px khi chấm.
