# Custom dataset — hỗ trợ học viên AI

- Nguồn: dữ liệu tổng hợp deterministic, seed 42; không cào web hay chứa dữ liệu cá nhân.
- Quy mô: 220 yêu cầu hỗ trợ, schema JSON bốn trường của lab.
- Miền: vận hành khóa học AI, Colab, GPU credit, bài lab và chứng chỉ; khác phân phối web phổ thông.
- Khử nhiễm: 0 trùng exact, 0 trùng normalized, 0 overlap với hai tập eval đóng băng.
- Cách tạo: tổ hợp có kiểm soát giữa sản phẩm, loại yêu cầu, mức khẩn cấp và sắc thái; mọi dòng được validate vocabulary/schema.
- Phạm vi: chỉ là bằng chứng bonus B2, không thay thế corpus hoặc baseline core đã đóng băng.
