# Lab 21 — Evaluation Report

**Họ tên**: CHƯA CUNG CẤP  **MSSV**: CHƯA CUNG CẤP  **Ngày**: 2026-08-21  
**Tier**: `LAPTOP`  **Base model**: `Qwen/Qwen3.5-2B`  **GPU thực tế**: `NVIDIA GeForce RTX 4060 Laptop GPU 8 GB (7.65 GiB khả dụng)`

> Mọi số liệu trong báo cáo này được chép từ `results/`; hai trường nhận diện phía trên cần được chủ bài nộp bổ sung trước khi nộp chính thức.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 — p95 đo được 98, giá trị gợi ý 256 (`results/token_stats.json`) |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 / 58 |

**Template có giữ khối `<think>` không?** Có. `results/template_check.json` ghi `ok: true`, giữ cả thẻ mở và nội dung reasoning, với phán quyết “reasoning preserved — safe to train on traces”. Tôi giữ `max_length=1024` theo tier LAPTOP để không thay đổi cấu hình chuẩn của lab; corpus thực tế có max 101 token nên không mẫu nào bị cắt.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.3936 (37/94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss (preview chỉ có một dòng JSON và token kết thúc):

```text
{"intent": "doi_tra", "urgency": "trung_binh",
 "product": "balo laptop", "sentiment": "trung_tinh"}
<|im_end|>
```

Như vậy câu trả lời đúng là phần được tối ưu, còn system prompt và ticket của người dùng không bị đưa vào loss.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---:|---:|---:|---:|
| (a) base + naive prompt | 0.000 | 0.5556 | 0.000 | 1244.8 |
| (b) base + optimized prompt | 0.600 | 0.5556 | 1.000 | 384.0 |
| (c) LoRA fine-tune | 0.995 | 0.0667 | 1.000 | 493.0 |

**(b) có thật sự mạnh hơn (a) không?** Có: target tăng 0.600 và format tăng từ 0 lên 1. Prompt (b) được đóng băng trước train với SHA `719e74d3b6232053`; tôi không sửa `OPTIMIZED_PROMPT`, không làm yếu baseline và không sửa eval sau khi thấy kết quả. Toàn bộ 50 target và 15 regression item được dùng; `smoke_mode=false`.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss | **target** | s | VRAM GB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | text-linear (12 nhóm) | 16 | 16,819,200 | 1e-4 | 0.2955 | 0.995 | 326.5 | 6.70 |
| `attn_only` | q,v (2 nhóm) | 322 matched | 16,816,128 | 1e-4 | 0.3131 | 0.990 | 286.3 | 6.70 |
| `wrong_lr` | text-linear (12 nhóm) | 16 | 16,819,200 | 1e-5 | 1.1208 | 0.470 | 330.1 | 6.70 |
| `qlora` | text-linear (12 nhóm), 4-bit | 16 | 16,819,200 | 1e-4 | 0.3013 | 1.000 | 351.3 | 4.65 |

Tất cả bốn run dùng đúng 58 optimizer step. `attn_only` lệch ngân sách tham số chỉ 3,072 tham số, xấp xỉ 0.018%, thấp hơn nhiều so với ngưỡng 5%. Xếp hạng theo **target** là `qlora` (1.000) > `correct` (0.995) > `attn_only` (0.990) > `wrong_lr` (0.470), không phải theo loss huấn luyện.

### 4.1 — Vị trí adapter so với rank

`attn_only` thua nhẹ `correct` trên target, 0.990 so với 0.995, dù ngân sách trainable gần như giống hệt. Thứ tự này giống thứ tự theo train loss trong lần chạy này: 0.3131 của `attn_only` cao hơn 0.2955 của `correct`, nhưng target mới là bằng chứng quyết định. Kết quả cho thấy tăng rank q,v lên 322 không tự động thay thế được việc phủ đúng các lớp text-linear; vị trí gắn adapter vẫn có tác động, dù chênh lệch chỉ 0.005 trên tác vụ triage hẹp nên không nên diễn giải quá mức.

### 4.2 — Learning rate sai

`wrong_lr` chỉ giảm LR mười lần, từ 1e-4 xuống 1e-5, nhưng final loss còn 1.1208 thay vì 0.2955. Log cho thấy loss của nó giảm chậm: các mốc đầu lần lượt khoảng 2.596, 2.402 và 1.935, trong khi run đúng đã xuống 2.390, 0.681 và 0.200 ở các mốc tương ứng. Nếu chỉ nhìn việc đường loss vẫn đi xuống, tôi có thể kết luận sai rằng run chỉ cần thêm thời gian; target 0.470, thấp hơn cả baseline prompt tốt 0.600, cho thấy LR ở thang full fine-tune đã làm LoRA học không đủ trong cùng ngân sách 58 step.

### 4.3 — QLoRA

QLoRA giảm peak VRAM từ 6.70 xuống 4.65 GiB, tiết kiệm 2.05 GiB, tương đương khoảng 30.6%. Đổi lại, thời gian train tăng từ 326.5 lên 351.3 giây và latency target tăng từ 493.0 lên 576.5 ms/mẫu; tuy nhiên target lại đạt 1.000, cao hơn `correct` 0.005. Vì vậy số đo này **không** ủng hộ khuyến nghị tuyệt đối “không dùng QLoRA cho dòng model này” về chất lượng target trong lần chạy hiện tại; nó chỉ cho thấy trade-off tốc độ lấy bộ nhớ, và chênh lệch chất lượng một phần tư trường trên 50 mẫu là quá nhỏ để khái quát rộng.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.395` · `regression Δ = -0.4889` · `valid_trace_rate = 0.0000`

Fine-tune thắng rõ baseline đã được prompt tử tế trên tác vụ đích: target tăng từ 0.600 lên 0.995, vẫn giữ format 1.000. Tuy nhiên nó làm regression giảm từ 0.5556 xuống 0.0667, tức giảm 0.4889 trong khi ngưỡng cho phép chỉ 0.020. Đây không phải lỗi mask: bằng chứng NB1 cho thấy câu hỏi đã bị mask và câu trả lời được supervise; cũng không phải lỗi format vì JSON đạt 100%. Các output regression cho thấy mô hình đã học quá mạnh hành vi “mọi đầu vào đều là ticket”: câu hỏi kiến thức chung bị biến thành JSON bốn trường. Vì vậy `FAILED` là phán quyết đúng và tôi không nới gate để biến nó thành PASSED. Tôi sẽ không deploy adapter này như model đa dụng; hướng khắc phục hợp lý là thêm 1–5% replay dữ liệu tổng quát, sau đó train lại cả lượt và đánh giá lại trên chính tập đã đóng băng.

---

## 6. Định tính — có cả ca thắng và ca thua

Hai dòng đầu là target; ba dòng sau minh họa regression để phản ánh trung thực catastrophic forgetting. Với các ca regression, “nhãn đúng” là từ khóa mà scorer yêu cầu.

| # | Đầu vào (rút gọn) | Nhãn đúng / từ khóa | (b) prompt hoặc kỳ vọng | (c) fine-tune | Nhận xét |
|---:|---|---|---|---|---|
| 1 | “Ốp lưng… Shipper không gọi. Hỏi cho biết thôi…” | `van_chuyen`, `thap`, sản phẩm đúng, `tich_cuc` | Base (b) đoán sai intent `hoi_thong_tin` | Đúng cả 4 trường | ✅ FT thắng |
| 2 | “Ốp lưng… Giá bao nhiêu. Mong shop phản hồi…” | `hoi_thong_tin`, `trung_binh`, sản phẩm đúng, `trung_tinh` | Base (b) sai sentiment thành `tich_cuc` | Đúng cả 4 trường | ✅ FT thắng |
| 3 | “Nồi chiên… Giao hàng chậm. Mong shop phản hồi…” | urgency `trung_binh` | Base (b) sai intent và sentiment | FT đoán urgency `cao` (3/4 trường đúng) | ❌ **FT thua target** |
| 4 | “Thủ đô của Việt Nam là thành phố nào?” | `Hà Nội` | Trả lời kiến thức chung | Trả JSON ticket, product=`thành phố` | ❌ **FT thua regression** |
| 5 | “1 km bằng bao nhiêu mét?” | `1000` | Trả lời phép đổi đơn vị | Trả JSON ticket, không có `1000` | ❌ **FT thua regression** |

Mẫu chung của các ca thua nghiêm trọng là đầu vào nằm ngoài miền ticket: adapter áp schema triage lên cả câu hỏi kiến thức. Ca target duy nhất chỉ sai mức khẩn cấp, cho thấy năng lực miền hẹp gần bão hòa nhưng ranh giới miền đã bị học quá cứng.

---

## 7. Kết luận & điều tôi học được

**Kết luận.** Tôi không nên deploy bản fine-tune này như một model dùng chung, mặc dù nó gần như hoàn hảo trên bài toán triage. Con số target 0.995 so với baseline prompt tốt 0.600 chứng minh LoRA đã chuyển hành vi phân loại vào trọng số rất hiệu quả, nhưng regression 0.0667 so với 0.5556 cũng chứng minh cái giá quá lớn: model biến gần như mọi yêu cầu thành JSON ticket. Nếu hệ thống được cô lập tuyệt đối sau một router đáng tin cậy, adapter có thể là ứng viên thử nghiệm; với endpoint nhận đầu vào mở, verdict FAILED buộc phải chặn deploy. Đòn bẩy nền tảng nhất là mask đúng, vì mask sai sẽ làm mọi thí nghiệm sau vô nghĩa. Trong các run hợp lệ, learning rate là đòn bẩy rõ nhất: giảm mười lần làm target rơi xuống 0.470. Vị trí adapter có ảnh hưởng nhỏ khi ngân sách đã khớp, còn QLoRA trong lần đo này tiết kiệm 30.6% VRAM mà không mất target, trái với một khuyến nghị tuyệt đối. Tuy nhiên chất lượng dữ liệu mới là bước tiếp theo quan trọng nhất: thêm một lượng nhỏ replay tổng quát có thể giữ hành vi ngoài miền mà không hy sinh target, và điều đó phải được xác nhận bằng một lượt train/eval mới chứ không bằng trực giác.

**Ba điều tôi học được**:

1. Assistant-only mask phải được giải mã và nhìn tận mắt; ở đây chỉ 39.36% token nằm trong loss, đúng là phần JSON trả lời.
2. Baseline prompt tốt có thể tăng target từ 0 lên 0.600 trước khi fine-tune, nên so với prompt ngây thơ sẽ phóng đại giá trị adapter.
3. Train loss không thay cho đánh giá tác vụ: QLoRA có loss 0.3013 nhưng đứng đầu target 1.000, còn regression gate vẫn loại bản `correct` vì quên thảm họa.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** trộn 1–5% replay câu hỏi tổng quát vào train, giữ nguyên seed/eval/prompt và toàn bộ ngân sách 58 step, rồi đo lại target, regression, format và latency. Tôi cũng sẽ lặp lại ít nhất ba seed để kiểm tra chênh lệch target 0.005 giữa QLoRA, `correct` và `attn_only` có bền vững hay chỉ là nhiễu trên 50 mẫu.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng
- [ ] B3 reasoning-trace collapse
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub

**Cảnh báo verifier đã giải thích:** `verdict=FAILED` là kết quả có thể chấm; nguyên nhân là regression giảm 0.4889 do catastrophic forgetting. Tôi giữ nguyên gate và phân tích thất bại thay vì nới tiêu chí.
