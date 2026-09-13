# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature tăng từ 0.0 lên 0.5, model trả lời bằng tiếng Việt, nhất quán và bám sát yêu cầu prompt (dù bị cắt vì đạt `max_tokens`). Từ temperature 1.0 trở lên, model bắt đầu chuyển sang tiếng Anh và sinh ra định dạng kỳ lạ — cho thấy temperature cao làm tăng tính ngẫu nhiên đến mức model không còn tuân theo ngôn ngữ đã yêu cầu. Temperature 1.5 thể hiện rõ nhất: câu trả lời lạc đề hoàn toàn, dùng từ ngữ không kiểm soát được.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt temperature khoảng **0.1–0.2** cho chatbot CSKH, vì mục tiêu cốt lõi là sự **nhất quán và độ tin cậy**: mọi khách hàng hỏi cùng một câu đều phải nhận được thông tin giống nhau, tránh mâu thuẫn. Temperature thấp giúp model luôn chọn câu trả lời "an toàn" và xác suất cao nhất, giảm thiểu rủi ro sinh ra thông tin sai lệch hoặc không phù hợp với chính sách công ty.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Với kịch bản 10.000 user × 3 lần × 350 token output/ngày (tổng ~10.5M token), GPT-4o tốn khoảng **$105/ngày** trong khi GPT-4o-mini chỉ tốn **$6.30/ngày** — tức **GPT-4o đắt hơn khoảng 16.7 lần**. GPT-4o xứng đáng với chi phí khi cần độ chính xác cao trong các tác vụ phức tạp như phân tích hợp đồng pháp lý, viết code phức tạp hoặc lập luận nhiều bước. Ngược lại, nên dùng mini cho các tác vụ đơn giản, lặp đi lặp lại như phân loại câu hỏi, trả lời FAQ, hoặc tóm tắt văn bản ngắn — nơi chất lượng chênh lệch không đáng kể nhưng tiết kiệm chi phí rất lớn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Khi đóng vai "giáo viên tiểu học", model trả lời bằng những câu ngắn, từ vựng thông dụng và dùng ví dụ gần gũi. Ngược lại, với persona "chuyên gia tài chính", model dùng câu dài hơn, nhiều thuật ngữ chuyên ngành (ví dụ: Asymmetric Encryption, Hash) và tập trung vào khía cạnh kỹ thuật phức tạp. Điều này cho thấy system prompt có sức mạnh "định hình" hoàn toàn giọng điệu, cách hành văn và độ sâu kiến thức của mô hình trước khi nó bắt đầu sinh text.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Trong thử nghiệm thực tế với 121 từ tiếng Việt, ước lượng theo công thức (số từ / 0.75) cho ra ~161 token, trong khi bộ đếm `tiktoken` trả về 137 token (chênh lệch khoảng -15%). Tiếng Việt thường tốn nhiều token hơn tiếng Anh vì các bộ mã hóa (tokenizer) đa phần được huấn luyện trên kho ngữ liệu tiếng Anh. Một từ tiếng Anh thường gói gọn trong 1 token, trong khi từ tiếng Việt có dấu thường bị cắt nhỏ thành 2-3 sub-token (dù model gpt-4o gần đây đã tối ưu tiếng Việt tốt hơn rất nhiều).

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng chatbot tương tác trực tiếp với người dùng (UX), vì nó giảm đáng kể thời gian chờ đến khi từ đầu tiên xuất hiện (Time to First Token), giúp người dùng không có cảm giác ứng dụng bị treo. Ngược lại, non-streaming phù hợp hơn cho các pipeline xử lý dữ liệu ngầm (background jobs), ví dụ như bóc tách dữ liệu ra định dạng JSON để lưu database, nơi hệ thống cần toàn bộ cấu trúc hoàn chỉnh để parse chứ không thể dùng từng đoạn cắt dở.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp tản mát thời gian gọi lại (ví dụ 1s, 2s, 4s, 8s) thay vì dồn dập, giúp server có thêm thời gian phục hồi. Nếu dùng delay cố định (ví dụ 1 giây) và server đang sập, hàng nghìn client sẽ cùng thức dậy và gọi lại đúng 1 giây sau đó. Sự kiện này tạo ra hiệu ứng "thundering herd" (bầy đàn sấm sét) giáng một đợt traffic khổng lồ cùng một lúc, khiến server vừa ngóc đầu lên đã lập tức sập tiếp.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> **System prompt:** "Bạn là một lập trình viên Senior. Hãy trả lời cực kỳ ngắn gọn, cung cấp code mẫu ngay lập tức mà không giải thích dài dòng trừ khi được hỏi cụ thể, và luôn trả về format Markdown."
> **Giải thích:** Việc yêu cầu "cực kỳ ngắn gọn" và "không giải thích dài dòng" giúp tiết kiệm đáng kể token output (giảm chi phí) và thời gian sinh text, đồng thời đáp ứng đúng nhu cầu của lập trình viên là cần tham khảo code ngay. Yêu cầu "trả về format Markdown" đảm bảo UX hiển thị code luôn có syntax highlighting rõ ràng.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là bộ nhớ ngắn hạn bị giới hạn cứng ở 3 lượt hội thoại (6 messages). Nếu người dùng nhắc lại một chi tiết từ 5 lượt trước, model sẽ "quên" hoàn toàn. 
> **Đề xuất cải thiện:** Thay vì cắt cứng theo số lượt (turn), ta nên cắt linh hoạt theo **số lượng token**. Ta có thể đếm tổng token của mảng messages; nếu vượt quá ngưỡng an toàn (ví dụ 2000 tokens), ta mới loại bỏ message cũ nhất. Cách này tận dụng tối đa context window mà không lãng phí ngữ cảnh như việc đếm lượt.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
