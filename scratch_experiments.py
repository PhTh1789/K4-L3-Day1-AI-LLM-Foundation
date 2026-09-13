import os
import tiktoken
from solution.solution import chat_with_system_prompt, count_tokens

print("=== QUESTION 2.1 ===")
prompt = "Giải thích blockchain là gì?"
sp1 = "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
sp2 = "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

try:
    ans1, lat1 = chat_with_system_prompt(sp1, prompt)
    print("--- Teacher Persona ---")
    print(ans1)
    
    ans2, lat2 = chat_with_system_prompt(sp2, prompt)
    print("\n--- Expert Persona ---")
    print(ans2)
except Exception as e:
    print("API error for 2.1:", e)

print("\n=== QUESTION 2.2 ===")
vn_text = """Bản hướng dẫn có giao diện đọc dễ hơn nằm trên VLearn. Nội dung giống nhau; chọn bản nào bạn thấy dễ theo hơn. Bài nộp cũng nộp ở đó. Nếu bạn đã clone từ trước, lấy bản mới nhất trước khi bắt đầu. Tạo môi trường ảo và cài thư viện. Chạy thử bộ test phải fail hàng loạt, đó là dấu hiệu đúng. Kỳ vọng môi trường đã sẵn sàng, chỉ còn thiếu code của bạn. Sau đó mở tài liệu và làm theo từng block. Bạn viết code trong file mẫu và câu trả lời trong phiếu quan sát; những file còn lại là giàn giáo. Pytest dùng mock nên không cần API key và không tốn tiền."""
words_count = len(vn_text.split())
estimated_tokens = words_count / 0.75
try:
    actual_tokens = count_tokens(vn_text)
    print(f"Words count: {words_count}")
    print(f"Estimated tokens (words / 0.75): {estimated_tokens:.2f}")
    print(f"Actual tokens (tiktoken): {actual_tokens}")
    difference = (actual_tokens - estimated_tokens) / estimated_tokens * 100
    print(f"Difference: {difference:.2f}%")
except Exception as e:
    print("Error for 2.2:", e)
