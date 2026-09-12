"""
File thử nghiệm cá nhân — chạy trực tiếp để quan sát kết quả API.
Tất cả hàm được import từ template.py.
Chạy: python main.py
"""

import time

from dotenv import load_dotenv

# Import toàn bộ hàm và hằng số từ template.py
from template import (
    call_openai,
    call_openai_mini,
    compare_models,
    chat_with_system_prompt,
    count_tokens,
    estimate_cost,
    retry_with_backoff,
    run_assistant,
    PRICING_PER_1K_TOKENS,
    OPENAI_MODEL,
    OPENAI_MINI_MODEL,
)

load_dotenv()

# ===========================================================================
# THỬ NGHIỆM BLOCK 1 — Chạy: python main.py
# ===========================================================================
if __name__ == "__main__":
    PROMPT = "Hãy kể cho tôi một sự thật thú vị về Việt Nam."

    # -----------------------------------------------------------------------
    # Câu 1.1 — Thử 4 mức temperature để quan sát sự thay đổi
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("CÂU 1.1 — ĐỘ NHẠY CỦA TEMPERATURE")
    print("=" * 60)
    for i, temp in enumerate([0.0, 0.5, 1.0, 1.5]):
        if i > 0:
            print(f"  (chờ 5 giây để tránh rate limit...)")
            time.sleep(5)
        text, latency = call_openai(PROMPT, temperature=temp, max_tokens=512)
        print(f"\n--- Temperature: {temp} | Latency: {latency:.2f}s ---")
        print(text)

    # -----------------------------------------------------------------------
    # Câu 1.2 — Dùng temperature phù hợp với chatbot CSKH (0.2) để so sánh
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CÂU 1.2 — CHATBOT CSKH (temperature=0.2, nhất quán, an toàn)")
    print("=" * 60)
    text_cskh, latency_cskh = call_openai(
        "Tôi muốn đổi mật khẩu tài khoản, tôi cần làm gì?",
        temperature=0.2,
    )
    print(f"[{latency_cskh:.2f}s] {text_cskh}")

    # -----------------------------------------------------------------------
    # Câu 1.3 — So sánh chi phí GPT-4o vs GPT-4o-mini
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CÂU 1.3 — SO SÁNH CHI PHÍ GPT-4o vs GPT-4o-mini")
    print("=" * 60)
    result = compare_models(PROMPT)
    print(f"\n[GPT-4o]      {result['gpt4o_latency']:.2f}s | Cost: ${result['gpt4o_cost_estimate']:.6f}")
    print(f"Response: {result['gpt4o_response'][:120]}...")
    print(f"\n[mini]        {result['mini_latency']:.2f}s")
    print(f"Response: {result['mini_response'][:120]}...")

    # Tính toán chi phí lý thuyết cho bài 1.3
    # Kịch bản: 10.000 user/ngày x 3 lần/user x 350 token output
    daily_output_tokens = 10_000 * 3 * 350  # = 10,500,000 token
    cost_4o   = (daily_output_tokens / 1000) * PRICING_PER_1K_TOKENS["gpt-4o"]["output"]
    cost_mini = (daily_output_tokens / 1000) * PRICING_PER_1K_TOKENS["gpt-4o-mini"]["output"]
    ratio     = cost_4o / cost_mini
    print(f"\n--- Phân tích chi phí hằng ngày (10K user, 3 lần, 350 token output) ---")
    print(f"GPT-4o      : ${cost_4o:.2f}/ngày")
    print(f"GPT-4o-mini : ${cost_mini:.2f}/ngày")
    print(f"GPT-4o đắt hơn mini khoảng: {ratio:.1f} lần")