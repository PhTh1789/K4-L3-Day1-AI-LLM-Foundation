import os
import time
from typing import Any, Callable

from dotenv import load_dotenv
from template import compare_models, PRICING_PER_1K_TOKENS

# Nạp OPENAI_API_KEY từ file .env (copy .env.example thành .env và dán key vào)
load_dotenv()

# ---------------------------------------------------------------------------
# Bảng giá ước tính (USD / 1K token) — cập nhật nếu giá thay đổi
# ---------------------------------------------------------------------------
PRICING_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.0025, "output": 0.010},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

# Tên model có thể đổi qua .env — ví dụ khi dùng NVIDIA NIM miễn phí
# (xem LAB_GUIDE.md, Phụ lục B). Không đặt gì trong .env thì mặc định OpenAI.
OPENAI_MODEL = os.getenv("LAB_MODEL", "gpt-4o")
OPENAI_MINI_MODEL = os.getenv("LAB_MINI_MODEL", "gpt-4o-mini")


# ===========================================================================
# PART 1 — API CƠ BẢN (Block 1: phút 60–100)
# ===========================================================================

# ---------------------------------------------------------------------------
# Task 1.1 — Gọi GPT-4o
# ---------------------------------------------------------------------------
def call_openai(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 256,
) -> tuple[str, float]:
    """
    Gọi OpenAI Chat Completions API, trả về nội dung phản hồi + độ trễ.

    Args:
        prompt:      Tin nhắn của người dùng.
        model:       Model OpenAI sử dụng (mặc định: gpt-4o).
        temperature: Độ ngẫu nhiên khi lấy mẫu (0.0 – 2.0).
        top_p:       Ngưỡng nucleus sampling.
        max_tokens:  Số token tối đa được sinh ra.

    Returns:
        Tuple (response_text: str, latency_seconds: float).

    Gợi ý:
        from openai import OpenAI            # import BÊN TRONG hàm
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # đo thời gian bằng time.perf_counter() trước và sau lời gọi API
        # (perf_counter là đồng hồ đo khoảng thời gian, độ phân giải cao trên
        #  mọi hệ điều hành; time.time() trên Windows có thể trả về 0.0)
    """
    # TODO: import OpenAI, tạo client, gọi chat.completions.create,
    #       đo start/end time, trả về (response_text, latency)
    # raise NotImplementedError("Implement call_openai")
    from openai import OpenAI
 
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - start
    return response.choices[0].message.content, latency


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
    for temp in [0.0, 0.5, 1.0, 1.5]:
        text, latency = call_openai(PROMPT, temperature=temp)
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
    print(f"\n[GPT-4o]       {result['gpt4o_latency']:.2f}s | Cost: ${result['gpt4o_cost_estimate']:.6f}")
    print(f"Response: {result['gpt4o_response'][:120]}...")
    print(f"\n[mini]         {result['mini_latency']:.2f}s")
    print(f"Response: {result['mini_response'][:120]}...")

    # Tính toán chi phí lý thuyết cho bài 1.3
    # Kịch bản: 10.000 user/ngày x 3 lần/user x 350 token output
    daily_output_tokens = 10_000 * 3 * 350  # = 10,500,000 token
    cost_4o   = (daily_output_tokens / 1000) * PRICING_PER_1K_TOKENS["gpt-4o"]["output"]
    cost_mini = (daily_output_tokens / 1000) * PRICING_PER_1K_TOKENS["gpt-4o-mini"]["output"]
    ratio     = cost_4o / cost_mini
    print(f"\n--- Phân tích chi phí hằng ngày (10K user, 3 lần, 350 token output) ---")
    print(f"GPT-4o   : ${cost_4o:.2f}/ngày")
    print(f"GPT-4o-mini: ${cost_mini:.2f}/ngày")
    print(f"GPT-4o đắt hơn mini khoảng: {ratio:.1f} lần")