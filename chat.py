import openai

# OpenAI-compatible vLLM 서버 설정
client = openai.OpenAI(
    base_url="http://15.165.34.159:8000/v1",  # vLLM 서버 주소
    api_key="token-abc123",                 # Docker에서 설정한 API 키
)

# 대화 메시지 초기화
messages = [
    {"role": "system", "content": "You are a helpful assistant."}
]

print("💬qwen3 Chatbot (type 'exit' to quit)")
while True:
    user_input = input("👤 You: ")
    if user_input.strip().lower() in {"exit", "quit"}:
        print("👋 종료합니다.")
        break

    # 사용자 입력 추가
    messages.append({"role": "user", "content": user_input})

    # 응답 요청
    try:
        response = client.chat.completions.create(
            model="/models/qwen3-32b",
            messages=messages,
            temperature=0.7,
            max_tokens=512,
        )
        reply = response.choices[0].message.content
        print(f"🤖 Bot: {reply}")

        # 응답 메시지도 히스토리에 추가
        messages.append({"role": "assistant", "content": reply})
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

