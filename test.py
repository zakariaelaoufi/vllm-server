from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",  # required by SDK but vLLM ignores it
)

response = client.chat.completions.create(
    model="Qwen/Qwen3.5-2B",
    messages=[
        {"role": "user", "content": "compare ollama to vllm"}
    ],
    temperature=0.7,
    max_tokens=512,
)
print(response.choices[0].message.content)