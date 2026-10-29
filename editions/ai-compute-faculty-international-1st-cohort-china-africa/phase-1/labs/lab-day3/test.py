from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="any" 
)

response = client.chat.completions.create(
    model="qwen",
    messages=[{"role": "user", "content": "Introduce yourself"}],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
