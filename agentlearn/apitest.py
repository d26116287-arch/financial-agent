#配置api deepseek为例
from openai import OpenAI
client = OpenAI(
    api_key='sk-63112b3e2de441fd9a2f4a8a3ab75753',
    base_url="https://api.deepseek.com")
while True:
    YOUR_QUESTION = input("Ask a question: ")
    if YOUR_QUESTION == "退出":
        print("Bye!")
        break
    response = client.chat.completions.create(
    model="deepseek-v4-pro",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": YOUR_QUESTION},
    ],
    stream=False,
   # max_tokens=
   # temperature=0.7
)
    print("Answer: ")
    print(response.choices[0].message.content)