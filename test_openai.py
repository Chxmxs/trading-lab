import os
import openai

# Retrieve the API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise EnvironmentError("OPENAI_API_KEY environment variable not set")

# Configure OpenAI client
openai.api_key = api_key

# Send a short prompt and print the response
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=10
)
print(response.choices[0].message.content)
