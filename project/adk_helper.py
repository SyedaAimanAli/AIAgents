import asyncio
import random


async def safe_call_adk(agent, prompt, context=None, retries=5, base_delay=5):
    for attempt in range(retries):
        result = await agent.call_adk(prompt, context)
        if result is not None:
            return result
        # Exponential backoff: 5s, 10s, 20s, etc + jitter
        delay = base_delay * (2 ** attempt) + random.random()
        print(f"ADK call failed, retrying in {delay:.1f}s...")
        await asyncio.sleep(delay)
    print("ADK call failed after retries.")
    return None


# try:
#     from google import genai
#     from google.genai import types
#     ADK_AVAILABLE = True
# except Exception:
#     ADK_AVAILABLE = False

# class ADKHelper:
#     def __init__(self, client=None, model_id=None, temperature=0.2, max_tokens=800):
#         self.client = client
#         self.model_id = model_id
#         self.temperature = temperature
#         self.max_tokens = max_tokens

#     async def generate(self, prompt: str, context: dict = None):
#         if not ADK_AVAILABLE or not self.client:
#             return None
#         import asyncio, json
#         loop = asyncio.get_event_loop()

#         full_prompt = prompt
#         if context:
#             full_prompt = f"Context:\n{json.dumps(context, default=str, indent=2)}\n\n{prompt}"

#         def blocking_call():
#             response = self.client.models.generate_content(
#                 model=self.model_id,
#                 contents=[{"role": "user", "parts": [{"text": full_prompt}]}],
#                 config=types.GenerateContentConfig(
#                     temperature=self.temperature,
#                     max_output_tokens=self.max_tokens
#                 )
#             )
#             return getattr(response, "text", str(response))

#         return await loop.run_in_executor(None, blocking_call)
