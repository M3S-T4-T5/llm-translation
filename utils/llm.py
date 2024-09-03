import time
import logging
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class LLM_Client:
    def __init__(self):
        # Load the API key from the environment
        self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")

        self.client = OpenAI(api_key=self.api_key)

    def chat(self, messages, model="gpt-3.5-turbo-0125", temperature=0, force_json=False):
        attempt_count = 0
        max_attempts = 3
        while attempt_count < max_attempts:
            try: 
                if force_json:
                    response = self.client.chat.completions.create(
                        model=model,
                        seed=12345,
                        response_format={"type": "json_object"},
                        temperature=temperature,
                        messages=messages
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=model,
                        seed=12345,
                        temperature=temperature,
                        messages=messages
                    )
                return response
            except Exception as e:
                logging.error(f"Attempt {attempt_count + 1}: chat API responded with: {e}")
                # time.sleep(1)
                attempt_count += 1
    
    