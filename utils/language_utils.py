from langdetect import detect
import logging

def is_english(text):
    return all(ord(char) < 128 for char in text)

def translate_text(text, llm_client):
    messages = [{"role": "user", "content": f"Translate the following text into English. Ensure the translation is accurate and retains the original meaning.\n\n{text}"}]
    resp = llm_client.chat(messages=messages)
    return resp.choices[0].message.content
