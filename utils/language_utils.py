from prompts import TRANSLATION_PROMPT
from lingua import Language, LanguageDetectorBuilder

detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()


def is_english(text):
    confidence = detector.compute_language_confidence_values(text)
    sorted_confidence = sorted(confidence, key=lambda x: x.value, reverse=True)
    top1_language = sorted_confidence[0].language
    return top1_language == Language.ENGLISH
    
def translate_text(text, llm_client):
    messages = [{"role": "user", "content": TRANSLATION_PROMPT.format(original_text=text)}]
    temperature = 0
    resp = llm_client.chat(messages=messages, temperature=temperature)
    return resp.choices[0].message.content

