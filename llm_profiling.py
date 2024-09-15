import time
import timeit
from utils.language_utils import translate_text
from utils.llm import LLM_Client

# Profiling function to compare models
def profile_translation(text, llm_client, iterations=10):        
    # Capture start time
    start_time = time.time()

    # Execute translation multiple times for profiling
    for _ in range(iterations):
        translate_text(text, llm_client)
    
    # Capture end time
    end_time = time.time()

    # Calculate and display the average time taken
    avg_time = (end_time - start_time) / iterations
    return avg_time

# Example usage with your llm_client and some input text
if __name__ == "__main__":
    llm_client = LLM_Client(service="GROQ")
    input_text = "Translate this sample text to profile performance."

    # Run profiling for both models with 10 iterations
    print("Profiling translation performance for GROQ...")
    avg_time = profile_translation(input_text, llm_client, iterations=10)
    print(f"Average execution time for groq with llama-3.1-8b: {avg_time:.4f} seconds")

    llm_client = LLM_Client(service="OPENAI")
    print("Profiling translation performance for OpenAI...")
    avg_time = profile_translation(input_text, llm_client, iterations=10)
    print(f"Average execution time for OpenAI with gpt-4o-mini: {avg_time:.4f} seconds")
