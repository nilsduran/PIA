import random
import json
from datasets import load_dataset
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import os
from collections import Counter
import matplotlib.pyplot as plt
import gradio as gr


# ----------------------------
# Configure your Gemini API key
# ----------------------------
# python3.11 -m venv .venv-stable
# .\.venv-stable\Scripts\activate
# $env:GOOGLE_API_KEY = "YOUR_ACTUAL_API_KEY_HERE"
# "AIzaSyA8KwZ5wYVoaiLlRMOI_ZsS2PYXH0qq4ms"
# Medicina general: "AIzaSyDxk7cxcrDx3mcofYIosCggfkVbyHedO4w"

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("API KEY NOT SET. Exiting.")
    exit()

genai.configure(api_key=API_KEY)

# Talk to the Gemini model with a chat UI
MODEL_NAME = "gemini-2.0-flash-lite"

# Load MedQA validation dataset
medqa = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
medqa = medqa.shuffle(seed=42)
print(medqa)


def get_random_questions(num_questions=10):
    """Get a random sample of questions from the MedQA dataset."""
    return medqa.shuffle(seed=42).select(range(num_questions))


def generate_questions_with_gemini(questions):
    """Generate questions using the Gemini model."""
    responses = []
    model = genai.GenerativeModel(MODEL_NAME)
    for question in questions:
        prompt = f"{question['question']}\n" f"Options: {', '.join(question['options'])}"
        try:
            response = model.generate_content(prompt, generation_config={"max_output_tokens": 1000})
            responses.append(response.text)
        except ResourceExhausted as e:
            print(f"API limit exceeded: {e}")
            break
        except Exception as e:
            print(f"Error generating question: {e}")
            responses.append("Error generating question")
    return responses


if __name__ == "__main__":
    # Get a random sample of questions
    # questions = get_random_questions(num_questions=10)

    # # Generate questions using Gemini
    # generated_questions = generate_questions_with_gemini(questions)

    # # Print the generated questions
    # for i, question in enumerate(generated_questions):
    #     print(f"Generated Question {i + 1}: {question}\n")
    # # Save the generated questions to a file
    # output_file = "generated_questions.json"
    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(generated_questions, f, ensure_ascii=False, indent=2)
    # print(f"Generated questions saved to {output_file}")

    # Create a simple Gradio interface
    def gradio_interface(question):
        """Gradio interface function to generate a response for a given question."""
        if not question:
            return "Please enter a question."
        try:
            response = generate_questions_with_gemini([{"question": question, "options": ["A", "B", "C", "D"]}])
            return response[0] if response else "No response generated."
        except Exception as e:
            return f"Error: {e}"

    gr.Interface(
        fn=gradio_interface,
        inputs=gr.Textbox(label="Enter your question"),
        outputs=gr.Textbox(label="Gemini Response"),
        title="Gemini Question Generator",
        description="Enter a medical question to get a response from the Gemini model.",
    ).launch()
