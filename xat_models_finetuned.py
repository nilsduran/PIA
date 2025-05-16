import requests
import gradio as gr


# MedQA pregunta de prova:
# A previously healthy 30-year-old woman comes to the physician for the evaluation of pain during sexual intercourse for 6 months. She also reports frequent episodes of crampy pelvic pain that starts one day before menses and lasts for 7 days. Her symptoms are not relieved with pain medication. Menses occur at regular 28-day intervals and last 5 days. Her last menstrual period was 2 weeks ago. She is sexually active with her husband. She uses a combined oral contraceptive pill. Her vital signs are within normal limits. Physical examination shows rectovaginal tenderness. Cervical and urethral swabs are negative. Transvaginal ultrasonography shows no abnormalities. Which of the following is the most appropriate next step in management?
# A: Measurement of CA-125 levels
# B: Hysterectomy
# C: Laparoscopy
# D: Hysteroscopy

# Respota correcta: C


def generate_content(api_key, tuned_model_id, prompt, temperature=0.7, max_output_tokens=2048):
    """Generate content using the fine-tuned model API."""
    # Construct the endpoint URL
    url = f"https://generativelanguage.googleapis.com/v1/tunedModels/{tuned_model_id}:generateContent?key={api_key}"

    # Set the request headers
    headers = {"Content-Type": "application/json"}

    # Prepare the payload
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_output_tokens},
    }

    # Make the POST request to the API endpoint
    response = requests.post(url, headers=headers, json=payload)

    # Check for a successful response
    if response.status_code == 200:
        response_json = response.json()
        try:
            # Extract the text from the response
            return response_json["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return "Error: Could not extract response text from API response."
    else:
        return f"Error {response.status_code}: {response.text}"


# Available models
MODELS = {
    "Medicina General": "medicinageneralcsv-q4i0ydc9l1uvxbzsxmii8",
    "Ciències Bàsiques": "cincies-bsiques-5x23mkxv2ftipprirc4i4714",
    "Patologia i Farmacologia": "patologia-i-farmacologia-3ipo0rdy5dkze8q",
    "Cirurgia": "cirurgia-6rm1gub7hny7bzm3hjgghwcf3tws7ar",
    "Pediatria i Ginecologia": "pediatria-i-ginecologia-q4n2dg2t5sweqdt9",
}

# API key
API_KEY = "AIzaSyDxk7cxcrDx3mcofYIosCggfkVbyHedO4w"


def respond(message, history, model_name, temperature):
    """Process user input and get model response."""
    # Get the model ID from the selected model name
    model_id = MODELS.get(model_name)
    if not model_id:
        return history + [[message, "Error: Model not found."]]

    # Generate content
    response = generate_content(API_KEY, model_id, message)

    return history + [[message, response]]


# Launch the app
if __name__ == "__main__":
    # Create the Gradio interface
    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        gr.Markdown("# Model Mèdic - Xat de Consulta")
        gr.Markdown("Aquest xat utilitza models ajustats per respondre a preguntes mèdiques.")

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    [],
                    elem_id="chatbot",
                    height=400,
                    label="Historial de xat",
                )
                msg = gr.Textbox(
                    placeholder="Escriu la teva pregunta aquí...",
                    container=False,
                    label="Pregunta",
                )
                clear = gr.Button("Netejar Xat")

            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value="Medicina General",
                    label="Model",
                )
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Temperatura",
                    info="Controla la creativitat de les respostes",
                )

        # Set up the response logic
        msg.submit(
            respond,
            inputs=[msg, chatbot, model_dropdown, temperature_slider],
            outputs=[chatbot],
        ).then(
            lambda: "", None, [msg]  # Clear the textbox after sending
        )

        # Clear chat button
        clear.click(lambda: [], outputs=[chatbot])

        # Add examples
        gr.Examples(
            examples=[
                "Am I sick or is this just me?",
                "Treatment for university related burnout",
                "When is a bypass vs a stent used?",
            ],
            inputs=msg,
        )

        # Add footer with information
        gr.Markdown(
            """
        ### Informació sobre els models
        - **Medicina General**: Coneixements generals de medicina
        - **Ciències Bàsiques**: Anatomia, fisiologia i altres ciències bàsiques
        - **Patologia i Farmacologia**: Especialitzat en malalties i tractaments farmacològics
        - **Cirurgia**: Procediments quirúrgics i atenció perioperatòria
        - **Pediatria i Ginecologia**: Salut infantil i de la dona
        """
        )

    demo.launch()
