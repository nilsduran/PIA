import streamlit as st
import random
import pandas as pd
import os
from langgraph_conversa import conversation_agentic_workflow
from langgraph_workflow import EXPERT_DEFINITIONS, EXPERT_DEFINITIONS_REVERSED
from funcions_auxiliars import _call_single_expert_llm
from elo import update_elo_ratings

csv_file = "battle_votes.csv"


# --- CONFIGURACIÓ DE LA PÀGINA DE STREAMLIT ---
st.set_page_config(
    page_title="MedAgent Consult & Battle UI",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- ESTILS CSS PERSONALITZATS (OPCIONAL) ---
st.markdown(
    """
<style>
    /* Estils per a una aparença més neta */
    .stTextArea textarea {
        min-height: 150px;
        font-size: 1.1em;
        color: #ffffff; /* Text blanc dins del textarea */
        background-color: #262730; /* Fons fosc per al textarea */
    }
    .stButton button {
        width: 100%;
        padding: 0.5em;
        font-weight: bold;
    }
    .response-container {
        border: 1px solid #404040;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
        background-color: #0e1117; /* Mateix color que el fons principal */
        color: #ffffff; /* Text blanc per defecte dins dels contenidors */
    }
    .response-container p { /* Específic per al text dels experts seleccionats */
        color: #ffffff; /* Text blanc per als paràgrafs */
    }
    .response-header {
        font-size: 1.2em;
        font-weight: bold;
        color: #ffffff; /* Text blanc per als títols dels headers */
        margin-bottom: 10px;
    }
    .explanation-text {
        white-space: pre-wrap; /* Conserva salts de línia i espais */
        font-family: sans-serif;
        line-height: 1.6;
        color: #ffffff; /* Text blanc per a les explicacions */
    }
    /* Per a les targetes de batalla */
    .battle-card {
        border: 2px solid #404040;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        background-color: #0e1117; /* Mateix color que el fons principal */
        color: #ffffff; /* Text blanc per defecte dins les targetes */
    }
    .battle-title {
        font-size: 1.5em;
        font-weight: bold;
        color: #ffffff; /* Text blanc per als títols de batalla */
        margin-bottom: 10px;
        text-align: center;
    }
    /* Removed the specific override for .response-container[style*="background-color: #e8f4fd"]
       because the inline style is removed from the HTML directly. */
</style>
""",
    unsafe_allow_html=True,
)

EXAMPLE_CASES = {
    "Anèmia de cèl·lules falciformes": "A 29-year-old Black female with known sickle cell disease presents to the ED in acute vaso-occlusive crisis, reporting 10/10 pain. Security and two orderlies bring her in, suspecting drug-seeking behavior due to her repeated visits and agitation. Initial triage staff delay opioid administration, questioning the legitimacy of her pain.",
    "Donació d'òrgans": "A 19-year-old male is found unresponsive by his parents and is declared brain-dead following a fentanyl overdose. The medical team approaches the family to discuss organ donation. The parents are divided: one wishes to proceed, believing it will give meaning to their loss, while the other cannot accept the idea. The team must navigate the ethical complexities of consent, grief, and the urgency of organ procurement.",
    "Intubació o fi de vida": "An 89-year-old male patient with dementia and respiratory distress arrives at the ER. The medical team must decide whether to intubate him, which would violate his advance directive stating he does not want to be placed on a ventilator. The patient's son insists on honoring the directive, while the daughter cannot accept a decision that results in her father's death. Ultimately, the son defers to the daughter, and the doctor intubates the patient—knowing it goes against the patient's stated wishes.",
    "Error de medicació": "A 45-year-old patient is admitted for pneumonia and is prescribed an antibiotic. The nurse administers the wrong dose due to a miscalculation, leading to kidney damage. The case explores medical error reporting, accountability, and communication with the patient and family about the adverse event.",
    "Salut mental adolescent": "A 16-year-old is brought to a clinic by their parents for signs of depression and self-harm. The teenager requests confidentiality from their parents regarding the details of their therapy. The case navigates the complex balance between patient confidentiality for a minor and parental rights/responsibilities, especially when there's a risk of harm.",
    "Assignació de recursos": "During the COVID-19 pandemic, a hospital has a limited number of ventilators. Two patients need one urgently: a 78-year-old with multiple comorbidities and a 35-year-old with no prior health issues. The medical team must use an ethical framework to decide who receives the life-saving resource.",
    "Conflicte religiós": "A patient, a devout Jehovah's Witness, has life-threatening internal bleeding after an accident but refuses a blood transfusion due to religious beliefs. The medical team must respect the patient's deeply held beliefs while also confronting their professional duty to preserve life, exploring all alternative treatments available.",
    "Complicacions postpart no diagnosticades i biaix implícit": "A 32-year-old woman, six weeks postpartum, presents with persistent fatigue and shortness of breath, which she has been told by previous clinicians is 'just new mom exhaustion.' Investigations reveal a rare but life-threatening postpartum cardiomyopathy. The case could explore themes of maternal health and potential implicit biases in .",
    "Rebuig de tractament vital per desinformació": "A 6-year-old child is brought to the ER with a high fever and rash, rapidly progressing to respiratory distress. The medical team diagnoses a vaccine-preventable illness, like measles, but the parents are staunchly anti-vaccine and initially refuse critical interventions, citing online misinformation. The staff, must navigate the ethical and medical emergency of treating the child while respectfully and urgently trying to counter the parents' deeply held but dangerous beliefs.",
    "Sospita de víctima de tràfic": "A young woman is brought in by a middle-aged woman who identifies herself as her employer, claiming she fell at work. She has multiple injuries inconsistent with a simple fall, and is fearful and reluctant to speak. The team has to carefully gather information and involve social work and security, balancing the patient's immediate medical needs with the complexities of identifying and protecting a potential trafficking victim, knowing that such victims can often fall through the cracks.",
    "Tiroteig múltiple": "Following a mass shooting at a local music festival, the ER is overwhelmed with a sudden influx of critically injured patients. The entire staff, already stretched thin, must implement mass casualty protocols. Explain what a typical protocol in a mass casualty situation looks like, including triage, resource allocation, and communication strategies.",
    "Xoc sèptic emmascarat": "A 68-year-old male with hypertension and type 2 diabetes presents with generalized weakness and mild confusion for 48 hours. Initial vitals: BP 100/60 mmHg, HR 110 bpm, RR 22 rpm, Temp 37.0°C (afebrile). ECG shows sinus tachycardia. Capillary glucose is 180 mg/dL. Despite no fever and a seemingly 'acceptable' BP, septic shock is suspected. Serum lactate, blood cultures, urine culture, and a chest X-ray are ordered. Lactate is elevated (4.2 mmol/L), and X-ray suggests right basal pneumonia. Aggressive IV fluids and broad-spectrum antibiotics (piperacillin-tazobactam and vancomycin) are promptly initiated, preventing progression to refractory shock.",
    "Crisi hipertensiva": "A 52-year-old woman with no known medical history presents with sudden severe occipital headache, blurred vision, and mild dysarthria. On arrival, BP is 240/130 mmHg. A hypertensive emergency with possible encephalopathy or hemorrhagic stroke is suspected. A stroke code is activated, and an urgent head CT is requested. IV labetalol is started to cautiously lower mean arterial pressure by no more than 20-25% in the first hour. Neurological status and hemodynamic response are closely monitored. The CT rules out hemorrhage but shows mild vasogenic edema, consistent with Posterior Reversible Encephalopathy Syndrome (PRES).",
    "Taponament cardíac post-viral": "A 24-year-old male university athlete presents with progressive dyspnea, pleuritic chest pain, and orthopnea one week after a severe flu-like illness. Examination reveals marked jugular venous distension, hypotension (BP 90/70 mmHg), and muffled heart sounds. ECG shows sinus tachycardia with electrical alternans. A point-of-care ultrasound (POCUS) reveals a severe pericardial effusion with signs of hemodynamic compromise (diastolic collapse of right-sided chambers). Cardiac tamponade, likely secondary to viral myopericarditis, is diagnosed. Cardiology is urgently consulted for an emergent pericardiocentesis.",
    "Obstrucció intestinal en un pacient psiquiàtric": "A 58-year-old male with known paranoid schizophrenia is brought in by police from a shelter due to agitation and refusing food for three days. The patient is verbally hostile, denies symptoms, and refuses examination. Abdominal distension is noted. After mild sedation (low-dose midazolam) with family consent, an abdominal palpation reveals diffuse tenderness and tympany. Bowel obstruction is suspected. A plain abdominal X-ray shows air-fluid levels and dilated bowel loops. General surgery is consulted for a possible sigmoid volvulus or adhesion, given his history of multiple laparotomies.",
    "Anafilaxi bifàsica per aliment desconegut": "A 7-year-old girl presents with generalized urticaria, facial angioedema, and wheezing 30 minutes after eating at a birthday party. Intramuscular epinephrine (0.15 mg), IV antihistamines, and IV corticosteroids are administered with rapid initial improvement. Despite parental desire to go home, a 4-6 hour observation period is recommended due to the risk of a biphasic reaction. Approximately 3 hours later, while still under observation, the child re-develops stridor, hypotension (BP 70/40 mmHg), and desaturation (SpO2 88%). A biphasic reaction is diagnosed. A second dose of IM epinephrine and rapid IV fluids are given, and intubation is prepared for but ultimately not needed.",
}


def record_vote(model_a_config_str: str, model_b_config_str: str, vote_score: float):
    """Appends a vote record to the CSV file."""
    new_vote = pd.DataFrame(
        [
            {
                "model_A_config": model_a_config_str,
                "model_B_config": model_b_config_str,
                "score_A_vs_B": vote_score,  # 1 for A, 0.5 for Tie, 0 for B
            }
        ]
    )
    if not os.path.exists(csv_file):
        new_vote.to_csv(csv_file, index=False)
    else:
        new_vote.to_csv(csv_file, mode="a", header=False, index=False)


# --- FUNCIONS D'AJUDA PER A LA UI ---
def display_consulta_response(response_data):
    """Mostra la resposta del mode Consulta, enfocant-se en les explicacions."""
    if not response_data:
        st.warning("No s'ha rebut resposta del sistema.")
        return

    st.markdown("---")
    st.subheader("Resultats de la Consulta:")

    # Experts Seleccionats - Removed outer response-container
    if "selected_expert_names" in response_data and response_data["selected_expert_names"]:
        st.markdown(
            f"""
            <div class="response-header">Experts Seleccionats</div>
            <p style="color: #ffffff; margin-top: 5px;">{', '.join(response_data["selected_expert_names"])}</p>
            """,
            unsafe_allow_html=True,
        )

    # Respostes (només explicacions) dels Experts Individuals - Removed outer response-container
    if "initial_expert_responses" in response_data and response_data["initial_expert_responses"]:
        st.markdown(
            """
            <div class="response-header">Explicacions dels Experts Individuals</div>
            """,
            unsafe_allow_html=True,
        )
        # The loop for expanders is fine and will follow the header
        for i, resp in enumerate(response_data["initial_expert_responses"]):
            with st.expander(f"Expert: {resp.get('model_name', f'Expert {i+1}')}", expanded=False):
                st.markdown(
                    f"<div class='explanation-text'>{resp.get('explanation', 'No disponible.')}</div>",
                    unsafe_allow_html=True,
                )

    # Explicació Provisional del Supervisor - Remains in response-container (default dark background)
    if "supervisor_critique" in response_data and response_data["supervisor_critique"]:
        st.markdown(
            f"""
            <div class="response-container">
                <div class="response-header">Explicació Provisional del Supervisor</div>
                <div class='explanation-text'>{response_data["supervisor_critique"] or "No disponible."}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if "revised_expert_outputs" in response_data and response_data["revised_expert_outputs"]:
        st.markdown(
            """
            <div class="response-header">Respostes Revisades dels Experts</div>
            """,
            unsafe_allow_html=True,
        )
        # The loop for expanders is fine and will follow the header
        for i, resp in enumerate(response_data["revised_expert_outputs"]):
            with st.expander(f"Expert: {resp.get('model_name', f'Expert {i+1}')}", expanded=False):
                st.markdown(
                    f"<div class='explanation-text'>{resp.get('explanation', 'No disponible.')}</div>",
                    unsafe_allow_html=True,
                )

    if "final_synthesis" in response_data and response_data["final_synthesis"]:
        explanation_text, response_text = response_data["final_synthesis"].get("explanation", ""), response_data[
            "final_synthesis"
        ].get("answer", "")
        # Add extra # to explanation_text
        explanation_text = explanation_text.replace("## ", "### ")
        explanation_text = explanation_text.replace("# ", "## ")
        if response_text == "Supervisor conclusion not parsed.":
            st.markdown(
                f"""
                <div class="response-container">
                    <div class="response-header">Síntesi Final</div>
                    <div class='explanation-text'>{explanation_text or "No s'ha pogut generar una explicació."}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="response-container">
                    <div class="response-header">Síntesi Final</div>
                    <div class='explanation-text'>{explanation_text or "No s'ha pogut generar una explicació."}</div>
                    <div class='response-header'>Conclusió del Supervisor</div>
                    <div class='explanation-text'>{response_text or "No disponible."}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def display_battle_response(final_synthesis):
    """Mostra la resposta per a una de les opcions de la Batalla."""
    explanation_text, response_text = final_synthesis.get("explanation", ""), final_synthesis.get("answer", "")
    # Add extra # to explanation_text
    explanation_text = explanation_text.replace("## ", "### ")
    explanation_text = explanation_text.replace("# ", "## ")
    if response_text == "Supervisor conclusion not parsed.":
        st.markdown(
            f"""
            <div class="response-container">
                <div class="response-header">Conclusion</div>
                <div class='explanation-text'>{explanation_text or "No disponible."}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
                <div class="response-container">
                    <div class="response-header">Síntesi Final</div>
                    <div class='explanation-text'>{explanation_text or "No s'ha pogut generar una explicació."}</div>
                    <div class='response-header'>Conclusió del Supervisor</div>
                    <div class='explanation-text'>{response_text or "No disponible."}</div>
                </div>
                """,
            unsafe_allow_html=True,
        )


# --- ESTAT DE LA SESSIÓ DE STREAMLIT ---
if "battle_mode" not in st.session_state:
    st.session_state.battle_mode = False
if "battle_responses" not in st.session_state:
    st.session_state.battle_responses = {"A": None, "B": None}
if "battle_question" not in st.session_state:
    st.session_state.battle_question = ""
if "vote_casted" not in st.session_state:
    st.session_state.vote_casted = False
if "random_cases" not in st.session_state:
    case_titles = list(EXAMPLE_CASES.keys())
    st.session_state.random_cases = random.sample(case_titles, 3)

# --- BARRA LATERAL PER A LA CONFIGURACIÓ ---
st.sidebar.title("⚕️ MedAgent UI")
app_mode = st.sidebar.radio("Selecciona el Mode:", ("Consulta", "Batalla", "Resultats"), key="app_mode_selector")

if app_mode == "Consulta":
    # Show sliders only in Consulta mode
    num_experts = st.sidebar.slider(
        "Nombre d'experts a consultar:",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="Selecciona quants experts vols que analitzin la consulta",
    )

    diversity_option = st.sidebar.radio(
        "Diversitat dels Experts:",
        options=["Baixa", "Mitjana", "Alta"],
        index=1,  # Default to Mitjana
        help="Controla la diversitat del flux d'agents.",
    )
elif app_mode == "Batalla":
    # Show description for Batalla mode
    st.sidebar.markdown("### Mode Batalla")
    st.sidebar.info(
        """
        - Es generen automàticament dues respostes diferents
        - Cada resposta té una configuració aleatòria de nombre d'experts i diversitat
        - Pots comparar i triar la resposta que prefereixis
        """
    )
elif app_mode == "Classificació elo":
    # Show description for Elo mode
    st.sidebar.markdown("### Classificació Elo")
    st.sidebar.info(
        """
        - Aquesta secció mostra la classificació Elo dels agents basat en les votacions de les batalles.
        - Les puntuacions es calculen a partir de les votacions registrades en el fitxer CSV.
        """
    )
    # Load and display the Elo ratings
    if os.path.exists(csv_file):
        try:
            matches_df = pd.read_csv(csv_file)
            if not matches_df.empty:
                st.sidebar.markdown("### Classificació Elo Actual:")
                elo_ratings = pd.read_csv("elo_ratings_with_ci.csv")
                st.sidebar.dataframe(elo_ratings, use_container_width=True)
            else:
                st.sidebar.warning("No hi ha dades de batalles registrades.")
        except Exception as e:
            st.sidebar.error(f"Error carregant les dades de classificació: {e}")
    else:
        st.sidebar.warning("No s'ha trobat el fitxer de votacions.")


# --- LÒGICA PRINCIPAL DE LA UI ---
if app_mode == "Consulta":
    st.session_state.battle_mode = False
    st.session_state.vote_casted = False

    st.title("Mode de Consulta Mèdica")
    st.markdown(
        "Fes una pregunta o descriu un cas clínic. El sistema consultarà els experts seleccionats i un supervisor sintetitzarà les respostes."
    )

    # Default queries for Consulta mode
    st.subheader("Consultes d'exemple per a consulta de The Pitt (2025):")
    cols = st.columns(3)
    for i, title in enumerate(st.session_state.random_cases):
        with cols[i]:
            if st.button(title, key=f"consulta_case_{i+1}"):
                st.session_state.consulta_question = EXAMPLE_CASES[title]

    # Initialize consulta_question if not exists
    if "consulta_question" not in st.session_state:
        st.session_state.consulta_question = ""

    question = st.text_area(
        "Introdueix la teva pregunta o cas clínic aquí:",
        height=200,
        key="consulta_question",
        placeholder="Ex: Pacient de 45 anys amb dolor toràcic agut...",
    )

    if st.button("Consultar Experts", key="consulta_button", type="primary"):
        if question.strip():
            with st.spinner("Processant la teva consulta... Aquest procés pot trigar una mica."):
                try:
                    response = conversation_agentic_workflow(
                        question_text=question,
                        num_experts_to_select=num_experts,
                        diversity_option=diversity_option,
                    )
                    display_consulta_response(response)
                except Exception as e:
                    st.error(f"S'ha produït un error durant el processament: {e}")
        else:
            st.warning("Si us plau, introdueix una pregunta.")

elif app_mode == "Batalla":
    st.title("Mode Batalla: Quina Resposta Prefereixes?")
    st.markdown(
        "Introdueix una pregunta. Es generaran dues explicacions utilitzant diferents configuracions (nombre d'experts i llur diversitat). Tria la que consideris millor."
    )

    # Default queries for Batalla mode
    st.subheader("Consultes d'exemple per a batalla de The Pitt (2025):")
    cols = st.columns(3)
    for i, title in enumerate(st.session_state.random_cases):
        with cols[i]:
            if st.button(title, key=f"battle_case_{i+1}"):
                st.session_state.battle_q = EXAMPLE_CASES[title]

    # Initialize battle_q if not exists
    if "battle_q" not in st.session_state:
        st.session_state.battle_q = ""

    current_question_battle = st.text_area(
        "Introdueix la teva pregunta aquí per a la batalla:",
        height=150,
        key="battle_q_input",
        value=st.session_state.battle_q,
        placeholder="Ex: Quines són les causes més comunes del mal d'esquena?",
    )

    if st.button("Iniciar Batalla / Nova Pregunta", key="battle_start_button", type="primary"):
        # st.session_state.vote_casted = False
        if current_question_battle.strip():
            st.session_state.battle_question = current_question_battle
            st.session_state.battle_mode = True
            st.session_state.vote_casted = False
            st.session_state.user_vote = None
            st.session_state.battle_responses = {"A": None, "B": None}
            st.session_state.battle_configs = {"A": "N/A", "B": "N/A"}
            st.session_state.battle_configs_parsed = {"A": {}, "B": {}}

            with st.spinner("Generant respostes per a la batalla..."):
                try:
                    if random.random() < 0.15:
                        st.session_state.battle_configs_parsed["A"] = {
                            "experts": None,
                            "diversity": None,
                            "fine_tuned_model": random.choice(list(EXPERT_DEFINITIONS.values())),
                        }
                        response_A_obj = _call_single_expert_llm(
                            expert_model_id=st.session_state.battle_configs_parsed["A"]["fine_tuned_model"],
                            question_text=st.session_state.battle_question,
                            temperature=0.7,
                            is_benchmark_mode=False,
                        )
                        st.session_state.battle_responses["A"] = {
                            "explanation": response_A_obj.get("explanation", "No disponible."),
                            "answer": response_A_obj.get("answer", "No disponible."),
                        }
                        st.session_state.battle_configs["A"] = EXPERT_DEFINITIONS_REVERSED.get(
                            st.session_state.battle_configs_parsed["A"]["fine_tuned_model"], "N/A"
                        )
                    else:
                        num_experts_A = random.randint(1, 5)
                        diversity_option_A = random.choice(["Baixa", "Mitjana", "Alta"])
                        st.session_state.battle_configs_parsed["A"] = {
                            "experts": num_experts_A,
                            "diversity": diversity_option_A,
                            "fine_tuned_model": None,
                        }
                        st.session_state.battle_configs["A"] = (
                            f"experts_{num_experts_A}_diversitat_{diversity_option_A}"
                        )
                        response_A_obj = conversation_agentic_workflow(
                            question_text=st.session_state.battle_question,
                            num_experts_to_select=num_experts_A,
                            diversity_option=diversity_option_A,
                        )
                        st.session_state.battle_responses["A"] = response_A_obj.get(
                            "final_synthesis", "Error en generar resposta A."
                        )

                    # Generació de Resposta B
                    if random.random() < 0.15:
                        st.session_state.battle_configs_parsed["B"] = {
                            "experts": None,
                            "diversity": None,
                            "fine_tuned_model": random.choice(list(EXPERT_DEFINITIONS.values())),
                        }
                        while (
                            st.session_state.battle_configs_parsed["A"]["fine_tuned_model"]
                            == st.session_state.battle_configs_parsed["B"]["fine_tuned_model"]
                        ):  # Ensure different expert for B
                            st.session_state.battle_configs_parsed["B"]["fine_tuned_models"] = random.choice(
                                list(EXPERT_DEFINITIONS.values())
                            )

                        response_B_obj = _call_single_expert_llm(
                            expert_model_id=st.session_state.battle_configs_parsed["B"]["fine_tuned_model"],
                            question_text=st.session_state.battle_question,
                            temperature=0.7,
                            is_benchmark_mode=False,
                        )
                        st.session_state.battle_responses["B"] = {
                            "explanation": response_B_obj.get("explanation", "No disponible."),
                            "answer": response_B_obj.get("answer", "No disponible."),
                        }
                        st.session_state.battle_configs["B"] = EXPERT_DEFINITIONS_REVERSED.get(
                            st.session_state.battle_configs_parsed["B"]["fine_tuned_model"], "N/A"
                        )
                    else:
                        num_experts_B = random.randint(1, 5)
                        diversity_option_B = random.choice(["Baixa", "Mitjana", "Alta"])
                        # Ensure different configurations for B
                        while (
                            st.session_state.battle_configs_parsed["A"]["experts"] == num_experts_B
                            and st.session_state.battle_configs_parsed["A"]["diversity"] == diversity_option_B
                        ):
                            num_experts_B = random.randint(1, 5)
                            diversity_option_B = random.choice(["Baixa", "Mitjana", "Alta"])
                        st.session_state.battle_configs_parsed["B"] = {
                            "experts": num_experts_B,
                            "diversity": diversity_option_B,
                            "fine_tuned_model": None,
                        }
                        st.session_state.battle_configs["B"] = (
                            f"experts_{num_experts_B}_diversitat_{diversity_option_B}"
                        )
                        response_B_obj = conversation_agentic_workflow(
                            question_text=st.session_state.battle_question,
                            num_experts_to_select=num_experts_B,
                            diversity_option=diversity_option_B,
                        )
                        st.session_state.battle_responses["B"] = response_B_obj.get(
                            "final_synthesis", "Error en generar resposta B."
                        )

                except Exception as e:
                    st.error(f"Error generant respostes per a la batalla: {e}")
                    st.exception(e)
                    st.session_state.battle_mode = False
        else:
            st.warning("Si us plau, introdueix una pregunta per a la batalla.")
            st.session_state.battle_mode = False

    if (
        st.session_state.get("battle_mode", False)
        and st.session_state.get("battle_question", "").strip()
        and (
            st.session_state.get("battle_responses", {}).get("A")
            or st.session_state.get("battle_responses", {}).get("B")
        )
    ):
        st.markdown("---")

        vote_casted = st.session_state.get("vote_casted", False)
        user_vote = st.session_state.get("user_vote")

        col1, col2 = st.columns(2)

        with col1:
            card_a_style = "border: 2px solid #404040;"  # Default border from .battle-card
            if vote_casted and user_vote == "A":
                card_a_style = "border: 3px solid lightgreen !important; box-shadow: 0 0 10px lightgreen !important;"
            elif vote_casted and user_vote == "Tie":
                card_a_style = "border: 3px solid orange !important;"

            st.markdown(f"<div class='battle-card' style='{card_a_style}'>", unsafe_allow_html=True)
            st.markdown(f"<div class='battle-title'>Opció A</div>", unsafe_allow_html=True)
            display_battle_response(st.session_state.battle_responses["A"])
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            card_b_style = "border: 2px solid #404040;"  # Default border
            if vote_casted and user_vote == "B":
                card_b_style = "border: 3px solid lightgreen !important; box-shadow: 0 0 10px lightgreen !important;"
            elif vote_casted and user_vote == "Tie":
                card_b_style = "border: 3px solid orange !important;"

            st.markdown(f"<div class='battle-card' style='{card_b_style}'>", unsafe_allow_html=True)
            st.markdown(f"<div class='battle-title'>Opció B</div>", unsafe_allow_html=True)
            display_battle_response(st.session_state.battle_responses["B"])
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        config_a_details = st.session_state.battle_configs_parsed.get("A", {})
        config_b_details = st.session_state.battle_configs_parsed.get("B", {})

        label_A = "Prefereixo l'Opció A"
        label_B = "Prefereixo l'Opció B"
        label_Tie = "Empat"

        if vote_casted:
            if st.session_state.battle_configs_parsed["A"].get("fine_tuned_model") is not None:
                label_A = f"Opció A (Model fine-tuned: {st.session_state.battle_configs['A']})"
            else:
                label_A = f"Opció A (# Experts: {config_a_details.get('experts', 'N')}, Diversitat: {config_a_details.get('diversity', 'N')})"
            if st.session_state.battle_configs_parsed["B"].get("fine_tuned_model") is not None:
                label_B = f"Opció B (Model fine-tuned: {st.session_state.battle_configs['B']})"
            else:
                label_B = f"Opció B (# Experts: {config_b_details.get('experts', 'N')}, Diversitat: {config_b_details.get('diversity', 'N')})"
            label_Tie = "Empat"

        vote_col1_btn, vote_col2_btn, vote_col3_btn = st.columns(3)
        with vote_col1_btn:
            if st.button(label_A, key="vote_A_button", use_container_width=True, disabled=vote_casted):
                if not st.session_state.get("vote_casted", False):
                    record_vote(st.session_state.battle_configs["A"], st.session_state.battle_configs["B"], 1.0)
                    st.session_state.vote_casted = True
                    st.session_state.user_vote = "A"
                    st.rerun()

        with vote_col2_btn:
            if st.button(label_Tie, key="vote_Tie_button", use_container_width=True, disabled=vote_casted):
                if not st.session_state.get("vote_casted", False):
                    record_vote(st.session_state.battle_configs["A"], st.session_state.battle_configs["B"], 0.5)
                    st.session_state.vote_casted = True
                    st.session_state.user_vote = "Tie"
                    st.rerun()

        with vote_col3_btn:
            if st.button(label_B, key="vote_B_button", use_container_width=True, disabled=vote_casted):
                if not st.session_state.get("vote_casted", False):
                    record_vote(st.session_state.battle_configs["A"], st.session_state.battle_configs["B"], 0.0)
                    st.session_state.vote_casted = True
                    st.session_state.user_vote = "B"
                    st.rerun()

        st.markdown("---")

        if vote_casted:
            vote_message_intro = ""
            if user_vote == "A":
                vote_message_intro = "Has votat per: Opció A."
            elif user_vote == "B":
                vote_message_intro = "Has votat per: Opció B."
            elif user_vote == "Tie":
                vote_message_intro = "Has votat per: Empat."

            # Using st.markdown for more control over the success message appearance if needed
            st.markdown(
                f"<p style='color: lightgreen; text-align: center;'>{vote_message_intro} Les configuracions s'han revelat als botons.</p>",
                unsafe_allow_html=True,
            )
            st.info("Per a una nova batalla, introdueix una nova pregunta i prem 'Iniciar Batalla / Nova Pregunta'.")

elif app_mode == "Resultats":
    st.title("Resultats")
    st.markdown(
        "Aquesta secció mostra la classificació Elo dels agents basat en les votacions de les batalles. I també mostra gràfics de resultats i anàlisis de les batalles realitzades."
    )
    if os.path.exists(csv_file):
        # Load and display the elo ratings as a table
        try:
            matches_df = pd.read_csv(csv_file)
            if not matches_df.empty:
                st.markdown("### Classificació Elo:")
                update_elo_ratings()
                elo_ratings = pd.read_csv("elo_ratings_with_ci.csv")
                # Add MedQA results column to elo_ratings
                MedQA_results = {
                    "Medicina General": 65.4,
                    "Ciències Bàsiques": 68.5,
                    "Patologia i Farmacologia": 71.5,
                    "Cirurgia": 65.5,
                    "Pediatria i Ginecologia": 69.7,
                    "experts_1_diversitat_Baixa": 87.3,
                    "experts_2_diversitat_Baixa": 84.9,
                    "experts_3_diversitat_Baixa": 83.1,
                    "experts_4_diversitat_Baixa": 85.7,
                    "experts_5_diversitat_Baixa": 85.9,
                    "experts_1_diversitat_Mitjana": 86.9,
                    "experts_2_diversitat_Mitjana": 88.0,
                    "experts_3_diversitat_Mitjana": 88.3,
                    "experts_4_diversitat_Mitjana": 88.1,
                    "experts_5_diversitat_Mitjana": 87.6,
                    "experts_1_diversitat_Alta": 85.6,
                    "experts_2_diversitat_Alta": 88.4,
                    "experts_3_diversitat_Alta": 89.8,
                    "experts_4_diversitat_Alta": 88.6,
                    "experts_5_diversitat_Alta": 87.7,
                }
                elo_ratings_display = pd.DataFrame(
                    {
                        "Rank": range(1, len(elo_ratings) + 1),
                        "Model": elo_ratings["player"],
                        "Elo Rating": elo_ratings["rating"].round(0).astype(int),
                        "95% CI": [
                            f"+{upper:.1f}/-{lower:.1f}"
                            for upper, lower in zip(elo_ratings["ci_upper"], elo_ratings["ci_lower"])
                        ],
                        "MedQA": [MedQA_results.get(model, "N/A") for model in elo_ratings["player"]],
                    }
                )
                st.dataframe(
                    elo_ratings_display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                        "Model": st.column_config.TextColumn("Model"),
                        "Elo Rating": st.column_config.NumberColumn("Elo Rating", format="%d"),
                        "95% CI": st.column_config.TextColumn("95% CI"),
                        "MedQA": st.column_config.NumberColumn(label="MedQA"),
                    },
                )
            else:
                st.warning("No hi ha dades de batalles registrades.")
        except Exception as e:
            st.error(f"Error carregant les dades de classificació: {e}")
    else:
        st.warning("No s'ha trobat el fitxer de votacions.")
    st.markdown("---")
    st.markdown("## Gràfics de Resultats")
    # Add image from file
    st.image("elo_ratings_with_ci.png", caption="Classificació Elo amb IC", use_container_width=True)

    st.image("expected_scores_matrix.png", caption="Matriu de Puntuacions Esperades", use_container_width=True)
    st.image("agentic_workflow_benchmark.png", caption="Flux d'Agents Mèdics", use_container_width=True)
    st.image("model_benchmarks.png", caption="Comparació de Models", use_container_width=True)
    st.image(
        "dissimilarity_matrix_heatmap.png",
        caption="Matriu de Dissimilaritat de Diversitat de Respostes",
        use_container_width=True,
    )
    st.image(
        "diversity_accuracy_correlation.png",
        caption="Correlació entre Diversitat i Precisions",
        use_container_width=True,
    )
    st.image("expert_ranks.png", caption="Experts triats a cada fase", use_container_width=True)
    st.image(
        "temperature_optimization_Patologia_i_Farmacologia.png",
        caption="Precisió i taxa de refutació per a Patologia i Farmacologia",
        use_container_width=True,
    )


st.markdown("---")
st.sidebar.info(
    "Aquesta UI permet interactuar amb un sistema d'agents mèdics. "
    "Mode Consulta per a anàlisi detallada, Mode Batalla per a comparar explicacions."
)
