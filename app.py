import streamlit as st
import random
import pandas as pd
import os
from langgraph_conversa import conversation_agentic_workflow

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


def record_vote(model_a_config_str: str, model_b_config_str: str, vote_score: float):
    """Appends a vote record to the CSV file."""
    new_vote = pd.DataFrame(
        [
            {
                "model_A_config": model_a_config_str,
                "model_B_config": model_b_config_str,
                "score_A_vs_B": vote_score,  # 1 for A, 0.5 for Tie, 0 for B
                "timestamp": pd.Timestamp.now(),
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
    else:
        raise ValueError("La resposta del supervisor provisional no està disponible. SIMBA")

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
    else:
        raise ValueError("Les respostes revisades dels experts no estan disponibles. NALA")

    if "final_synthesis" in response_data and response_data["final_synthesis"]:
        st.markdown(
            f"""
            <div class="response-container">
                <div class="response-header">Síntesi Final</div>
                <div class='explanation-text'>{response_data["final_synthesis"] or "No disponible."}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def display_battle_response(response_text, title):
    """Mostra la resposta per a una de les opcions de la Batalla."""
    st.markdown(f"<div class='battle-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='explanation-text'>{response_text or 'No s''ha pogut generar una explicació.'}</div>",
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


# --- BARRA LATERAL PER A LA CONFIGURACIÓ ---
st.sidebar.title("⚕️ MedAgent UI")
app_mode = st.sidebar.radio("Selecciona el Mode:", ("Consulta", "Batalla de Respostes"), key="app_mode_selector")

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
else:
    # Show description for Batalla mode
    st.sidebar.markdown("### Mode Batalla")
    st.sidebar.info(
        """
        - Es generen automàticament dues respostes diferents
        - Cada resposta té una configuració aleatòria de nombre d'experts i diversitat
        - Pots comparar i triar la resposta que prefereixis
        """
    )


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
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Anèmia de cèl·lules falciformes", key="default_query1"):
            st.session_state.consulta_question = "A 29-year-old Black female with known sickle cell disease presents to the ED in acute vaso-occlusive crisis, reporting 10/10 pain. Security and two orderlies bring her in, suspecting drug-seeking behavior due to her repeated visits and agitation. Initial triage staff delay opioid administration, questioning the legitimacy of her pain."

    with col2:
        if st.button("Donació d'òrgans", key="default_query2"):
            st.session_state.consulta_question = "A 19-year-old male is found unresponsive by his parents and is declared brain-dead following a fentanyl overdose. The medical team approaches the family to discuss organ donation. The parents are divided: one wishes to proceed, believing it will give meaning to their loss, while the other cannot accept the idea. The team must navigate the ethical complexities of consent, grief, and the urgency of organ procurement."

    with col3:
        if st.button("Intubació o fi de vida", key="default_query3"):
            st.session_state.consulta_question = "An 89-year-old male patient with dementia and respiratory distress arrives at the ER. The medical team must decide whether to intubate him, which would violate his advance directive stating he does not want to be placed on a ventilator. The patient's son insists on honoring the directive, while the daughter cannot accept a decision that results in her father's death. Ultimately, the son defers to the daughter, and the doctor intubates the patient—knowing it goes against the patient's stated wishes."

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

elif app_mode == "Batalla de Respostes":
    st.title("Mode Batalla: Quina Resposta Prefereixes?")
    st.markdown(
        "Introdueix una pregunta. Es generaran dues explicacions utilitzant diferents configuracions (nombre d'experts i llur diversitat). Tria la que consideris millor."
    )

    # Default queries for Batalla mode
    st.subheader("Consultes d'exemple per a batalla de The Pitt (2025):")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Anèmia de cèl·lules falciformes", key="default_battle1"):
            st.session_state.battle_q = "A 29-year-old Black female with known sickle cell disease presents to the ED in acute vaso-occlusive crisis, reporting 10/10 pain. Security and two orderlies bring her in, suspecting drug-seeking behavior due to her repeated visits and agitation. Initial triage staff delay opioid administration, questioning the legitimacy of her pain."

    with col2:
        if st.button("Donació d'òrgans", key="default_battle2"):
            st.session_state.battle_q = "A 19-year-old male is found unresponsive by his parents and is declared brain-dead following a fentanyl overdose. The medical team approaches the family to discuss organ donation. The parents are divided: one wishes to proceed, believing it will give meaning to their loss, while the other cannot accept the idea. The team must navigate the ethical complexities of consent, grief, and the urgency of organ procurement."

    with col3:
        if st.button("Intubació o fi de vida", key="default_battle3"):
            st.session_state.battle_q = "An 89-year-old male patient with dementia and respiratory distress arrives at the ER. The medical team must decide whether to intubate him, which would violate his advance directive stating he does not want to be placed on a ventilator. The patient's son insists on honoring the directive, while the daughter cannot accept a decision that results in her father's death. Ultimately, the son defers to the daughter, and the doctor intubates the patient—knowing it goes against the patient's stated wishes."

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
                    # Generació de Resposta A
                    num_experts_A = random.randint(1, 5)
                    diversity_option_A = random.choice(["Baixa", "Mitjana", "Alta"])
                    st.session_state.battle_configs_parsed["A"] = {
                        "experts": num_experts_A,
                        "diversity": diversity_option_A,
                    }
                    st.session_state.battle_configs["A"] = f"experts_{num_experts_A}_diversity_{diversity_option_A}"

                    response_A_obj = conversation_agentic_workflow(
                        question_text=st.session_state.battle_question,
                        num_experts_to_select=num_experts_A,
                        diversity_option=diversity_option_A,
                    )
                    st.session_state.battle_responses["A"] = response_A_obj.get(
                        "final_synthesis", "Error en generar resposta A."
                    )

                    # Generació de Resposta B
                    num_experts_B = random.randint(1, 5)
                    diversity_option_B = random.choice(["Baixa", "Mitjana", "Alta"])
                    while (
                        num_experts_A == num_experts_B and diversity_option_A == diversity_option_B
                    ):  # Ensure different configs
                        num_experts_B = random.randint(1, 5)
                        diversity_option_B = random.choice(["Baixa", "Mitjana", "Alta"])
                    st.session_state.battle_configs_parsed["B"] = {
                        "experts": num_experts_B,
                        "diversity": diversity_option_B,
                    }
                    st.session_state.battle_configs["B"] = f"experts_{num_experts_B}_diversity_{diversity_option_B}"

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
            display_battle_response(st.session_state.battle_responses["A"], "")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            card_b_style = "border: 2px solid #404040;"  # Default border
            if vote_casted and user_vote == "B":
                card_b_style = "border: 3px solid lightgreen !important; box-shadow: 0 0 10px lightgreen !important;"
            elif vote_casted and user_vote == "Tie":
                card_b_style = "border: 3px solid orange !important;"

            st.markdown(f"<div class='battle-card' style='{card_b_style}'>", unsafe_allow_html=True)
            st.markdown(f"<div class='battle-title'>Opció B</div>", unsafe_allow_html=True)
            display_battle_response(st.session_state.battle_responses["B"], "")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        config_a_details = st.session_state.battle_configs_parsed.get("A", {})
        config_b_details = st.session_state.battle_configs_parsed.get("B", {})

        label_A = "Prefereixo l'Opció A"
        label_B = "Prefereixo l'Opció B"
        label_Tie = "Empat"

        if vote_casted:
            label_A = f"Opció A (# Experts: {config_a_details.get('experts', 'N')}, Diversitat: {config_a_details.get('diversity', 'N')})"
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

st.markdown("---")
st.sidebar.info(
    "Aquesta UI permet interactuar amb un sistema d'agents mèdics. "
    "Mode Consulta per a anàlisi detallada, Mode Batalla per a comparar explicacions."
)
