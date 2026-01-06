import requests
import json
import re
import time
import pandas as pd

# =========================
# CONFIG
# =========================
token = "sk-dac51a8d68f449ff9e8f7224fb43e149"
url = "https://k2vm-74.mde.epf.fr/api/chat/completions"

# Modèle testé (votre chatbot)
TEST_MODEL = "n8n"

# Modèle juge (vérificateur)
JUDGE_MODEL = "llama3.2:latest"  # <-- mettez ici le nom du modèle vérificateur dans OpenWebUI (ex: "qwen2.5", "llama3.1", etc.)

INPUT_CSV = "base_csv.csv"
OUTPUT_CSV = "eu_ai_act_qna_scored.csv"

SLEEP_BETWEEN_CALLS_S = 0.2
TIMEOUT_S = 120

# =========================
# API CALL (même style que vous)
# =========================
def chat_with_model(token, message="bonjour", model=TEST_MODEL):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": message}
        ],
        "temperature": 0.2,
    }
    response = requests.post(url, headers=headers, json=data, timeout=TIMEOUT_S)
    response.raise_for_status()
    return response.json()

def extract_content(openai_like_json):
    """
    Extrait le texte de réponse depuis une réponse type OpenAI:
    resp["choices"][0]["message"]["content"]
    """
    try:
        return openai_like_json["choices"][0]["message"]["content"].strip()
    except Exception:
        # En cas de format différent, on renvoie le JSON brut (utile pour debug)
        return json.dumps(openai_like_json, ensure_ascii=False)

# =========================
# JUDGE PROMPT + PARSING
# =========================
JUDGE_INSTRUCTIONS = (
    "Vous êtes un évaluateur strict d'un chatbot de compliance (EU AI Act). "
    "Vous devez noter la réponse du modèle par rapport à la réponse de référence.\n\n"
    "Donnez 2 notes de 0 à 5 :\n"
    "- fidelity_score : fidélité factuelle/normative vis-à-vis de la réponse de référence (0=faux, 5=parfaitement fidèle)\n"
    "- quality_score : clarté, structure, actionnabilité, prudence, absence d'hallucinations (0=mauvais, 5=excellent)\n\n"
    "Répondez EXCLUSIVEMENT en JSON compact sur une seule ligne, au format :\n"
    "{\"fidelity_score\":<0-5>,\"quality_score\":<0-5>,\"comment\":\"...\"}\n"
)

def parse_judge_json(text):
    """
    Rend l'évaluation robuste si le juge renvoie du texte autour du JSON.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Judge output not JSON: {text[:200]}")
        return json.loads(m.group(0))

def judge_answer(question, reference_answer, model_answer):
    judge_message = (
        f"{JUDGE_INSTRUCTIONS}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"RÉPONSE DE RÉFÉRENCE:\n{reference_answer}\n\n"
        f"RÉPONSE DU MODÈLE:\n{model_answer}\n"
    )

    judge_resp = chat_with_model(token, message=judge_message, model=JUDGE_MODEL)
    judge_text = extract_content(judge_resp)
    judge_obj = parse_judge_json(judge_text)

    fidelity = float(judge_obj.get("fidelity_score"))
    quality = float(judge_obj.get("quality_score"))
    comment = str(judge_obj.get("comment", "")).strip()

    return fidelity, quality, comment

# =========================
# MAIN
# =========================
def main():
    df = pd.read_csv(INPUT_CSV)

    # colonnes attendues
    if "question" not in df.columns or "reponse" not in df.columns:
        raise ValueError("Le CSV doit contenir au minimum les colonnes: 'question' et 'reponse' (référence).")

    # colonnes de sortie
    for col in ["model_response", "fidelity_score", "quality_score", "judge_comment", "score"]:
        if col not in df.columns:
            df[col] = "" if col in ["model_response", "judge_comment"] else None

    for index, row in df.iterrows():
        question = str(row["question"]).strip()
        reference_answer = str(row["reponse"]).strip()

        # Reprise: si déjà évalué, on saute
        if isinstance(row.get("model_response", ""), str) and row["model_response"].strip():
            continue

        print(f"\nRow {index}: {question}")

        # 1) Appel modèle testé
        test_resp_json = chat_with_model(token, message=question, model=TEST_MODEL)
        model_answer = extract_content(test_resp_json)
        df.at[index, "model_response"] = model_answer

        # 2) Appel modèle juge immédiatement
        fidelity, quality, comment = judge_answer(question, reference_answer, model_answer)
        df.at[index, "fidelity_score"] = fidelity
        df.at[index, "quality_score"] = quality
        df.at[index, "judge_comment"] = comment

        # Score agrégé (optionnel)
        df.at[index, "score"] = round(0.7 * fidelity + 0.3 * quality, 3)

        print(f"Model answer: {model_answer[:160]}{'...' if len(model_answer) > 160 else ''}")
        print(f"Scores => fidelity: {fidelity}, quality: {quality}, score: {df.at[index, 'score']}")
        print(f"Judge comment: {comment}")

        # Sauvegarde progressive (sécurité)
        df.to_csv(OUTPUT_CSV, index=False)

        time.sleep(SLEEP_BETWEEN_CALLS_S)

    print(f"\nTerminé. Résultats enregistrés dans: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
