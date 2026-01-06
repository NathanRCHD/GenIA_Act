import requests
import json
import re

token = "sk-dac51a8d68f449ff9e8f7224fb43e149"

def chat_with_model(token, message="bonjour"):
    url = 'https://k2vm-74.mde.epf.fr/api/chat/completions'
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    data = {
      "model": "n8n",
      "messages": [
        {
          "role": "user",
          "content": message
        }
      ]
    }
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def _headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h

def call_openwebui_chat(model: str, user_message: str, system_message: Optional[str] = None) -> str:
    """
    Calls OpenWebUI using an OpenAI-like chat.completions endpoint.
    Returns assistant text.
    """
    url = "https://k2vm-74.mde.epf.fr/api/chat/completions"

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }

    MAX_RETRIES = 3

    

    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(url, headers=_headers(), json=payload, timeout=REQUEST_TIMEOUT_S)
            r.raise_for_status()
            data = r.json()

            # OpenAI-like shape:
            # data["choices"][0]["message"]["content"]
            content = data["choices"][0]["message"]["content"]
            return content.strip()

        except Exception as e:
            last_err = e
            if attempt < MAX_RETRIES:
                time.sleep(1.0 * attempt)
            else:
                raise RuntimeError(f"OpenWebUI call failed after {MAX_RETRIES} tries: {last_err}") from last_err
            
JUDGE_SYSTEM = (
    "Vous êtes un évaluateur strict d'un chatbot de compliance (EU AI Act). "
    "Vous devez noter la réponse du modèle par rapport à la réponse de référence.\n\n"
    "Définitions des scores (0 à 5):\n"
    "- fidelity_score: fidélité factuelle et normative par rapport à la référence (0 = faux, 5 = parfaitement fidèle)\n"
    "- quality_score: clarté, structure, actionnabilité, absence d'hallucinations, bonne mise en garde (0 = mauvais, 5 = excellent)\n\n"
    "Répondez EXCLUSIVEMENT en JSON compact sur une seule ligne, au format:\n"
    "{\"fidelity_score\":<0-5>,\"quality_score\":<0-5>,\"comment\":\"...\"}\n"
)

def parse_judge_json(text: str) -> Dict[str, Any]:
    """
    Extract JSON from judge output (robust to extra text).
    """
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Extract first {...}
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError(f"Judge output is not JSON: {text[:200]}")
        return json.loads(m.group(0))


def judge_pair(reference_answer: str, model_answer: str, question: str) -> Tuple[float, float, str]:
    user_msg = (
        "Évaluez la réponse du modèle.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"RÉPONSE DE RÉFÉRENCE:\n{reference_answer}\n\n"
        f"RÉPONSE DU MODÈLE:\n{model_answer}\n"
    )
    raw = call_openwebui_chat(JUDGE_MODEL_NAME, user_msg, system_message=JUDGE_SYSTEM)
    obj = parse_judge_json(raw)

    fidelity = float(obj.get("fidelity_score"))
    quality = float(obj.get("quality_score"))
    comment = str(obj.get("comment", "")).strip()
    return fidelity, quality, comment


# =========================
# 3) MAIN LOOP
# =========================

def main():
    df = pd.read_csv(INPUT_CSV)

    # Ensure columns exist
    if "question" not in df.columns or "reference_answer" not in df.columns:
        raise ValueError("CSV must contain columns: question, reference_answer")

    # Prepare output columns
    if "model_response" not in df.columns:
        df["model_response"] = ""
    if "fidelity_score" not in df.columns:
        df["fidelity_score"] = None
    if "quality_score" not in df.columns:
        df["quality_score"] = None
    if "judge_comment" not in df.columns:
        df["judge_comment"] = ""

    # Optional: a short system prompt for your chatbot behavior
    chatbot_system = (
        "Vous êtes un assistant de compliance centré sur l'EU AI Act. "
        "Répondez clairement, structurez la réponse, et si vous n'êtes pas certain, dites-le."
    )

    for i, row in df.iterrows():
        q = str(row["question"]).strip()
        ref = str(row["reference_answer"]).strip()

        # Skip if already processed (useful for resume)
        if isinstance(row.get("model_response", ""), str) and row["model_response"].strip():
            continue

        # 1) Model answer
        model_ans = call_openwebui_chat(MODEL_NAME, q, system_message=chatbot_system)
        df.at[i, "model_response"] = model_ans

        # 2) Judge scores
        fidelity, quality, comment = judge_pair(ref, model_ans, q)
        df.at[i, "fidelity_score"] = fidelity
        df.at[i, "quality_score"] = quality
        df.at[i, "judge_comment"] = comment

        # Persist at each step (safe)
        df.to_csv(OUTPUT_CSV, index=False)

        time.sleep(SLEEP_BETWEEN_CALLS_S)

    print(f"Done. Scored CSV saved to: {OUTPUT_CSV}")

#load csv and iterate over rows to chat with model
import pandas as pd
df = pd.read_csv('eu_ai_act_qna_gold.csv')
for index, row in df.iterrows():
    print(f"Row {index}: {row['question']}")
    response = chat_with_model(token, message=row['question'])
    print(f"Response: {response}")