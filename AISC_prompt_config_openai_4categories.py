# -*- coding: utf-8 -*-
import os
import re
import pandas as pd
from openai import OpenAI


def _get_deepseek_client():
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing DEEPSEEK_API_KEY. Please set environment variable DEEPSEEK_API_KEY first."
        )
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


four_category_prompt = """
You are an expert in medical literature classification and clinical informatics.

Please read the following study information (title + abstract), and assign the research article to all relevant categories (select all that apply) from the list below, for the domain of AI in smoking cessation:

- Diagnosis: Identification, detection, screening, or classification related to smoking behavior, tobacco use status, nicotine dependence, or cessation-relevant conditions; includes studies on diagnostic tools, biomarkers, questionnaires, imaging, AI models for diagnosis, early detection, and screening programs.
- Intervention: Healthcare interventions such as smoking cessation treatment, counseling, cessation programs, pharmacological support (e.g., NRT), quitline services, digital interventions, behavioral or educational programs, or assessment of intervention effectiveness or safety.
- Monitor: Studies on monitoring, follow-up, surveillance, adherence, longitudinal tracking, remote monitoring, relapse monitoring, follow-up abstinence assessment, or implementation of clinical decision support for ongoing cessation management.
- Predict: Risk prediction, outcome prediction, predictive modeling, survival analysis, identification of predictive factors, quit success prediction, relapse risk prediction, abstinence prediction, long-term outcomes, or evaluation of cessation trajectories.

Please provide a structured report in the following format:

### Classification:
- category: (list all relevant categories separated by commas, e.g., Diagnosis, Predict)
- primary_category: (choose the single most relevant category)

### Reason:
- reason: (Concise, professional, and specific explanation for your choices. Reference key elements such as study design, methods, data source, patient population, outcomes, AI/ML algorithms, and clinical focus. If the study covers multiple categories, briefly justify each.)

Additional instructions:
- If the abstract is not relevant to any category above, write 'Unclassified' for both category and primary_category, and explain the reason.
- Respond strictly in the specified format and in English.
- Ensure your classification and rationale are suitable for systematic review, evidence mapping, and clinical guideline development in smoking cessation research.
- primary_category rule: Choose the single most relevant category that reflects the study's main objective and primary outcome. If multiple categories apply, prioritize the category emphasized in the title/aims and tied to the primary endpoint. If still ambiguous, use the following precedence order to break ties: Intervention > Diagnosis > Predict > Monitor.
- Missing/unclear information: If critical details are not available in the title/abstract (e.g., AI/ML method not specified, unclear study design, unclear clinical setting or endpoints), still classify using the best available evidence from the text, and explicitly state the limitation(s) in the Reason (e.g., "AI method not specified in abstract" / "study design unclear from abstract").

Study Title:
{title}

Study Abstract:
{abstract}
"""


def parse_llm_output(content):
    result = {
        "category": "Unclassified",
        "primary_category": "Unclassified",
        "reason": "Unable to parse LLM output.",
    }

    if not content or not content.strip():
        return result

    m_cat = re.search(r"(?im)^\s*[-*]?\s*category\s*:\s*(.+?)\s*$", content)
    valid_categories = ["Diagnosis", "Intervention", "Monitor", "Predict"]

    def normalize_category_list(raw_text):
        text = raw_text.lower()
        found = []
        if (
            "diagnos" in text
            or "detect" in text
            or "screen" in text
            or "dependence" in text
            or "nicotine dependence" in text
        ):
            found.append("Diagnosis")
        if (
            "intervention" in text
            or "treatment" in text
            or "therapy" in text
            or "cessation program" in text
            or "counseling" in text
            or "nrt" in text
            or "quitline" in text
            or "digital intervention" in text
        ):
            found.append("Intervention")
        if (
            "monitor" in text
            or "follow-up" in text
            or "follow up" in text
            or "surveillance" in text
            or "follow-up abstinence" in text
            or "adherence" in text
            or "relapse monitoring" in text
        ):
            found.append("Monitor")
        if (
            "predict" in text
            or "risk" in text
            or "prognos" in text
            or "quit success" in text
            or "relapse risk" in text
            or "abstinence prediction" in text
        ):
            found.append("Predict")
        if "unclassified" in text:
            return ["Unclassified"]
        deduped = []
        for item in found:
            if item not in deduped:
                deduped.append(item)
        return deduped if deduped else ["Unclassified"]

    if m_cat:
        cat_list = normalize_category_list(m_cat.group(1).strip())
        result["category"] = ", ".join(cat_list)
    else:
        cat_list = normalize_category_list(content)
        result["category"] = ", ".join(cat_list)

    m_primary = re.search(r"(?im)^\s*[-*]?\s*primary_category\s*:\s*(.+?)\s*$", content)
    if m_primary:
        p_raw = m_primary.group(1).strip().lower()
        if (
            "diagnos" in p_raw
            or "detect" in p_raw
            or "screen" in p_raw
            or "dependence" in p_raw
            or "nicotine dependence" in p_raw
        ):
            result["primary_category"] = "Diagnosis"
        elif (
            "intervention" in p_raw
            or "treatment" in p_raw
            or "therapy" in p_raw
            or "cessation program" in p_raw
            or "counseling" in p_raw
            or "nrt" in p_raw
            or "quitline" in p_raw
            or "digital intervention" in p_raw
        ):
            result["primary_category"] = "Intervention"
        elif (
            "monitor" in p_raw
            or "follow-up" in p_raw
            or "surveillance" in p_raw
            or "follow-up abstinence" in p_raw
            or "adherence" in p_raw
            or "relapse monitoring" in p_raw
        ):
            result["primary_category"] = "Monitor"
        elif (
            "predict" in p_raw
            or "risk" in p_raw
            or "prognos" in p_raw
            or "quit success" in p_raw
            or "relapse risk" in p_raw
            or "abstinence prediction" in p_raw
        ):
            result["primary_category"] = "Predict"
        elif "unclassified" in p_raw:
            result["primary_category"] = "Unclassified"
    else:
        first_cat = result["category"].split(",")[0].strip()
        result["primary_category"] = first_cat if first_cat in valid_categories else "Unclassified"

    m_reason = re.search(
        r"(?is)^\s*[-*]?\s*reason\s*:\s*(.+?)(?=^\s*#{0,3}\s*\w+\s*:|\Z)",
        content,
        re.MULTILINE,
    )
    if m_reason:
        result["reason"] = re.sub(r"\s+", " ", m_reason.group(1).strip())

    return result


def classify_with_deepseek_openai_4categories(title, abstract, prompt_template=four_category_prompt):
    title = str(title).strip() if title and pd.notna(title) else ""
    abstract = str(abstract).strip() if abstract and pd.notna(abstract) else ""

    if not title and not abstract:
        return {
            "category": "Unclassified",
            "primary_category": "Unclassified",
            "reason": "No title or abstract provided.",
        }

    user_message = prompt_template.format(title=title, abstract=abstract)

    try:
        client = _get_deepseek_client()
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "You are precise and follow output format strictly.",
                },
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content
        return parse_llm_output(content)
    except Exception as e:
        return {
            "category": "Unclassified",
            "primary_category": "Unclassified",
            "reason": f"Error: {str(e)}",
        }
