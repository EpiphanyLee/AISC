# -*- coding: utf-8 -*-
import time
import re
import pandas as pd
from AISC_prompt_config_openai_4categories import classify_with_deepseek_openai_4categories


input_file = r".\AISC_deduplicate.xlsx"
merged_output_file = r".\merged_AISC_deduplicate.xlsx"
output_file = r".\classified_AISC_4categories.xlsx"


def _clean_cell(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _merge_series_values(series, prefer_longest=False):
    values = []
    for v in series:
        text = _clean_cell(v)
        if text and text not in values:
            values.append(text)
    if not values:
        return ""
    if prefer_longest:
        return max(values, key=len)
    return " | ".join(values)


def _normalize_doi(doi_text):
    doi = _clean_cell(doi_text).lower()
    doi = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
    return doi.strip()


def _normalize_title(title_text):
    title = _clean_cell(title_text).lower()
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def merge_duplicate_articles(df):
    df = df.copy()
    original_columns = list(df.columns)

    def build_merge_key(row):
        doi_key = _normalize_doi(row.get("DOI", ""))
        if doi_key:
            return f"doi::{doi_key}"
        title_key = _normalize_title(row.get("Title", ""))
        if title_key:
            return f"title::{title_key}"
        return f"row::{row.name}"

    df["_merge_key"] = df.apply(build_merge_key, axis=1)

    merged_rows = []
    for _, group in df.groupby("_merge_key", sort=False):
        merged = {}
        for col in original_columns:
            if col in ["Title", "Abstract Note"]:
                merged[col] = _merge_series_values(group[col], prefer_longest=True)
            else:
                merged[col] = _merge_series_values(group[col], prefer_longest=False)
        merged_rows.append(merged)

    return pd.DataFrame(merged_rows, columns=original_columns)


def _contains_any_keyword(text, keywords):
    return any(k in text for k in keywords)


def is_aisc_relevant(title, abstract):
    title_text = _clean_cell(title).lower()
    abstract_text = _clean_cell(abstract).lower()

    # Title: conservative dictionary (higher precision)
    ai_title_keywords = [
        "artificial intelligence", "machine learning", "deep learning",
        "neural network", "natural language processing", "large language model",
        "chatbot", "computer vision", "clinical decision support",
    ]
    smoking_title_keywords = [
        "smoking cessation", "quit smoking", "tobacco cessation",
        "nicotine dependence", "smoking abstinence", "relapse prevention",
        "quitline", "nicotine replacement therapy", "nrt",
    ]

    # Abstract: comprehensive dictionary (higher recall)
    ai_abstract_keywords = [
        "artificial intelligence", "machine learning", "deep learning",
        "neural network", "neural networks", "nlp", "natural language processing",
        "llm", "large language model", "large language models", "chatbot",
        "chatbots", "conversational agent", "conversational agents",
        "language model", "transformer model", "transformer", "bert",
        "gpt", "predictive model", "prediction model", "risk model",
        "risk stratification", "classification model", "regression model",
        "supervised learning", "unsupervised learning", "reinforcement learning",
        "ensemble learning", "transfer learning", "representation learning",
        "feature engineering", "feature selection", "model training",
        "model development", "model validation", "external validation",
        "algorithm", "algorithms", "data mining", "text mining",
        "computer vision", "image recognition", "speech recognition",
        "recommender system", "recommendation system", "decision support",
        "clinical decision support", "cdss", "digital phenotyping",
        "predictive analytics", "computational model", "automated detection",
        "automated classification", "random forest", "xgboost", "lightgbm",
        "svm", "support vector machine", "logistic regression",
        "naive bayes", "k nearest neighbor", "knn", "lstm", "cnn",
        "rnn", "gradient boosting", "bayesian network", "markov model",
        "expert system", "rule based model", "ehealth algorithm",
        "mhealth algorithm", "mobile health ai", "digital health ai",
    ]
    smoking_abstract_keywords = [
        "smoking cessation", "quit smoking", "quitting smoking", "smoking quit",
        "tobacco cessation", "tobacco control", "tobacco treatment",
        "tobacco dependence", "tobacco use disorder", "smoker", "smokers",
        "smoking", "smoking behavior", "smoking status", "smoking relapse",
        "smoking abstinence", "tobacco", "tobacco use", "nicotine",
        "nicotine dependence", "nicotine addiction", "nicotine withdrawal",
        "abstinence", "point prevalence abstinence", "continuous abstinence",
        "prolonged abstinence", "relapse", "relapse prevention",
        "relapse monitoring", "quit attempt", "quit attempts", "quit rate",
        "quit success", "cessation success", "cessation outcome",
        "smoking reduction", "cigarette consumption", "cigarettes per day",
        "cpd", "pack year", "pack-years", "carbon monoxide verified",
        "cotinine", "fagerstrom", "ftnd", "quitline", "nrt",
        "nicotine replacement therapy", "varenicline", "bupropion",
        "counseling", "behavioural counseling", "behavioral counseling",
        "motivational interviewing", "contingency management",
        "brief intervention", "cessation program", "smoking cessation program",
        "digital intervention", "mhealth cessation", "app based cessation",
        "mobile cessation", "text messaging cessation", "sms cessation",
        "vaping cessation", "e cigarette cessation", "e-cigarette cessation",
        "heated tobacco", "hookah", "waterpipe",
    ]

    ai_related = (
        _contains_any_keyword(title_text, ai_title_keywords)
        or _contains_any_keyword(abstract_text, ai_abstract_keywords)
    )
    smoking_related = (
        _contains_any_keyword(title_text, smoking_title_keywords)
        or _contains_any_keyword(abstract_text, smoking_abstract_keywords)
    )

    return ai_related and smoking_related, ai_related, smoking_related


def _split_categories(category_text):
    text = _clean_cell(category_text)
    if not text:
        return []
    items = [item.strip() for item in text.split(",") if item.strip()]
    return items


df = pd.read_excel(input_file)
original_count = len(df)

required_columns = ["Title", "Abstract Note"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"输入表格缺少必需的列: {missing_columns}")

df = merge_duplicate_articles(df)
print(f"重复文献合并完成: {original_count} -> {len(df)}")
df.to_excel(merged_output_file, index=False)
print(f"合并后的文件已保存到 {merged_output_file}")

df["Title"] = df["Title"].fillna("")
df["Abstract Note"] = df["Abstract Note"].fillna("")

df = df[(df["Title"].str.strip() != "") | (df["Abstract Note"].str.strip() != "")]

print(f"开始处理 {len(df)} 条记录（4分类多标签）...")
print("=" * 80)

results = []
for i, row in df.iterrows():
    title = row["Title"]
    abstract = row["Abstract Note"]

    print(f"\n处理第 {i + 1}/{len(df)} 条记录...")
    print(f"标题: {title[:100] if len(str(title)) > 100 else title}...")

    res = classify_with_deepseek_openai_4categories(title, abstract)
    results.append(res)

    print(f"分类结果: {res['category']} | 主分类: {res['primary_category']}")
    print("-" * 80)
    time.sleep(1)

results_df = pd.DataFrame(results)
df_output = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)

unclassified_mask = (
    df_output["category"].fillna("").str.strip().str.lower().eq("unclassified")
    | df_output["primary_category"].fillna("").str.strip().str.lower().eq("unclassified")
)

unclassified_df = df_output[unclassified_mask].copy()
excluded_df = pd.DataFrame(columns=list(df_output.columns) + ["exclude_reason"])

if len(unclassified_df) > 0:
    relevance_info = unclassified_df.apply(
        lambda row: is_aisc_relevant(row["Title"], row["Abstract Note"]),
        axis=1,
        result_type="expand",
    )
    relevance_info.columns = ["is_aisc_relevant", "ai_related", "smoking_related"]
    unclassified_df = pd.concat([unclassified_df, relevance_info], axis=1)

    excluded_df = unclassified_df[~unclassified_df["is_aisc_relevant"]].copy()
    if len(excluded_df) > 0:
        excluded_df["exclude_reason"] = excluded_df.apply(
            lambda row: (
                "No AI and smoking-cessation relevance"
                if (not row["ai_related"] and not row["smoking_related"])
                else ("No AI relevance" if not row["ai_related"] else "No smoking-cessation relevance")
            ),
            axis=1,
        )
        # Remove only unrelated records from the unclassified subset.
        df_output = df_output.drop(index=excluded_df.index, errors="ignore")
        print(f"已从 Unclassified 中剔除 {len(excluded_df)} 条非AISC记录，将保存到输出文件独立sheet")

    excluded_df = excluded_df.drop(columns=["is_aisc_relevant", "ai_related", "smoking_related"], errors="ignore")

with pd.ExcelWriter(output_file) as writer:
    df_output.to_excel(writer, index=False, sheet_name="classified")
    excluded_df.to_excel(writer, index=False, sheet_name="excluded_non_AISC")

    # Count primary and secondary categories in final retained records.
    primary_counts = (
        df_output["primary_category"]
        .fillna("Unclassified")
        .astype(str)
        .str.strip()
        .replace("", "Unclassified")
        .value_counts()
    )

    secondary_counter = {}
    for _, row in df_output.iterrows():
        primary = _clean_cell(row.get("primary_category", ""))
        labels = _split_categories(row.get("category", ""))
        for label in labels:
            if label.lower() == "unclassified":
                continue
            if primary and label.lower() == primary.lower():
                continue
            secondary_counter[label] = secondary_counter.get(label, 0) + 1

    secondary_counts = pd.Series(secondary_counter).sort_values(ascending=False)

    total = len(df_output) if len(df_output) > 0 else 1
    primary_df = pd.DataFrame(
        {
            "count_type": "primary",
            "category": primary_counts.index,
            "count": primary_counts.values,
            "percentage": (primary_counts.values / total * 100).round(1),
        }
    )
    secondary_df = pd.DataFrame(
        {
            "count_type": "secondary",
            "category": secondary_counts.index if len(secondary_counts) > 0 else [],
            "count": secondary_counts.values if len(secondary_counts) > 0 else [],
            "percentage": ((secondary_counts.values / total) * 100).round(1) if len(secondary_counts) > 0 else [],
        }
    )
    stats_df = pd.concat([primary_df, secondary_df], ignore_index=True)
    stats_df.to_excel(writer, index=False, sheet_name="category_stats")

print(f"\n全部处理完毕，结果已保存到 {output_file}")

print("\n" + "=" * 80)
print("主分类统计（primary_category）:")
print("=" * 80)
for category, count in primary_counts.items():
    print(f"  {category}: {count} 条 ({count / len(df_output) * 100:.1f}%)")

print("\n次分类统计（secondary，来自多标签中除主分类外的标签）:")
if len(secondary_counts) == 0:
    print("  无")
else:
    for category, count in secondary_counts.items():
        print(f"  {category}: {count} 条 ({count / len(df_output) * 100:.1f}%)")
