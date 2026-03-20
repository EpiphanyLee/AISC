# -*- coding: utf-8 -*-
import os
import re
import time
import hashlib
import pandas as pd
from AISC_prompt_config_openai_4categories import classify_with_deepseek_openai_4categories

try:
    import msvcrt
    _KEYBOARD_AVAILABLE = True
except ImportError:
    msvcrt = None
    _KEYBOARD_AVAILABLE = False


input_file = r".\AISC_deduplicate.xlsx"
merged_input_file = r".\merged_AISC_deduplicate.xlsx"
checkpoint_file = r".\AISC_classification_checkpoint.csv"
output_file = r".\classified_AISC_4categories_resume.xlsx"


def _clean_cell(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _normalize_doi(doi_text):
    doi = _clean_cell(doi_text).lower()
    doi = doi.replace("https://doi.org/", "").replace("http://dx.doi.org/", "")
    return doi.strip()


def _normalize_title(title_text):
    title = _clean_cell(title_text).lower()
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


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


def _build_record_id(row):
    doi = _normalize_doi(row.get("DOI", ""))
    title = _normalize_title(row.get("Title", ""))
    abstract = _clean_cell(row.get("Abstract Note", "")).lower().strip()
    base = f"{doi}|{title}|{abstract}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()


def _load_checkpoint(path):
    if not os.path.exists(path):
        return {}
    ckpt_df = pd.read_csv(path, dtype=str).fillna("")
    if "record_id" not in ckpt_df.columns:
        return {}
    ckpt_df = ckpt_df.drop_duplicates(subset=["record_id"], keep="last")
    result_map = {}
    for _, row in ckpt_df.iterrows():
        result_map[row["record_id"]] = {
            "category": row.get("category", "Unclassified"),
            "primary_category": row.get("primary_category", "Unclassified"),
            "reason": row.get("reason", "Unable to parse LLM output."),
        }
    return result_map


def _append_checkpoint(path, record_id, result):
    row_df = pd.DataFrame(
        [{
            "record_id": record_id,
            "category": result.get("category", "Unclassified"),
            "primary_category": result.get("primary_category", "Unclassified"),
            "reason": result.get("reason", ""),
        }]
    )
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    row_df.to_csv(path, mode="a", index=False, header=write_header, encoding="utf-8-sig")


def _poll_space_toggle(paused):
    if not _KEYBOARD_AVAILABLE:
        return paused
    while msvcrt.kbhit():
        key = msvcrt.getwch()
        if key == " ":
            paused = not paused
            if paused:
                print("已暂停。按空格继续...")
            else:
                print("已继续运行。")
    return paused


def _wait_while_paused(paused):
    while paused:
        time.sleep(0.2)
        paused = _poll_space_toggle(paused)
    return paused


if os.path.exists(merged_input_file):
    df = pd.read_excel(merged_input_file)
    print(f"使用已合并数据: {merged_input_file}")
else:
    df = pd.read_excel(input_file)
    df = merge_duplicate_articles(df)
    df.to_excel(merged_input_file, index=False)
    print(f"未找到已合并数据，已自动合并并保存: {merged_input_file}")

required_columns = ["Title", "Abstract Note"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"输入表格缺少必需的列: {missing_columns}")

df["Title"] = df["Title"].fillna("")
df["Abstract Note"] = df["Abstract Note"].fillna("")
df = df[(df["Title"].str.strip() != "") | (df["Abstract Note"].str.strip() != "")].reset_index(drop=True)
df["record_id"] = df.apply(_build_record_id, axis=1)

cached_results = _load_checkpoint(checkpoint_file)
print(f"已加载断点缓存: {len(cached_results)} 条")
print(f"本次待检查总记录: {len(df)} 条")
if _KEYBOARD_AVAILABLE:
    print("运行中按空格可暂停/继续（当前记录请求发出后会先等待其返回）。")
else:
    print("当前环境不支持空格键监听，请使用 Ctrl+C 中断后重启继续。")

results = []
paused = False
for i, row in df.iterrows():
    paused = _poll_space_toggle(paused)
    paused = _wait_while_paused(paused)

    title = row["Title"]
    abstract = row["Abstract Note"]
    record_id = row["record_id"]

    print(f"\n处理第 {i + 1}/{len(df)} 条...")
    if record_id in cached_results:
        res = cached_results[record_id]
        print("命中断点缓存，跳过API调用。")
    else:
        res = classify_with_deepseek_openai_4categories(title, abstract)
        _append_checkpoint(checkpoint_file, record_id, res)
        cached_results[record_id] = res
    results.append(res)
    print(f"分类结果: {res['category']} | 主分类: {res['primary_category']}")
    for _ in range(10):
        time.sleep(0.1)
        paused = _poll_space_toggle(paused)
        paused = _wait_while_paused(paused)

results_df = pd.DataFrame(results)
df_output = pd.concat([df.drop(columns=["record_id"]).reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
df_output.to_excel(output_file, index=False)
print(f"\n断点续跑分类完成，结果已保存到 {output_file}")
