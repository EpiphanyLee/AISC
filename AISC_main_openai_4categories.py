# -*- coding: utf-8 -*-
import time
import pandas as pd
from AISC_prompt_config_openai_4categories import classify_with_deepseek_openai_4categories


input_file = r".\AISC_deduplicate.xlsx"
output_file = r".\classified_AISC_4categories.xlsx"


df = pd.read_excel(input_file)

required_columns = ["Title", "Abstract Note"]
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"输入表格缺少必需的列: {missing_columns}")

df["Title"] = df["Title"].fillna("")
df["Abstract Note"] = df["Abstract Note"].fillna("")

df = df[(df["Title"].str.strip() != "") | (df["Abstract Note"].str.strip() != "")]
df = df.head(10)

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
df_output.to_excel(output_file, index=False)

print(f"\n全部处理完毕，结果已保存到 {output_file}")

print("\n" + "=" * 80)
print("多标签分类统计（category 字段）:")
print("=" * 80)
category_counts = results_df["category"].value_counts()
for category, count in category_counts.items():
    print(f"  {category}: {count} 条 ({count / len(results_df) * 100:.1f}%)")

print("\n主分类统计（primary_category）:")
primary_counts = results_df["primary_category"].value_counts()
for primary, count in primary_counts.items():
    print(f"  {primary}: {count} 条 ({count / len(results_df) * 100:.1f}%)")
