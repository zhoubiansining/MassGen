#!/usr/bin/env bash
set -euo pipefail

# ==================== 配置区域 ====================
# 修改为你的 pkl 路径和采样数量
PKL_PATH="/Users/zyc/Desktop/rag-zyc/original_survey_df.pkl"
SAMPLE_N=5                          # 取前 N 条样本
OUT_JSON="analysis_from_pkl.json"  # 生成的 analysis JSON

# ==================== 生成 analysis JSON ====================
python - <<'PY' "$PKL_PATH" "$SAMPLE_N" "$OUT_JSON"
import sys, json, pandas as pd, pathlib

pkl_path, sample_n, out_path = sys.argv[1], int(sys.argv[2]), pathlib.Path(sys.argv[3])
df = pd.read_pickle(pkl_path).head(sample_n)

datas = []
for idx, row in df.iterrows():
    datas.append({
        "id": str(idx),
        "title": row.get("title", ""),
        "abstract": row.get("abstract", ""),
        "authors": [],
        "year": None
    })

clusters = [{"cluster_id": 0, "paper_ids": [d["id"] for d in datas], "summary": "Sample cluster"}]
analysis = {"datas": datas, "clusters": clusters, "summary": "SciReviewGen sample"}

out_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] analysis json saved -> {out_path.resolve()}")
PY

# ==================== 跑全流程（跳过检索/分析，直接写作+评审） ====================
python main_all.py --analysis-json "$OUT_JSON"
