# survey-agents
A Multi-Agent System for Survey Generation

## per-paper 多 API 启动示例（文章级并行，每篇固定一个 API Key）

```bash
python main_all.py \
  --dataset-pkl /Users/zyc/Desktop/rag-zyc/original_survey_df.pkl \
  --use-test-only \
  --sample-n -1 \
  --skip-n 0 \
  --per-paper \
  --api-keys-file api_keys.txt
```

- `--sample-n -1`：使用全部样本，可改为具体条数。
- `--skip-n`：跳过前 N 条（如中断后从第 177 条继续则设 176）。
- `--use-test-only`：仅使用 `split == 'test'` 的样本，可按需去掉。
- `--api-keys-file`：包含多个 `sk-` Key 的文件（支持 `KEY = sk-...` 行或 JSON 数组），按 API 数并行，单个 API 串行多篇。
