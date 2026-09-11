# reports/

Generated documents, committed so a reader can see the conclusions without cluster access.
They are outputs, not sources — regenerate rather than hand-edit.

## REPORT_L0.md

The L0 model x precision bake-off: 7 arms scored by the dataset's own scorer over the same
items, with paired McNemar tests and per-arm distance calibration.

```bash
python scripts/l0_report.py \
  --dir   <L0 run dir>          \  # E_cosmos3edge_{bf16,int4,nf4}_*, E_gemma4e4b_q4kxl_*
  --round1 <round-1 run dir>    \  # E/ and E_mcq/ from the first bake-off
  --trt   <TensorRT run dir>    \  # E_awq_trt_*  (from scripts/l0_trt_eval.py)
  --data  <spatial-qa dataset>  \
  --out   reports/REPORT_L0.md
```

The run directories are not in this repo: they hold per-item model outputs over a gated
dataset. The report carries every aggregate needed to check the conclusions, and
`scripts/l0_trt_eval.py` plus `scripts/l0_cosmos_int4_sim.py` reproduce the arms.
