#!/bin/bash
# Qwen2.5-32B-Instruct trial: bf16, both GPUs (device_map=auto), layer 45/64, 1 seed x 100 respondents.
# Prerequisite: python build_vectors.py --spec qwen --device auto --bs 40 --max_new 120 --limit 24
# sensitivity probes use 30 respondents (--n 30) to save time (a 32B forward over 100 respondents takes ~10+ min).
PY=${PY:-python}
cd "$(dirname "$0")"
M="direct pe psii pe_noise"
$PY scale.py --spec qwen --device auto --layers 32 36 40 44 45 48 51 --sigmas 0.07
$PY followup.py sens --spec qwen --device auto --readout r1 --n 30
$PY followup.py run  --spec qwen --device auto --readout r1 --seeds 0 --configs $M
$PY followup.py sens --spec qwen --device auto --readout r3 --n 30 --bs 64
$PY followup.py run  --spec qwen --device auto --readout r3 --seeds 0 --bs 64 --configs $M
# extra, run afterwards: PSII - noise under R3 (vectors without noise)
# $PY followup.py run --spec qwen --device auto --readout r3 --seeds 0 --bs 64 --configs psii_nonoise
# Planned but NOT run (stage deadline): R1 extras psii_nonoise lin:a=2:s=0 lin:a=4:s=0 lin:a=4:s=0:rand on Qwen.
