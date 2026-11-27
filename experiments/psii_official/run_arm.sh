#!/usr/bin/env bash
# Run one arm of the PSII reproduction on Llama-3.1-8B-Instruct with the authors' run.py.
# usage: run_arm.sh GPU ARM SEED
# Mirrors the argument handling of the authors' run.sh (llama3.1-8b branch, same layers, coef, sigma).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
GPU=$1; ARM=$2; SEED=$3
PSII_DIR=${PSII_DIR:-$HERE/../../../psii_official}
OUT=${OUT:-$HERE/../../../psii_outputs/official}
PY=${PY:-python}
export PSII_LLAMA_PATH=${PSII_LLAMA_PATH:-$(ls -d $HOME/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/* | head -1)}
export PSII_VLLM_GPU_UTIL=${PSII_VLLM_GPU_UTIL:-0.45}
MODEL=llama3.1-8b
Q="271:16,285:16,263:21,289:21,260:26,273:26,275:26,279:26,281:26,284:26,286:26,287:26,288:26"
Q70="9999:22"
NOISE="--noise_std 0.07"
STEER="--coef 2.0 --steering_type all"
DV="--demographic_vectors_dir demographic_vectors/vectors_${MODEL}"
VV="--value_vectors_dir value_vectors/vectors_${MODEL}"
case "$ARM" in
  # ---- paper baselines / main
  direct)          A="--method direct --temperature 0.7" ;;
  pe)              A="--method prompt_engineering" ;;
  psii)            A="--method value_vector,demographic_vectors,prompt_engineering $DV $VV $STEER --vector_q_codes $Q $NOISE" ;;
  # ---- Table 2 ablations (as in run.sh)
  no_value)        A="--method demographic_vectors,prompt_engineering $DV $STEER --vector_q_codes $Q $NOISE" ;;
  no_demo)         A="--method value_vector,prompt_engineering $VV" ;;
  no_profile)      A="--method value_vector,demographic_vectors $DV $VV $STEER --vector_q_codes $Q $NOISE" ;;
  no_noise)        A="--method value_vector,demographic_vectors,prompt_engineering $DV $VV $STEER --vector_q_codes $Q" ;;
  layer70)         A="--method value_vector,demographic_vectors,prompt_engineering $DV $VV $STEER --vector_q_codes $Q70 $NOISE" ;;
  # ---- controls (ours)
  noise_only_value) A="--method value_vector,demographic_vectors,prompt_engineering $DV $VV $STEER --vector_q_codes $Q $NOISE --vector_control zero" ;;
  noise_only)      A="--method demographic_vectors,prompt_engineering $DV $STEER --vector_q_codes $Q $NOISE --vector_control zero" ;;
  demo_only)       A="--method demographic_vectors,prompt_engineering $DV $STEER --vector_q_codes $Q" ;;
  random_value)    A="--method value_vector,demographic_vectors,prompt_engineering $DV $VV $STEER --vector_q_codes $Q $NOISE --vector_control random" ;;
  random)          A="--method demographic_vectors,prompt_engineering $DV $STEER --vector_q_codes $Q $NOISE --vector_control random" ;;
  random_nonoise)  A="--method demographic_vectors,prompt_engineering $DV $STEER --vector_q_codes $Q --vector_control random" ;;
  # PE in 5 random languages (what the value_vector method switches on), no vectors, no noise
  pe_multi)        A="--method multilingual,prompt_engineering --no_vllm" ;;
  # zero-vector noise control with random languages but WITHOUT the value vector
  noise_only_multi) A="--method multilingual,demographic_vectors,prompt_engineering $DV $STEER --vector_q_codes $Q $NOISE --vector_control zero" ;;
  pe_novllm)       A="--method prompt_engineering --no_vllm" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
mkdir -p "$OUT/$ARM" "$OUT/logs"
cd "$PSII_DIR"
echo "[$(date -u +%FT%T)] START $ARM seed=$SEED gpu=$GPU"
CUDA_VISIBLE_DEVICES=$GPU $PY run.py --use_local_model --model_name $MODEL --output_dir "$OUT/$ARM" --seed $SEED ${QSTRIDE:+--q_stride $QSTRIDE} ${EXTRA:-} $A
echo "[$(date -u +%FT%T)] DONE $ARM seed=$SEED"
