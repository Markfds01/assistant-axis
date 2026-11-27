#!/bin/bash
# Llama follow-up driver. Stream A (R1) on GPU0, stream B (R3) on GPU1. Logs in $OUT/logs.
# Usage: bash run_followup_llama.sh A | B1 | B2 <best_alpha>
PY=${PY:-python}
LOG=${PSII_OUT:-../../../psii_outputs}/logs; mkdir -p $LOG
cd "$(dirname "$0")"
MAIN="direct pe pv psii psii_nonoise pe_noise"
SWEEP=""
for a in 0 1 2 4 8; do for s in 0 0.07 cal; do SWEEP="$SWEEP lin:a=$a:s=$s"; done; done
CTRL="lin:a=1:s=0:np lin:a=4:s=0:np lin:a=8:s=0:np lin:a=4:s=0:rand lin:a=8:s=0:rand"
case $1 in
  A)  $PY followup.py sens --readout r1 --device cuda:0
      $PY followup.py run --readout r1 --device cuda:0 --configs $MAIN $SWEEP $CTRL ;;
  B1) $PY scale.py --spec llama --device cuda:1 --layers 14 16 18 20 22 24 26 28 --sigmas 0.07
      $PY followup.py sens --readout r3 --device cuda:1
      $PY followup.py run --readout r3 --device cuda:1 --configs $MAIN ;;
  B2) # R3 sweep: alpha in {0 (=pe), best R1 alpha, 8} at sigma=0, plus PSII at calibrated sigma (in MAIN)
      $PY followup.py run --readout r3 --device cuda:1 --configs "lin:a=$2:s=0" "lin:a=8:s=0" ;;
esac
# R3 sweep was launched by hand as two chained jobs once the main streams finished (best R1 alpha with option mass >= 0.5 is 4):
#   GPU0: followup.py run --readout r3 --device cuda:0 --configs lin:a=4:s=0 lin:a=8:s=0
#   GPU1: followup.py run --readout r3 --device cuda:1 --configs lin:a=2:s=0 lin:a=4:s=0:rand
