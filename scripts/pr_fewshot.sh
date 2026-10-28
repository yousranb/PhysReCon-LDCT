#!/bin/sh
# Few-shot sweep for PhysReCon-LDCT (stage 2) + evaluation. Writes nothing it has already done.
#
#   sh pr_fewshot.sh norsvin 0        # Norsvin, fold 0
#   sh pr_fewshot.sh piglet ASIR40    # piglet held-out reconstruction ASIR40
#   sh pr_fewshot.sh piglet MBIR
#
# Edit the lists below to shorten the sweep. Every run uses the same test slices (TEST_STRIDE) so the
# few-shot curves compare like with like; the full-resolution benchmark run is done separately (README).
set -u
CKPT=${CKPT:-pr_runs/physrecon_main/final.pt}
STEPS=${STEPS:-300}
TEST_STRIDE=${TEST_STRIDE:-4}
DRAWS=${DRAWS:-"0 1 2"}
MODES=${MODES:-"z zfilm full"}

if [ "$1" = "norsvin" ]; then
  FOLD=${2:-0}
  DATA=data/norsvin; RECON=IR; DOSES="25 10"; KS=${KS:-"1 3 5"}
  RUNS=runs_fs_norsvin_f$FOLD; EXTRA="--exclude_subjects ER01035833 --fold $FOLD"
  TGTS="25_IR 10_IR"
else
  RECON=${2:-ASIR40}
  DATA=data/piglet; DOSES="5 10 25"; KS=${KS:-"1 2 5 10"}
  RUNS=runs_fs_piglet_$RECON; EXTRA="--fold 0"
  TGTS="5_$RECON 10_$RECON 25_$RECON"
fi
mkdir -p $RUNS

run() {   # name, args...
  name=$1; shift
  first=$(echo $TGTS | cut -d' ' -f1)
  if [ -f $RUNS/${name}_100_${RECON}_to_${first}_f*_s0/pred.npy ] 2>/dev/null; then
    echo "[done] $name"; return
  fi
  python -u pr_adapt.py --ckpt $CKPT --data $DATA --recon $RECON --doses $DOSES --runs $RUNS --name $name \
         --test_stride $TEST_STRIDE --steps $STEPS $EXTRA "$@"
}

run physrecon_zero_k0 --k 0 --mode zero
for k in $KS; do
  for d in $DRAWS; do
    for m in $MODES; do
      run physrecon_${m}_k${k}_d${d} --k $k --draw $d --mode $m
    done
  done
done
# training from scratch on the largest support set only (baseline)
lastk=$(echo $KS | awk '{print $NF}')
for d in $DRAWS; do run physrecon_scratch_k${lastk}_d${d} --k $lastk --draw $d --mode scratch --steps 2000; done

# evaluation: categories A-C (evaluate.py) and D-F (pr_eval.py), then few-shot table/curves
for t in $TGTS; do
  dirs=$(ls -d $RUNS/physrecon_*_to_${t}_f*_s0)
  python -u evaluate.py --runs $dirs --out results_$RUNS --baselines identity
  python -u pr_eval.py --runs $dirs --out results_${RUNS}_pr
done
python pr_collect.py --evaluate results_$RUNS/* --pr_eval results_${RUNS}_pr --out fewshot_$RUNS
