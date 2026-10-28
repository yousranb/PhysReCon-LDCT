#!/bin/sh
# Conference v3 additions (Norsvin IR):
#  - PhysReCon trained from scratch on the Norsvin training pigs (same pigs as the adaptation support set)
#  - PhysHVS-LDCT (CADM) trained on piglet, sampled on Norsvin without adaptation (zero-shot)
#  - one unified evaluation of every Table-1 method (same slices, same reference)
# 2 GPUs: sh run_conf_v3.sh        1 GPU: G2=0 sh run_conf_v3.sh
G2=${G2:-1}
CADM_PIG=${CADM_PIG:-cadm_runs/cadm_FBP_s-n-c_f0_long/final}
[ -f $CADM_PIG/lora.pt ] || { echo "missing $CADM_PIG (set CADM_PIG=...)"; exit 1; }
rm -f v3_cadm.done
CUDA_VISIBLE_DEVICES=$G2 nohup sh -c "
python -u train_cadm.py sample --ckpt $CADM_PIG --target_data data/norsvin --target_recon IR --doses 25 10 --test_stride 8 --runs runs_cross_norsvin --name cadm_piglet ;
touch v3_cadm.done
" > v3_cadm.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 nohup sh -c "
python -u pr_train.py --data data/norsvin --physics pr_calib/physics.json --recons IR --ref_recon IR --doses 25 10 --exclude_subjects ER01035833 --st_varmap --proj_resp --sig_blur --iters 6000 --out pr_runs --tag norsvin_scratch ;
python -u pr_adapt.py --ckpt pr_runs/physrecon_norsvin_scratch/final.pt --data data/norsvin --recon IR --doses 25 10 --k 99 --mode z --exclude_subjects ER01035833 --test_stride 8 --runs runs_pr_norsvin --name physrecon_scratch ;
while [ ! -f v3_cadm.done ]; do sleep 60; done ;
for d in 25 10; do python -u evaluate.py --runs \
  runs_pr_norsvin/physrecon_a0_100_IR_to_\${d}_IR_f0_s0 runs_pr_norsvin/physrecon_final_100_IR_to_\${d}_IR_f0_s0 runs_pr_norsvin/physrecon_scratch_100_IR_to_\${d}_IR_f0_s0 \
  runs_old_norsvin/cyclegan_100_IR_to_\${d}_IR_f0_s0 runs_old_norsvin/cut_100_IR_to_\${d}_IR_f0_s0 runs_old_norsvin/vae_100_IR_to_\${d}_IR_f0_s0 runs_old_norsvin/cadm_cal16_100_IR_to_\${d}_IR_f0_s0 \
  runs_cross_norsvin/cut_piglet_100_IR_to_\${d}_IR_f0_s0 runs_cross_norsvin/cadm_piglet_100_IR_to_\${d}_IR_f0_s0 runs_abl_norsvin/abl_zeroshot_s0_100_IR_to_\${d}_IR_f0_s0 \
  --out results_table1 --baselines identity --unit_range -150 250 ; done ;
cat results_table1/*/summary.csv > table1.csv ; echo '[v3] done: table1.csv'
" > v3_main.log 2>&1 &
echo "launched: tail -f v3_main.log v3_cadm.log"
