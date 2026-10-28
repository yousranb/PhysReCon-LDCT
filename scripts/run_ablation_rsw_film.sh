#!/bin/sh
# Clean ablation of the ray-wise statistical weighting (RSW, lambda_omega) and of the domain conditioning
# (embedding z and FiLM of F_theta) on Norsvin IR. Same protocol for every cell: stage-1 checkpoint
# physrecon_stvar, early stopping on a held-out support pig, 600 steps, 3 seeds (seed = support hold-out
# pig + noise realisations), every 8th test slice (282 per dose, the slices of the joint evaluation).
# Usage on a node with 2 GPUs:  sh run_ablation_rsw_film.sh   (then: python pr_abl_collect.py)
CK=pr_runs/physrecon_stvar/final.pt
A="--ckpt $CK --data data/norsvin --recon IR --doses 25 10 --k 99 --steps 600 --exclude_subjects ER01035833 --test_stride 8 --runs runs_abl_norsvin"
rm -f abl_gpu1.done
CUDA_VISIBLE_DEVICES=0 nohup sh -c "
for s in 0 1 2; do
  python -u pr_adapt.py $A --seed \$s --mode zfilm                 --name abl_full_s\$s ;
  python -u pr_adapt.py $A --seed \$s --mode zfilm --no_proj_resp  --name abl_noRSW_s\$s ;
done ;
while [ ! -f abl_gpu1.done ]; do sleep 60; done ;
for d in 25 10; do python -u evaluate.py --runs \$(ls -d runs_abl_norsvin/*_to_\${d}_IR_f0_s0) --out results_abl_norsvin --baselines identity --unit_range -150 250 ; done ;
python -u pr_abl_collect.py --results results_abl_norsvin
" > abl_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup sh -c "
python -u pr_adapt.py $A --mode zero --name abl_zeroshot_s0 ;
for s in 0 1 2; do
  python -u pr_adapt.py $A --seed \$s --mode heads                           --name abl_noFiLM_s\$s ;
  python -u pr_adapt.py $A --seed \$s --mode zfilm --z_init zero --freeze_z  --name abl_noZ_s\$s ;
  python -u pr_adapt.py $A --seed \$s --mode heads --z_init zero --freeze_z  --name abl_noCond_s\$s ;
done ;
touch abl_gpu1.done
" > abl_gpu1.log 2>&1 &
echo "launched: tail -f abl_gpu0.log abl_gpu1.log"
