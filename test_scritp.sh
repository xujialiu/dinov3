# build MultiScaleDeformableAttention
cd dinov3/eval/segmentation/models/utils/ops
python setup.py build_ext --inplace
cp dinov3/eval/segmentation/models/utils/ops/MultiScaleDeformableAttention.cpython-311-x86_64-linux-gnu.so  /home/xujialiu/miniconda3/envs/dinov3/lib/python3.11/site-packages
# verify it can work
python -c "
import torch  # Load torch first to set up library paths
import sys
import MultiScaleDeformableAttention as MSDA
print('✓ Import successful!')
print(f'Available functions: {[x for x in dir(MSDA) if not x.startswith(\"_\")]}')"


python dinov3/eval/segmentation/run.py \
      config=dinov3/eval/segmentation/configs/config-ade20k-m2f-training.yaml \
      output_dir=./output/m2f_smoke_test \
      model.dino_hub=dinov3_vitl16 \
      datasets.root=./ADEChallengeData2016 \
      scheduler.total_iter=10 \
      eval.eval_interval=10 \
      eval.max_val_samples=10 \
      bs=1 \
      n_gpus=1

