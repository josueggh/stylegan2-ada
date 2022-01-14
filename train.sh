data="/data/TFRecords/Dubai"
seed=1002

aug="noaug"
# aug="ada --target .7"

gamma="5"
lrate=.002

# resume="results/00048-NatureCoral-mirror-stylegan2-gamma5-noaug-resumecustom/network-snapshot-000614.pkl"

cfg="stylegan2"

DNNLIB_CACHE_DIR=$PWD/.cache python train.py \
  --outdir=./results \
  --gpus=4 \
  --mirror=1 \
  --metrics=none \
  --seed=$seed \
  --aug=$aug \
  --cfg=$cfg \
  --gamma=$gamma \
  --lrate=$lrate \
  --data=$data
#   --resume=$resume
