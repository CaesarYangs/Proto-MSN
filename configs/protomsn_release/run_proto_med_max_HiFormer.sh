#!/usr/bin/env bash
SCRIPTPATH="$(
  cd "$(dirname "$0")" >/dev/null 2>&1
  pwd -P
)"
cd $SCRIPTPATH
cd ../../

#^ SAVING SECTION
SCRATCH_BASE_ROOT="res/dev_test/BUSI"
SCRATCH_ROOT="${SCRATCH_BASE_ROOT}/FINAL/0_MAX_HiFormer(dice035_ce065)_V3s_test2"
SAVE_DIR="${SCRATCH_ROOT}/Cityscapes/seg_results/"

#^ DATA SECTION
DATA_ROOT="/DATA/datasets"
ASSET_ROOT=${DATA_ROOT}
DATA_DIR="${DATA_ROOT}/BUSI"

#^ CONFIG SECTION
CONFIGS="./Proto_Med_HiFormer.json"
CONFIGS_TEST="./Proto_Med_TEST.json"

#^ MODEL SECTION
BACKBONE="encoder"
MODEL_NAME="HiFormer_PROTO_MAX"
LOSS_TYPE="PixelPrototypeCELoss_Ultra_V3"
LOSS_CORE="ce"

#^ TRAINING SECTION
GPU_SLOT=1
MAX_EPOCHS=200
BATCH_SIZE=4
BASE_LR=0.01

#^ LOG SECTION
CHECKPOINTS_ROOT="${SCRATCH_ROOT}/MED_CE_test"
CHECKPOINTS_NAME="${MODEL_NAME}_lr1x_"$2
LOG_FILE="${SCRATCH_ROOT}/logs/MED_TEST/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p $(dirname $LOG_FILE)

#-----------------------------------------------start train / val / test----------------------------------------------------
# main training section, start from main.py
if [ "$1" == "train" ]; then
  python -u main.py --configs ${CONFIGS} \
    --drop_last y \
    --phase train \
    --gathered n \
    --loss_balance y \
    --log_to_file n \
    --backbone ${BACKBONE} \
    --model_name ${MODEL_NAME} \
    --gpu ${GPU_SLOT} \
    --data_dir ${DATA_DIR} \
    --loss_type ${LOSS_TYPE} \
    --loss_core ${LOSS_CORE} \
    --max_epoch ${MAX_EPOCHS} \
    --checkpoints_root ${CHECKPOINTS_ROOT} \
    --checkpoints_name ${CHECKPOINTS_NAME} \
    --train_batch_size ${BATCH_SIZE} \
    --base_lr ${BASE_LR} \
    2>&1 | tee ${LOG_FILE}

elif [ "$1" == "resume" ]; then
  python -u main.py --configs ${CONFIGS} \
    --drop_last y \
    --phase train \
    --gathered n \
    --loss_balance y \
    --log_to_file n \
    --backbone ${BACKBONE} \
    --model_name ${MODEL_NAME} \
    --max_iters ${MAX_ITERS} \
    --data_dir ${DATA_DIR} \
    --loss_type ${LOSS_TYPE} \
    --gpu 0 \
    --checkpoints_root ${CHECKPOINTS_ROOT} \
    --checkpoints_name ${CHECKPOINTS_NAME} \
    --resume_continue y \
    --resume ${CHECKPOINTS_ROOT}/checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
    --train_batch_size ${BATCH_SIZE} \
    --distributed \
    2>&1 | tee -a ${LOG_FILE}

elif [ "$1" == "val" ]; then
  python -u main.py --configs ${CONFIGS} --drop_last y --data_dir ${DATA_DIR} \
    --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
    --phase test --gpu 7 --resume ${CHECKPOINTS_ROOT}/checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
    --loss_type ${LOSS_TYPE} --test_dir ${DATA_DIR}/val/image \
    --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms

  # python -m lib.metrics.cityscapes_evaluator --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms/label  \
  #                                      --gt_dir ${DATA_DIR}/val/label

elif [ "$1" == "segfix" ]; then
  if [ "$3" == "test" ]; then
    DIR=${SAVE_DIR}${CHECKPOINTS_NAME}_test_ss/label
    echo "Applying SegFix for $DIR"
    ${PYTHON} scripts/cityscapes/segfix.py \
      --input $DIR \
      --split test \
      --offset ${DATA_ROOT}/cityscapes/test_offset/semantic/offset_hrnext/
  elif [ "$3"x == "val"x ]; then
    DIR=${SAVE_DIR}${CHECKPOINTS_NAME}_val/label
    echo "Applying SegFix for $DIR"
    ${PYTHON} scripts/cityscapes/segfix.py \
      --input $DIR \
      --split val \
      --offset ${DATA_ROOT}/cityscapes/val/offset_pred/semantic/offset_hrnext/
  fi

elif [ "$1" == "test" ]; then
  if [ "$5" == "ss" ]; then
    echo "[single scale] test"
    python -u main.py --configs ${CONFIGS} --drop_last y --data_dir ${DATA_DIR} \
      --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
      --phase test --gpu 0 --resume ${CHECKPOINTS_ROOT}/checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
      --test_dir ${DATA_DIR}/test --log_to_file n \
      --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ss
  else
    echo "[multiple scale + flip] test"
    python -u main.py --configs ${CONFIGS_TEST} --drop_last y --data_dir ${DATA_DIR} \
      --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
      --phase test --gpu 0 --resume ${CHECKPOINTS_ROOT}/checkpoints/cityscapes/${CHECKPOINTS_NAME}_latest.pth \
      --test_dir ${DATA_DIR}/test --log_to_file n \
      --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ms
  fi

else
  echo "$1"x" is invalid..."
fi
