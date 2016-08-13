#!/bin/bash
# Usage:
# ./experiments/scripts/faster_rcnn_end2end.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

TRAINING=false
TESTING=false
EVALUATION=true
# For testing
TRAIN_DATE="2016-08-08_09-47-40"
TRAIN_ITERS=30000
# For evaluation
#EVAL_DATE="2016-08-10_18-40-54"
EVAL_DATE="2016-08-11_18-45-35"
EVAL_ITERS=30000

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  kitti)
    TRAIN_IMDB="kitti_train"
    TEST_IMDB="kitti_val"
    PT_DIR="kitti"
    ITERS=30000
    ;;
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=70000
    ;;
  coco)
    # This is a very long and slow training schedule
    # You can probably use fewer iterations and reduce the
    # time to the LR drop (set in the solver to 350,000 iterations).
    TRAIN_IMDB="coco_2014_train"
    TEST_IMDB="coco_2014_minival"
    PT_DIR="coco"
    ITERS=490000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

if [ "$TRAINING" = true -o "$TESTING" = true ]
then
	LOG="experiments/logs/faster_rcnn_end2end_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
	exec &> >(tee -a "$LOG")
	echo Logging output to "$LOG"
fi

if [ "$TRAINING" = true ]
then
	time ./tools/train_net.py --gpu ${GPU_ID} \
	  --solver models/${PT_DIR}/${NET}/faster_rcnn_end2end/solver_30000.prototxt \
	  --weights data/imagenet_models/${NET}.v2.caffemodel \
	  --imdb ${TRAIN_IMDB} \
	  --iters ${ITERS} \
	  --cfg experiments/cfgs/faster_rcnn_end2end_600_224.yml \
	  ${EXTRA_ARGS}

	set +x
	NET_FINAL=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`
	set -x
else
	NET_FINAL="/home/liujie/ICRA/py-faster-rcnn/output/faster_rcnn_end2end/kitti_train_${TRAIN_DATE}/${NET}_faster_rcnn_iter_${TRAIN_ITERS}.caffemodel"
	if [ ! -f "$NET_FINAL" ]
	then
		NET_FINAL="/home/liujie/ICRA/py-faster-rcnn/output/faster_rcnn_end2end/kitti_train_${TRAIN_DATE}/`echo "$NET" | awk '{print tolower($0)}'`_faster_rcnn_iter_${TRAIN_ITERS}.caffemodel"
	fi
fi

if [ "$TESTING" = true ]
then
	time ./tools/test_net.py --gpu ${GPU_ID} \
	  --def models/${PT_DIR}/${NET}/faster_rcnn_end2end/test.prototxt \
	  --net ${NET_FINAL} \
	  --imdb ${TEST_IMDB} \
	  --cfg experiments/cfgs/faster_rcnn_end2end_600_224.yml \
	  ${EXTRA_ARGS}
fi

if [ "$EVALUATION" = true ]
then
	RESULT_DIR="/home/liujie/ICRA/py-faster-rcnn/output/faster_rcnn_end2end/kitti_val_${EVAL_DATE}/${NET}_faster_rcnn_iter_${EVAL_ITERS}"
	if [ ! -d "$RESULT_DIR" ]
	then
		RESULT_DIR="/home/liujie/ICRA/py-faster-rcnn/output/faster_rcnn_end2end/kitti_val_${EVAL_DATE}/`echo "$NET" | awk '{print tolower($0)}'`_faster_rcnn_iter_${EVAL_ITERS}"
	fi
	time ./tools/KITTI/obj_eval ${RESULT_DIR}  
	time ./tools/KITTI/AP.py ${RESULT_DIR}/plot
fi
