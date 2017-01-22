#### ry_alpha_regression
python train_end2end.py \
--network vgg \
--pretrained model/faster_rcnn/final --epoch 1 \
--prefix model/faster_rcnn/rcnn  --begin_epoch 1 --end_epoch 500 \
--lr 0.00001 --lr_step 30000 --gpus 3 \
--root_path 'data' --dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'val' \
--frequent 20 2>&1 | tee -a train_reg.log
