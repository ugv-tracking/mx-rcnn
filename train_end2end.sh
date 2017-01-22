

#python train_end2end.py --network vgg --pretrained model/vgg-e2e --epoch 10 --prefix model/Orientation/Orientation \
#		--begin_epoch 0 --end_epoch 50 --lr 0.001 --lr_step 50000 --gpus 7 \
#		--root_path '/home/xinya/rcnn' --dataset_path 'data/VOCdevkit' --dataset 'PascalVOC' --image_set '2007_trainval' \ 
#		2>&1 | tee -a train_voc.log

#python train_end2end.py --network vgg --pretrained model/vgg-e2e --epoch 10 --prefix model/Orientation/Orientation --begin_epoch 0 --end_epoch 50 --lr 0.001 --lr_step 50000 --gpus 6 --root_path 'data' \
#--dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'train' --frequent 1  #2>&1 | tee -a train_kitti.log


#python train_end2end.py --network vgg --pretrained model/vgg-e2e --epoch 10 --prefix model/Orientation/Orientation --begin_epoch 0 --end_epoch 50 --lr 0.001 --lr_step 50000 --gpus 3 --root_path 'data' \
#--dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'train_O' --frequent 20 2>&1 | tee -a train_kitti.log


#python train_end2end.py --network vgg  --resume --pretrained model/Orientation/Orientation --epoch 28 --prefix model/Orientation/Orientation --begin_epoch 28 --end_epoch 50 --lr 0.000001 --lr_step 50000 --gpus 3 --root_path 'data' \
#--dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'train_O' --frequent 20 2>&1 | tee -a train_kitti.log


#python train_end2end.py --network vgg  --pretrained model/vgg-e2e --epoch 10 --prefix model/Orientation/ry  --begin_epoch 0 --end_epoch 500 --lr 0.001 --lr_step 30000 --gpus 2 --root_path 'data' \
#--dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'train_ry' --frequent 20 2>&1 | tee -a train_kitti_ry.log

#python train_end2end.py --network vgg  --pretrained model/Orientation/ry --epoch 25 --prefix model/Orientation/ry  --begin_epoch 25 --end_epoch 500 --lr 0.00001 --lr_step 30000 --gpus 2 --root_path 'data' \
#--dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'train_ry' --frequent 20 2>&1 | tee -a train_kitti_ry.log


#python train_end2end.py --network vgg  --pretrained model/Orientation/ry --epoch 25 --prefix model/Orientation/ry_context_roi  --begin_epoch 0 --end_epoch 500 --lr 0.00001 --lr_step 30000 --gpus 2 --root_path 'data' \
#--dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'train_ry' --frequent 20 2>&1 | tee -a train_kitti_ry.log

#### up1
#python train_end2end.py --network vgg --pretrained model/Orientation/ry_car_only_cls  --epoch 25 --prefix model/Orientation/ry_car_only_cls  --begin_epoch 25 --end_epoch 500 --lr 0.001 --lr_step 30000 --gpus 6 --root_path 'data' \
#--dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'train_ry_car_only' --frequent 20 2>&1 | tee -a train_kitti_ry_car_only_cls.log

#### up2
#python train_end2end.py --network vgg --pretrained /data01/llb/model/faster_rcnn/kitti_ry_cls_input_up_2/ry_car_only_cls --epoch 26 --prefix /data01/llb/model/faster_rcnn/kitti_ry_cls_input_up_2/ry_car_only_cls  --begin_epoch 26 --end_epoch 500 --lr 0.00001 --lr_step 30000 --gpus 6 --root_path 'data' \
#--dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'train_ry_car_only' --frequent 20 2>&1 | tee -a train_kitti_ry_car_only_cls_input_up_2.log

#### up3
#python train_end2end.py --network vgg --pretrained /data01/llb/model/faster_rcnn/kitti_ry_cls_input_up_3/ry_car_only_cls --epoch 9 --prefix /data01/llb/model/faster_rcnn/kitti_ry_cls_input_up_3/ry_car_only_cls  --begin_epoch 9 --end_epoch 500 --lr 0.00001 --lr_step 30000 --gpus 6 --root_path 'data' \
#--dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'trainval_ry_car_only' --frequent 20 2>&1 | tee -a train_kitti_ry_car_only_cls_input_up_3.log

#### ry_alpha_cls
#python train_end2end.py --network vgg --pretrained /data01/llb/model/faster_rcnn/kitti_ry_cls_input_up_3/ry_alpha_car_only_cls --epoch 13 --prefix /data01/llb/model/faster_rcnn/kitti_ry_cls_input_up_3/ry_alpha_car_only_cls  --begin_epoch 13 --end_epoch 500 --lr 0.0001 --lr_step 30000 --gpus 3 --root_path 'data' \
#--dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'train_ry_alpha_car_only' --frequent 20 2>&1 | tee -a train_ry_alpha_cls_car_only.log

#### ry_alpha_regression
python train_end2end.py --network vgg --pretrained /data01/llb/model/faster_rcnn/kitti_ry_cls_input_up_2/ry_alpha_car_only_reg --epoch 15 --prefix /data01/llb/model/faster_rcnn/kitti_ry_cls_input_up_2/ry_alpha_car_only_reg  --begin_epoch 15 --end_epoch 500 --lr 0.00001 --lr_step 30000 --gpus 3 --root_path 'data' \
--dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'train_ry_alpha_car_only' --frequent 20 2>&1 | tee -a train_ry_alpha_car_only_reg.log
