

#python train_end2end.py --network vgg --pretrained model/vgg-e2e --epoch 10 --prefix model/Orientation/Orientation \
#		--begin_epoch 0 --end_epoch 50 --lr 0.001 --lr_step 50000 --gpus 7 \
#		--root_path '/home/xinya/rcnn' --dataset_path 'data/VOCdevkit' --dataset 'PascalVOC' --image_set '2007_trainval' \ 
#		2>&1 | tee -a train_voc.log

#python test_rcnn.py --network vgg --prefix model/vgg-e2e --epoch 10 --dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'val' --has_rpn --vis --gpu 3
#python test_rcnn.py --network vgg --prefix model/Orientation/Orientation --epoch 1 --dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'val' --has_rpn --vis --gpu 7

#python test_rcnn.py --network vgg --prefix model/Orientation/Orientation --epoch 10 --dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'val_O' --has_rpn  --gpu 7


#python test_rcnn.py --network vgg --prefix model/Orientation/ry_car_only_cls --epoch 18 --dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'val_ry' --has_rpn  --gpu 0



## regress ry and alpha
python test_rcnn.py --network vgg --prefix /data01/llb/model/faster_rcnn/kitti_ry_cls_input_up_2/ry_alpha_car_only_reg --epoch 16 --dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'val_ry_alpha_car_only' --has_rpn  --gpu 7
