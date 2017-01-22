
## regress ry and alpha
python test_rcnn.py --network vgg --prefix /data01/llb/model/faster_rcnn/kitti_ry_cls_input_up_2/ry_alpha_car_only_reg --epoch 16 --dataset_path 'data/kitti' --dataset 'Kitti' --image_set 'val_ry_alpha_car_only' --has_rpn  --gpu 7
