cp data/kitti/results/* kitti_eval/results/frcnn/data/
cd kitti_eval
./a.out frcnn
python calculate.py
cd ..
