
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

def show_gt(list_path, image_path, num_classes = 4) :
    HAVE_ORIENTATION = True
    obj_box_dim = 5 if HAVE_ORIENTATION else 4
    color = (random.random(), random.random(), random.random())

    assert os.path.exists(list_path), 'Path does not exist: {}'.format(list_path)
    with open(list_path, 'r') as f:
            for line in f:
                box_list = []
                label = line.strip().split(':')
                image_name = label[0]

                bbox = label[1:]
                for i in range(num_classes - 1):
                    if len(bbox[i]) == 0:
                        box_list.append([])
                        continue
                    else:
                        class_i_box = map(float, bbox[i].strip().split(' '))
                        box_list.append(class_i_box)              

                boxes = np.concatenate([np.array(box_list[i], dtype=np.float32) for i in range(num_classes - 1)], axis=0)
                boxes = boxes.reshape(-1, obj_box_dim)
                print boxes

                im = cv2.imread( image_path + image_name)

                for box in boxes:
                    if HAVE_ORIENTATION:
                        bbox = box[1:]
                        orientation = box[0]
                    else:
                        box = box_
                        orientation = 0

                    rect = plt.Rectangle((bbox[0], bbox[1]),
                                     bbox[2] - bbox[0],
                                     bbox[3] - bbox[1], fill=False,
                                     edgecolor=color, linewidth=3.5)  
                    plt.gca().add_patch(rect)                        
                    plt.gca().text(bbox[0], bbox[3] - 2,
                               '{:s} {:.3f}'.format('Ori', orientation),
                               bbox=dict(facecolor=color, alpha=0.5), fontsize=12, color='white')


                plt.imshow(im)
                plt.show()

if __name__ == '__main__':

    list_path = "data/kitti/imglists/val_O.lst"    
    image_path = "data/kitti/images/"

    show_gt(list_path, image_path)