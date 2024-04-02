import numpy as np
import os
import json
import cv2
import imageio
import matplotlib.pyplot as plt
import tqdm

def cal_Square(bbox):
    if bbox is None or bbox[4]<0.6:return 0
    return (bbox[0]-bbox[2])*(bbox[1]-bbox[3])


if __name__ == '__main__':
    data_path = '../data/internet-rotate'

    annots_path = os.path.join(data_path, 'annots')
    schp_path = os.path.join(data_path, 'schp')

    for sub_dir in os.listdir(annots_path):
        sub_path = os.path.join(annots_path, sub_dir)
        for annot_dir in tqdm.tqdm(sorted(os.listdir(sub_path))[0:]):
            annot_path = os.path.join(sub_path, annot_dir)
            # print(annot_path)
            with open(annot_path, 'r') as f:
                annot = json.load(f)
            # print(annot['annots'])

            max_bbox = None
            for person in annot['annots']:
                # print(person)
                # print(cal_Square(person['bbox']))
                if(cal_Square(max_bbox)<cal_Square(person['bbox'])):
                    max_bbox = person['bbox']
            max_bbox = np.array(max_bbox).astype(int)
            assert max_bbox is not None
            # print(max_bbox)

            schp_path = annot_path.replace('annots', 'schp').replace('json', 'png')
            # print(schp_path)
            img = imageio.v2.imread(schp_path)
            img[:max_bbox[1], :] = [0, 0, 0, 255]  # 上
            img[max_bbox[3]:, :] = [0, 0, 0, 255]  # 下
            img[:, :max_bbox[0]] = [0, 0, 0, 255]  # 左
            img[:, max_bbox[2]:] = [0, 0, 0, 255]  # 右

            imageio.imwrite(schp_path, img)
            # plt.imshow(img)
            # plt.scatter([max_bbox[0], max_bbox[2]], [max_bbox[1], max_bbox[3]])
            # plt.show()
            # break
        # break