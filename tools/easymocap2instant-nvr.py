import os
import numpy as np
import json
from easymocap.mytools.camera_utils import read_cameras
from easymocap.bodymodel.smpl import SMPLModel

def main():
    data_root='../data/h36m/S9'
    print(f"Processing {data_root}...")

    origin_params_dir=os.path.join(data_root, 'output-smpl-3d/smpl')
    params_dir=os.path.join(data_root, 'smpl_params')
    vertices_dir=os.path.join(data_root, 'smpl_vertices')
    if not os.path.exists(params_dir): os.mkdir(params_dir)
    if not os.path.exists(vertices_dir): os.mkdir(vertices_dir)

    cfg={
        'model_path': '../data/smpl-meta/SMPL_NEUTRAL.pkl',
        'regressor_path': '../data/smpl-meta/J_regressor_body25.npy',
    }

    smpl = SMPLModel(**cfg)

    # 求 vertices, 并转换 vertices 与 params
    print("Converting vertices and params...")
    for name in os.listdir(origin_params_dir):
        origin_params_path=os.path.join(origin_params_dir, name)
        frame=int(name.split('.')[0])
        params_path=os.path.join(params_dir, f"{frame}.npy")
        vertices_path=os.path.join(vertices_dir, f"{frame}.npy")
        # print(params_path)
        # print(vertices_path)

        with open(origin_params_path, 'r') as f:
            params = json.load(f)['annots'][0]
        params.pop('id')
        for key, item in params.items():
            params[key]=np.array(params[key])

        vertices = smpl.vertices(params, return_tensor=False)[0]
        np.save(vertices_path, vertices)

        params['poses']=smpl.export_full_poses(params['poses'])
        np.save(params_path, params, allow_pickle=True)

    # annots
    annots_path=os.path.join(data_root, 'annots.npy')
    print("Converting annots...")

    cameras=read_cameras(data_root)
    num_cameras=len(cameras.keys())
    cameras = dict(sorted(cameras.items(), key=lambda x: int(x[0])))
    cams = {
        'D': [camera['dist'].reshape(1,-1) for camera in cameras.values()],
        'K': [camera['K'] for camera in cameras.values()],
        'R': [camera['R'] for camera in cameras.values()],
        'T': [camera['T']*1000. for camera in cameras.values()]
    }

    images_dir=os.path.join(data_root, 'images')
    image_list=[]
    for sub_dir in sorted(os.listdir(images_dir)):
        sub_dir=os.path.join(images_dir, sub_dir)
        for image_path in sorted(os.listdir(sub_dir)):
            image_path=os.path.join(sub_dir, image_path)
            image_path=image_path.split(data_root)[-1][1:]
            image_list.append(image_path)
    num_images=len(image_list)//num_cameras
    ims=[{'ims': image_list[i::num_images]} for i in range(num_images)]

    annots={
        'cams':cams,
        'ims':ims
    }

    np.save(annots_path, annots, allow_pickle=True)

if __name__ == '__main__':
    main()