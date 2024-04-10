# import open3d as o3d
import os
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from termcolor import colored
import subprocess
import time
import datetime
import shutil


def toc():
    return time.time() * 1000

def myprint(cmd, level):
    color = {'run': 'blue', 'info': 'green', 'warn': 'yellow', 'error': 'red'}[level]
    print(colored(cmd, color))

def log(text):
    myprint(text, 'info')

def log_time(text):
    strf = get_time()
    print(colored(strf, 'yellow'), colored(text, 'green'))

def mywarn(text):
    myprint(text, 'warn')

warning_infos = set()

def oncewarn(text):
    if text in warning_infos:
        return
    warning_infos.add(text)
    myprint(text, 'warn')


def myerror(text):
    myprint(text, 'error')

def run_cmd(cmd, verbo=True, bg=False):
    if verbo: myprint('[run] ' + cmd, 'run')
    if bg:
        args = cmd.split()
        print(args)
        p = subprocess.Popen(args)
        return [p]
    else:
        exit_status = os.system(cmd)
        if exit_status != 0:
            raise RuntimeError
        return []

def mkdir(path):
    if os.path.exists(path):
        return 0
    log('mkdir {}'.format(path))
    os.makedirs(path, exist_ok=True)

def cp(srcname, dstname):
    mkdir(os.join(os.path.dirname(dstname)))
    shutil.copyfile(srcname, dstname)

def check_exists(path):
    flag1 = os.path.isfile(path) and os.path.exists(path)
    flag2 = os.path.isdir(path) and len(os.listdir(path)) >= 10
    return flag1 or flag2


try:
    from lib.config import cfg

    def get_pre(name, time = False):
        if not os.path.exists('debug'): 
            os.makedirs('debug')
        if not os.path.exists(f'debug/{cfg.exp_name}'): 
            os.makedirs(f'debug/{cfg.exp_name}')

        if time:
            output_dir = f"debug/{cfg.exp_name}/{get_time()}_{name}"
        else: output_dir = f"debug/{cfg.exp_name}/{name}" 
        return output_dir
except:
    def get_pre(name, time = False):
        if not os.path.exists('debug'): 
            os.makedirs('debug')

        if time:
            output_dir = f"debug/{get_time()}_{name}"
        else: output_dir = f"debug/{name}" 
        return output_dir

def get_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

def to_numpy(a)->np.ndarray:
    if type(a) == torch.Tensor:
        if a.is_cuda:
            a = a.cpu()
        return a.detach().numpy()
    elif type(a) == np.ndarray:
        return a
    else:
        try:
            return np.array(a)
        except:
            raise TypeError('Unsupported data type')

def save_point_cloud(point_cloud, filename):
    if not cfg.debug: return
    return
    point_cloud = to_numpy(point_cloud)
    
    # 将numpy数组转换为open3d的PointCloud对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # 保存PointCloud到文件
    o3d.io.write_point_cloud(get_pre(filename), pcd)

def output_debug_log(output, name):
    if not cfg.debug: return
    if type(output) != str:
        output = str(output)

    with open(f"{get_pre(name)}.log", 'w') as f:
        f.write(output)
        f.write('\n')

def save_debug(a, name, time = False):
    if not cfg.debug: return
    np.save( f'{get_pre(name, time)}.npy', to_numpy(a))

def save_img(img, name, time = False):
    if not cfg.debug: return
    img = to_numpy(img)
    img = img * 255
    img = img.astype(np.uint8)
    imageio.imwrite(f'{get_pre(name, time)}.png', img)

def save_imgs(msks, name, time = False):
    if not cfg.debug: return
    """Save imgs in a grid"""
    n = len(msks)
    fig, axs = plt.subplots(-(-n//3), 3, figsize=(15, 5*-(-n//3)))
    for i, ax in enumerate(axs.flat):
        ax.axis('off')
        if i<n:
            ax.imshow(msks[i])
    plt.tight_layout()
    plt.savefig(f'{get_pre(name, time)}.png')
    plt.close()
