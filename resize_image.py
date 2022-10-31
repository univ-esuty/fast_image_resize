import argparse
import pathlib
import sys, os
import glob
import socket

from mpi4py import MPI
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch, torchvision
import torch.distributed as dist


GPUS_PER_NODE = 8
SETUP_RETRY_COUNT = 3

def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()

def setup_dist(gpu_id_start=0):
    """
    Setup a distributed process group.
    """
    if dist.is_initialized():
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE + gpu_id_start}"

    comm = MPI.COMM_WORLD
    backend = "gloo" if not torch.cuda.is_available() else "nccl"

    if backend == "gloo":
        hostname = "localhost"
    else:
        hostname = socket.gethostbyname(socket.getfqdn())
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)
    os.environ["WORLD_SIZE"] = str(comm.size)

    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend=backend, init_method="env://")


def getpathes(data_dir, eliminate_head=True):
    results = []
    dirs = []
    
    pathes = glob.glob(f"{data_dir}/**/*", recursive=True)
    for path in pathes:
        if os.path.isdir(path):
            dirs.append(path)
        
        elif path.split(".")[-1].lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(path)
    
    if eliminate_head:
        for i in range(len(results)):
            results[i] = results[i].replace(f"{data_dir}/", "")
        for i in range(len(dirs)):
            dirs[i] = dirs[i].replace(f"{data_dir}/", "")

        
    return results, dirs


## resize on GPU
def resize(imgs, out_size=(224, 224), device='cuda'):
    arr = torch.from_numpy(imgs.astype(np.float32)).to(device).permute(0,3,1,2)
    arr = torchvision.transforms.functional.resize(img=arr, size=out_size)
    imgs = arr.permute(0,2,3,1).cpu().numpy().astype(np.uint8)
    return imgs


## main function   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('in_dir', help='original images directory', type=str)
    parser.add_argument('in_size', help='resized image size', type=list)
    parser.add_argument('out_size', help='resized image size', type=list)
    parser.add_argument('--batch', type=int, default=2048)
    args = parser.parse_args()
    
    all_pathes, all_dirs = getpathes(args.in_dir)
    rank = MPI.COMM_WORLD.Get_rank() 
    length = len(all_pathes) // MPI.COMM_WORLD.Get_size() + 1 
    pathes = all_pathes[rank*length:rank*length+length]
    
    if rank == 0:
        os.makedirs(f'{args.in_dir}_resized', exist_ok=True)
        for dir in all_dirs:
            os.makedirs(f'{args.in_dir}_resized/{dir}', exist_ok=True,)
        print(f"Directory {args.in_dir}_resized was created successfully.")
        
        pbar = tqdm(total=len(pathes)//args.batch+1)
        
    for i in range(0, len(pathes), args.batch):
        imgs = np.zeros((args.batch, args.in_size[0], args.in_size[1], 3), dtype=np.uint8)
        
        for j, path in enumerate(pathes[i: i+args.batch]):
            pil_image = Image.open(f"{args.in_dir}/{path}")
            imgs[j] = np.array(pil_image.convert("RGB"))
            
        imgs = resize(imgs, out_size=args.out_size, device=f'cuda:{rank}')
        
        for j, path in enumerate(pathes[i: i+args.batch]):
            new_file = pathlib.Path(f"{args.in_dir}_resized/{path}")
            new_file.touch()
            pil_image = Image.fromarray(imgs[j])
            pil_image.save(f"{args.in_dir}_resized/{path}", quality=100)
            
        if rank == 0:
            pbar.update(1)
        
        MPI.COMM_WORLD.barrier()
        
            
    