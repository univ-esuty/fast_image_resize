## Fast Image Resizer!

### Description

- It can resize numerous images very fast by using multiple GPUs.
- It creates a directory including resized images by renaming original directory as following. 

```
input: images/* → output: images_resized/*
```



#### How to Run!

- Simply 

```
python resize_image.py path/to/img_dir [in_w, in_h] [out_w, out_h] --batch=4096
```

- you can use multiple GPUs, 

```
mpiexec -n $num_gpus python resize_image.py path/to/img_dir [in_w, in_h] [out_w, out_h] --batch=4096
```



### benchmarks

- 75K images, 256x256→224x224 on RTX3090 x2：almost 2min30sec

 