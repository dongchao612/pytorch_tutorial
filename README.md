# pytorch_tutorial
pytorch教程
## 01 pytorch 的安装
查看NVIDIA的版本
```text
nvidia-smi 
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 531.97                 Driver Version: 531.97       CUDA Version: 12.1     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                      TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf            Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 4060 L...  WDDM | 00000000:01:00.0 Off |                  N/A |
| N/A   55C    P0               50W /  N/A|   3187MiB /  8188MiB |     53%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A     18040      C   ...s\anaconda3\envs\pytorch\python.exe    N/A      |
+---------------------------------------------------------------------------------------+
```
查看安装之后是否可用

```python
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

## 加载数据

```python
class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        pass

    def __getitem__(self, idx):
        return img, label

    def __len__(self):
        return len(self.img_path)

```