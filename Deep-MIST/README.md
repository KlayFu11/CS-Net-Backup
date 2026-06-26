# Installation
```shell
conda create -n Deep-MIST python=3.8
conda activate Deep-MIST
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
pip install python-dateutil  # 如果装torch不报warning就不需要
```
```shell
pip install -r requirements.txt
```
```
pip3 install natten==0.14.6+torch1110cu113 -f https://shi-labs.com/natten/wheels
```
```
cd ./deepmist/models/singleframe/ISNet/DCNv2
python3 setup.py build develop
```
# Datasets
```
├── datasets
    ├── DSAT
        ├── train
            ├── data1
                ├── imgs
                    ├── 0.bmp
                    ├── ...
                ├── gt
                    ├── 0.png
                    ├── ...
            ├── ...
        ├── test
```
# Train
```shell
python train.py
```
