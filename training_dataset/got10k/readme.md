##下载got10k数据集
[下载网址：](http://got-10k.aitestunion.com/downloads)：http://got-10k.aitestunion.com/downloads

##解析got10k数据集
```shell script
#在parse_got10k.py文件中修改got10k路径
root_dir = '/home/rainzsy/datasets/got10k/'
subset = 'train'

#got10k目录下的文件结构

├── train
│   ├── GOT-10k_Train_000001
│   ├── GOT-10k_Train_000002
│   ├── GOT-10k_Train_000003
... 
│   ├── GOT-10k_Train_xxxxxx
│   ├── GOT-10k_Train_xxxxxx
│   └── list.txt
├── val
│   ├── GOT-10k_Train_xxxxxx
│   ├── GOT-10k_Train_xxxxxx
│   ├── GOT-10k_Train_xxxxxx
... 
│   ├── GOT-10k_Train_xxxxxx
│   ├── GOT-10k_Train_xxxxxx
│   └── list.txt

```


##生成json文件
```shell script
#gen_got10k_json.py文件中修改subset
运行gen_got10k_json.py,生成train.json,val.json

```


##生成符合pysot框架下siamese系列输入crop图像
```shell script
#修改par_crop_got10k.py 中的got10k的路径
运行par_crop_got10k.py
```



