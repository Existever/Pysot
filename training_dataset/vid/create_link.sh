

VID_base_path='/media/solanliu/zs402_2/dataset'

#创建pysot工程下文件夹，用于建立软链接
mkdir -p ILSVRC2015/Annotations/VID/train/
mkdir -p ILSVRC2015/Data/VID/train/

#标签文件的软连接
ln -sfbv $VID_base_path/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0000 ILSVRC2015/Annotations/VID/train/a
ln -sfbv $VID_base_path/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0001 ILSVRC2015/Annotations/VID/train/b
ln -sfbv $VID_base_path/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0002 ILSVRC2015/Annotations/VID/train/c
ln -sfbv $VID_base_path/ILSVRC2015/Annotations/VID/train/ILSVRC2015_VID_train_0003 ILSVRC2015/Annotations/VID/train/d
ln -sfbv $VID_base_path/ILSVRC2015/Annotations/VID/val ILSVRC2015/Annotations/VID/train/e


#数据文件的软链接
ln -sfbv $VID_base_path/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0000 ILSVRC2015/Data/VID/train/a
ln -sfbv $VID_base_path/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0001 ILSVRC2015/Data/VID/train/b
ln -sfbv $VID_base_path/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0002 ILSVRC2015/Data/VID/train/c
ln -sfbv $VID_base_path/ILSVRC2015/Data/VID/train/ILSVRC2015_VID_train_0003 ILSVRC2015/Data/VID/train/d
ln -sfbv $VID_base_path/ILSVRC2015/Data/VID/val ILSVRC2015/Data/VID/train/e
