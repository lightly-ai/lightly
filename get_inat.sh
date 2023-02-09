wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/train_mini.json.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.tar.gz
wget https://ml-inat-competition-datasets.s3.amazonaws.com/2021/val.json.tar.gz
#
tar -xvf train_mini.tar.gz
tar -xvf train_mini.json.tar.gz
tar -xvf val.tar.gz
tar -xvf val.json.tar.gz

mkdir datasets
mkdir datasets/inat
mv train_mini datasets/inat
mv train_mini.json datasets/inat
mv val datasets/inat
mv val.json datasets/inat