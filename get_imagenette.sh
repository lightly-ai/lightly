mkdir datasets
cd datasets
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz
tar -xvzf imagenette2-160.tgz
rm imagenette2-160.tgz
cd ..

cd datasets
wget https://www.kaggle.com/datasets/swaroopkml/cifar10-pngs-in-folders/download?datasetVersionNumber=1