Benchmarks 
===================================
We show benchmarks of the different models for self-supervised learning
and their performance on public datasets.


We have benchmarks we regularly update for these datasets:

- CIFAR10 `CIFAR10`_
- ImageNette `ImageNette`_
- Imagenet100 `Imagenet100`_


ImageNette
-----------------------------------

We use the ImageNette dataset provided here: https://github.com/fastai/imagenette

For our benchmarks we use the 160px version and resize the input images to 128 pixels. 
Training a single model for 800 epochs on a V100 GPU takes around 5 hours.

The current benchmark contains the following models:

- :ref:`BarlowTwins <barlowtwins>`
- :ref:`BYOL <byol>`
- :ref:`DCL <dcl>`
- :ref:`DCLW <dcl>`
- :ref:`DINO <dino>`
- :ref:`MSN <msn>`
- :ref:`MoCo <moco>`
- :ref:`NNCLR <nnclr>`
- :ref:`PMSN <pmsn>`
- :ref:`SimCLR <simclr>`
- :ref:`SimMiM <simmim>`
- :ref:`SimSiam <simsiam>`
- :ref:`SwAV <swav>`
- :ref:`SwAV Queue <swav_queue>`
- :ref:`TiCo <tico>`
- :ref:`VICReg <vicreg>`
- :ref:`VICRegL <vicregl>`


.. csv-table:: ImageNette benchmark results using kNN evaluation on the test set using 128x128 input resolution.
   :header: "Model", "Epochs", "Batch Size", "Accuracy", "Runtime", "GPU Memory"
   :widths: 20, 20, 20, 20, 20, 20

  "BarlowTwins", "800", "256", "0.850", "279.5 Min", "5.7 GByte"
  "BYOL", "800", "256", "0.887", "202.7 Min", "4.3 GByte"
  "DCL", "800", "256", "0.864", "183.7 Min", "3.7 GByte"
  "DCLW", "800", "256", "0.861", "188.5 Min", "3.7 GByte"
  "DINO (Res18)", "800", "256", "0.887", "291.6 Min", "8.5 GByte"
  "FastSiam", "800", "256", "0.865", "280.9 Min", "7.3 GByte"
  "MAE (ViT-S)", "800", "256", "0.620", "208.2 Min", "4.6 GByte"
  "MSN (ViT-S)", "800", "256", "0.833", "394.0 Min", "16.3 GByte"
  "Moco", "800", "256", "0.874", "220.7 Min", "4.2 GByte"
  "NNCLR", "800", "256", "0.885", "207.1 Min", "3.8 GByte"
  "PMSN (ViT-S)", 200, 512, 0.830, "401.1 Min", "16.3 GByte"
  "SimCLR", "800", "256", "0.889", "206.4 Min", "3.7 GByte"
  "SimMIM (ViT-B32)", "800", "256", "0.351", "302.8 Min", "10.5 GByte"
  "SimSiam", "800", "256", "0.871", "178.2 Min", "3.9 GByte"
  "SwaV", "800", "256", "0.899", "309.0 Min", "6.4 GByte"
  "SwaVQueue", "800", "256", "0.898", "300.3 Min", "6.4 GByte"
  "SMoG", "800", "256", "0.782", "250.2 Min", "2.5 GByte"
  "TiCo", "800", "256", "0.857", "184.7 Min", "2.5 GByte"
  "VICReg", "800", "256", "0.843", "192.9 Min", "5.7 GByte"
  "VICRegL", "800", "256", "0.781", "207.4 Min", "5.7 GByte"

You can reproduce the benchmarks using the following script:
:download:`imagenette_benchmark.py <benchmarks/imagenette_benchmark.py>` 


CIFAR10
-----------------------------------

Cifar10 consists of 50k training images and 10k testing images. We train the
self-supervised models from scratch on the training data. At the end of every
epoch we embed all training images and use the features for a kNN classifier 
with k=200 on the test set. The reported kNN test accuracy is the max accuracy
over all epochs the model reached.
All experiments use the same ResNet-18 backbone and we disable the gaussian blur
augmentation due to the small image sizes.

.. note:: The ResNet-18 backbone in this benchmark is slightly different from 
          the torchvision variant as it starts with a 3x3 convolution and has no
          stride and no `MaxPool2d`. This is a typical variation used for cifar10
          benchmarks of SSL methods.

.. role:: raw-html(raw)
   :format: html

.. csv-table:: Cifar10 benchmark results showing kNN test accuracy, runtime and peak GPU memory consumption for different training setups.
  :header: "Model", "Epochs", "Batch Size", "Accuracy", "Runtime", "GPU Memory"
  :widths: 20, 20, 20, 20, 20, 20

  "BarlowTwins", 200, 128, 0.835, "193.4 Min", "2.2 GByte"
  "BYOL", 200, 128, 0.872, "217.0 Min", "2.3 GByte"
  "DCL (*)", 200, 128, 0.842, "126.9 Min", "1.7 GByte"
  "DCLW (*)", 200, 128, 0.833, "127.5 Min", "1.8 GByte"
  "DINO", 200, 128, 0.868, "220.7 Min", "2.3 GByte"
  "FastSiam (*)", 200, 128, 0.906, "164.0 Min", "2.7 GByte"
  "Moco", 200, 128, 0.838, "229.5 Min", "2.3 GByte"
  "NNCLR", 200, 128, 0.838, "198.7 Min", "2.2 GByte"
  "SimCLR", 200, 128, 0.822, "182.7 Min", "2.2 GByte"
  "SimSiam (*)", 200, 128, 0.837, "92.5 Min", "1.5 GByte"
  "SwaV", 200, 128, 0.806, "182.4 Min", "2.2 GByte"
  "BarlowTwins", 200, 512, 0.827, "160.7 Min", "7.5 GByte"
  "BYOL", 200, 512, 0.872, "188.5 Min", "7.7 GByte"
  "DCL (*)", 200, 512, 0.834, "113.6 Min", 6.1 GByte"
  "DCLW (*)", 200, 512, 0.830, "113.8 Min", 6.2 GByte"
  "DINO", 200, 512, 0.862, "191.1 Min", "7.5 GByte"
  "FastSiam (*)", 200, 512, 0.788, 146.9 Min", "9.5 GByte"
  "Moco (**)", 200, 512, 0.850, "196.8 Min", "7.8 GByte"
  "NNCLR (**)", 200, 512, 0.836, "164.7 Min", "7.6 GByte"
  "SimCLR", 200, 512, 0.828, "158.2 Min", "7.5 GByte"
  "SimSiam (*)", 200, 512, 0.817, "83.6 Min", "4.9 GByte"
  "SwaV", 200, 512, 0.833, "158.4 Min", "7.5 GByte"
  "BarlowTwins", 800, 512, 0.857, "641.5 Min", "7.5 GByte"
  "BYOL", 800, 512, 0.911, "754.2 Min", "7.8 GByte"
  "DCL (*)", 800, 512, 0.873, "459.6 Min", "6.1 GByte"
  "DCLW (*)", 800, 512, 0.873, "455.8 Min", "6.1 GByte"
  "DINO", 800, 512, 0.884, "765.5 Min", "7.6 GByte"
  "FastSiam (*)", 800, 512, 0.902, "582.0 Min", "9.5 GByte"
  "Moco (**)", 800, 512, 0.900, "787.7 Min", "7.8 GByte"
  "NNCLR (**)", 800, 512, 0.896, "659.2 Min", "7.6 GByte"
  "SimCLR", 800, 512, 0.875, "632.5 Min", "7.5 GByte"
  "SimSiam (*)", 800, 512, 0.902, "329.8 Min", "4.9 GByte"
  "SwaV", 800, 512, 0.881, "634.9 Min", "7.5 GByte"

(*): Smaller runtime and memory requirements due to different hardware settings
and pytorch version. Runtime and memory requirements are comparable to SimCLR
with the default settings.
(**): Increased size of memory bank from 4096 to 8192 to avoid too quickly 
changing memory bank due to larger batch size.

We make the following observations running the benchmark:

- Self-Supervised models benefit from larger batch sizes and longer training.
- All models need around 3-4h to complete the 200 epoch benchmark and 11-13h
  for the 800 epoch benchmark.
- Memory consumption is roughly the same for all models.
- Some models, like MoCo or SwaV, learn quickly in the beginning and then 
  plateau. Other models, like SimSiam or NNCLR, take longer to warm up but then
  catch up when training for 800 epochs. This can also be seen in the 
  figure below.
  

.. figure:: images/cifar10_benchmark_knn_accuracy_800_epochs.png
    :align: center
    :alt: kNN accuracy on test set of models trained for 800 epochs

    kNN accuracy on test set of models trained for 800 epochs with batch size 
    512.

Interactive plots of the 800 epoch accuracy and training loss are hosted on
`tensorboard <https://tensorboard.dev/experiment/2XsJe3Y4TWCQSzHyDFaPQA>`__.

You can reproduce the benchmarks using the following script:
:download:`cifar10_benchmark.py <benchmarks/cifar10_benchmark.py>` 


Imagenet100
-----------

Imagenet100 is a subset of the popular ImageNet-1k dataset. It consists of 100 classes
with 1300 training and 50 validation images per class. We train the
self-supervised models from scratch on the training data. At the end of every
epoch we embed all training images and use the features for a kNN classifier 
with k=20 on the test set. The reported kNN test accuracy is the max accuracy
over all epochs the model reached. All experiments use the same ResNet-18 backbone and
with the default ImageNet-1k training parameters from the respective papers.


.. csv-table:: Imagenet100 benchmark results showing kNN test accuracy, runtime and peak GPU memory consumption for different training setups.
  :header: "Model", "Epochs", "Batch Size", "Accuracy", "Runtime", "GPU Memory"
  :widths: 20, 20, 20, 20, 20, 20

  "BarlowTwins", 200, 256, 0.465, "1319.3 Min", "11.3 GByte"
  "BYOL", 200, 256, 0.439, "1315.4 Min", "12.9 GByte"
  "DINO", 200, 256, 0.518, "1868.5 Min", "17.4 GByte"
  "FastSiam", 200, 256, 0.559, "1856.2 Min", "22.0 GByte"
  "Moco", 200, 256, 0.560, "1314.2 Min", "13.1 GByte"
  "NNCLR", 200, 256, 0.453, "1198.6 Min", "11.8 GByte"
  "SimCLR", 200, 256, 0.469, "1207.7 Min", "11.3 GByte"
  "SimSiam", 200, 256, 0.534, "1175.0 Min", "11.1 GByte"
  "SwaV", 200, 256, 0.678, "1569.2 Min", "16.9 GByte"

You can reproduce the benchmarks using the following script:
:download:`imagenet100_benchmark.py <benchmarks/imagenet100_benchmark.py>` 


Next Steps
----------

Now that you understand the performance of the different lightly methods how about
looking into a tutorial to implement your favorite model?

- :ref:`input-structure-label`
- :ref:`lightly-moco-tutorial-2`
- :ref:`lightly-simclr-tutorial-3`  
- :ref:`lightly-simsiam-tutorial-4`
- :ref:`lightly-custom-augmentation-5`