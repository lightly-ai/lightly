Benchmarks 
===================================
We show benchmarks of the different models for self-supervised learning
and their performance on public datasets.


We have benchmarks we regularly update for these datasets:

- `Imagenet`_
- `Imagenet100`_
- `ImageNette`_
- `CIFAR-10`_

ImageNet
--------

We use the ImageNet1k ILSVRC2012 split provided here: https://image-net.org/download.php.

Self-supervised training of a SimCLR model for 100 epochs with total batch size 256
takes about four days including evaluation on two GeForce RTX 4090 GPUs. You can reproduce the results with
the code at `benchmarks/imagenet/resnet50 <https://github.com/lightly-ai/lightly/tree/master/benchmarks/imagenet/resnet50>`_.

Evaluation settings are based on these papers:

- Linear: `SimCLR <https://arxiv.org/abs/2002.05709>`_
- Finetune: `SimCLR <https://arxiv.org/abs/2002.05709>`_
- KNN: `InstDisc <https://arxiv.org/abs/1805.01978>`_

See the `benchmarking scripts <https://github.com/lightly-ai/lightly/tree/master/benchmarks/imagenet/resnet50>`_ for details.


.. csv-table:: Imagenet benchmark results.
  :header: "Model", "Backbone", "Batch Size", "Epochs", "Linear Top1", "Linear Top5", "Finetune Top1", "Finetune Top5", "KNN Top1", "KNN Top5", "Tensorboard", "Checkpoint"
  :widths: 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20

  "BarlowTwins", "Res50", "256", "100", "62.9", "84.3", "72.6", "90.9", "45.6", "73.9", "`link <https://tensorboard.dev/experiment/NxyNRiQsQjWZ82I9b0PvKg/>`_", "`link <https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_barlowtwins_2023-08-18_00-11-03/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt>`_"
  "BYOL", "Res50", "256", "100", "62.4", "84.7", "74.0", "91.9", "45.6", "74.8", "`link <https://tensorboard.dev/experiment/Z0iG2JLaTJe5nuBD7DK1bg>`_", "`link <https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_byol_2023-07-10_10-37-32/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt>`_"
  "DINO", "Res50", "128", "100", "68.2", "87.9", "72.5", "90.8", "49.9", "78.7", "`link <https://tensorboard.dev/experiment/DvKHX9sNSWWqDrRksllPLA>`_", "`link <https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dino_2023-06-06_13-59-48/pretrain/version_0/checkpoints/epoch%3D99-step%3D1000900.ckpt>`_"
  "SimCLR*", "Res50", "256", "100", "63.2", "85.2", "73.9", "91.9", "44.8", "73.9", "`link <https://tensorboard.dev/experiment/Ugol97adQdezgcVibDYMMA>`_", "`link <https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_simclr_2023-06-22_09-11-13/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt>`_"
  "SimCLR* + DCL", "Res50", "256", "100", "65.1", "86.2", "73.5", "91.7", "49.6", "77.5", "`link <https://tensorboard.dev/experiment/k4ZonZ77QzmBkc0lXswQlg>`_", "`link <https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dcl_2023-07-04_16-51-40/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt>`_"
  "SimCLR* + DCLW", "Res50", "256", "100", "64.5", "86.0", "73.2", "91.5", "48.5", "76.8", "`link <https://tensorboard.dev/experiment/TrALnpwFQ4OkZV3uvaX7wQ>`_", "`link <https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dclw_2023-07-07_14-57-13/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt>`_"
  "SwAV", "Res50", "256", "100", "67.2", "88.1", "75.4", "92.7", "49.5", "78.6", "`link <https://tensorboard.dev/experiment/Ipx4Oxl5Qkqm5Sl5kWyKKg>`_", "`link <https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_swav_2023-05-25_08-29-14/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt>`_"

*\*We use square root learning rate scaling instead of linear scaling as it yields better results for smaller batch sizes. See Appendix B.1 in SimCLR paper.*


ImageNette
-----------------------------------

We use the ImageNette dataset provided here: https://github.com/fastai/imagenette

For our benchmarks we use the 160px version and resize the input images to 128 pixels. 
Training a single model for 800 epochs on a A6000 GPU takes about 3-5 hours.


.. csv-table:: ImageNette benchmark results using kNN evaluation on the test set using 128x128 input resolution.
  :header: "Model", "Batch Size", "Epochs", "KNN Test Accuracy", "Runtime", "GPU Memory"
  :widths: 20, 20, 20, 20, 20, 20

  "BarlowTwins", "256", "800", "0.852", "298.5 Min", "4.0 GByte"
  "BYOL", "256", "800", "0.887", "214.8 Min", "4.3 GByte"
  "DCL", "256", "800", "0.861", "189.1 Min", "3.7 GByte"
  "DCLW", "256", "800", "0.865", "192.2 Min", "3.7 GByte"
  "DINO (Res18)", "256", "800", "0.888", "312.3 Min", "6.6 GByte"
  "FastSiam", "256", "800", "0.873", "299.6 Min", "7.3 GByte"
  "MAE (ViT-S)", "256", "800", "0.610", "248.2 Min", "4.4 GByte"
  "MSN (ViT-S)", "256", "800", "0.828", "515.5 Min", "14.7 GByte"
  "Moco", "256", "800", "0.874", "231.7 Min", "4.3 GByte"
  "NNCLR", "256", "800", "0.884", "212.5 Min", "3.8 GByte"
  "PMSN (ViT-S)", "256", "800", "0.822", "505.8 Min", "14.7 GByte"
  "SimCLR", "256", "800", "0.889", "193.5 Min", "3.7 GByte"
  "SimMIM (ViT-B32)", "256", "800", "0.343", "446.5 Min", "9.7 GByte"
  "SimSiam", "256", "800", "0.872", "206.4 Min", "3.9 GByte"
  "SwaV", "256", "800", "0.902", "283.2 Min", "6.4 GByte"
  "SwaVQueue", "256", "800", "0.890", "282.7 Min", "6.4 GByte"
  "SMoG", "256", "800", "0.788", "232.1 Min", "2.6 GByte"
  "TiCo", "256", "800", "0.856", "177.8 Min", "2.5 GByte"
  "VICReg", "256", "800", "0.845", "205.6 Min", "4.0 GByte"
  "VICRegL", "256", "800", "0.778", "218.7 Min", "4.0 GByte"

You can reproduce the benchmarks using the following script:
:download:`imagenette_benchmark.py <benchmarks/imagenette_benchmark.py>` 


CIFAR-10
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
  :header: "Model", "Batch Size", "Epochs", "KNN Test Accuracy", "Runtime", "GPU Memory"
  :widths: 20, 20, 20, 30, 20, 20

  "BarlowTwins", "128", "200", "0.842", "375.9 Min", "1.7 GByte"
  "BYOL", "128", "200", "0.869", "121.9 Min", "1.6 GByte"
  "DCL", "128", "200", "0.844", "102.2 Min", "1.5 GByte"
  "DCLW", "128", "200", "0.833", "100.4 Min", "1.5 GByte"
  "DINO", "128", "200", "0.840", "120.3 Min", "1.6 GByte"
  "FastSiam", "128", "200", "0.906", "164.0 Min", "2.7 GByte"
  "Moco", "128", "200", "0.838", "128.8 Min", "1.7 GByte"
  "NNCLR", "128", "200", "0.834", "101.5 Min", "1.5 GByte"
  "SimCLR", "128", "200", "0.847", "97.7 Min", "1.5 GByte"
  "SimSiam", "128", "200", "0.819", "97.3 Min", "1.6 GByte"
  "SwaV", "128", "200", "0.812", "99.6 Min", "1.5 GByte"
  "SMoG", "128", "200", "0.743", "192.2 Min", "1.2 GByte"
  "BarlowTwins", "512", "200", "0.819", "153.3 Min", "5.1 GByte"
  "BYOL", "512", "200", "0.868", "108.3 Min", "5.6 GByte"
  "DCL", "512", "200", "0.840", "88.2 Min", "4.9 GByte"
  "DCLW", "512", "200", "0.824", "87.9 Min", "4.9 GByte"
  "DINO", "512", "200", "0.813", "108.6 Min", "5.0 GByte"
  "FastSiam", "512", "200", "0.788", "146.9 Min", "9.5 GByte"
  "Moco (*)", "512", "200", "0.847", "112.2 Min", "5.6 GByte"
  "NNCLR (*)", "512", "200", "0.815", "88.1 Min", "5.0 GByte"
  "SimCLR", "512", "200", "0.848", "87.1 Min", "4.9 GByte"
  "SimSiam", "512", "200", "0.764", "87.8 Min", "5.0 GByte"
  "SwaV", "512", "200", "0.842", "88.7 Min", "4.9 GByte"
  "SMoG", "512", "200", "0.686", "110.0 Min", "3.4 GByte"
  "BarlowTwins", "512", "800", "0.859", "517.5 Min", "7.9 GByte"
  "BYOL", "512", "800", "0.910", "400.9 Min", "5.4 GByte"
  "DCL", "512", "800", "0.874", "334.6 Min", "4.9 GByte"
  "DCLW", "512", "800", "0.871", "333.3 Min", "4.9 GByte"
  "DINO", "512", "800", "0.848", "405.2 Min", "5.0 GByte"
  "FastSiam", "512", "800", "0.902", "582.0 Min", "9.5 GByte"
  "Moco (*)", "512", "800", "0.899", "417.8 Min", "5.4 GByte"
  "NNCLR (*)", "512", "800", "0.892", "335.0 Min", "5.0 GByte"
  "SimCLR", "512", "800", "0.879", "331.1 Min", "4.9 GByte"
  "SimSiam", "512", "800", "0.904", "333.7 Min", "5.1 GByte"
  "SwaV", "512", "800", "0.884", "330.5 Min", "5.0 GByte"
  "SMoG", "512", "800", "0.800", "415.6 Min", "3.2 GByte"

(*): Increased size of memory bank from 4096 to 8192 to avoid too quickly 
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
  :header: "Model", "Batch Size", "Epochs", "KNN Test Accuracy", "Runtime", "GPU Memory"
  :widths: 20, 20, 20, 20, 20, 20

  "BarlowTwins", "256", "200", "0.465", "1319.3 Min", "11.3 GByte"
  "BYOL", "256", "200", "0.439", "1315.4 Min", "12.9 GByte"
  "DINO", "256", "200", "0.518", "1868.5 Min", "17.4 GByte"
  "FastSiam", "256", "200", "0.559", "1856.2 Min", "22.0 GByte"
  "Moco", "256", "200", "0.560", "1314.2 Min", "13.1 GByte"
  "NNCLR", "256", "200", "0.453", "1198.6 Min", "11.8 GByte"
  "SimCLR", "256", "200", "0.469", "1207.7 Min", "11.3 GByte"
  "SimSiam", "256", "200", "0.534", "1175.0 Min", "11.1 GByte"
  "SwaV", "256", "200", "0.678", "1569.2 Min", "16.9 GByte"

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