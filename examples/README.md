# Examples

We provide example implementations for self-supervised learning models for PyTorch and PyTorch Lightning to give you a headstart when implementing your own model! 

Note that we adapted the parameters to make the examples easily run on a machine with a single GPU. The examples are not optimized for efficiency, accuracy or distributed training. Please consult the reference publications for the respective models for the optimal settings.


All examples can be run from the terminal with:

```
python <path to example.py>
```

The dataset can be downloaded with:

```
git clone https://github.com/alexeygrigorev/clothing-dataset /datasets/clothing-dataset
```

The examples should also run on [Google Colab](https://colab.research.google.com/). Remember to activate the GPU otherwise training will be very slow! You can simply copy paste the code and add the following lines at the beginning of the notebook to install lightly and download the data:

```
!pip install lightly
!git clone https://github.com/alexeygrigorev/clothing-dataset /datasets/clothing-dataset

# add code from example below
```


You can find additional information for each model in our [Documentation](https://docs.lightly.ai//examples/models.html#)
