
.. _lightly-tutorial-label-studio-al

Tutorial 7: Lightly and LabelStudio for active learning
=============================================

This tutorial combines Lightly and LabelStudio for showing a complete workflow
of creating a machine learning model including Active Learning. It is hosted
in a separate github repo: https://github.com/lightly-ai/Lightly_LabelStudio_AL


1. It starts with collecting unlabelled data.
2. Then it uses Lightly to choose a subset of the unlabelled to be labelled.
3. This subset is labelled with the help of LabelStudio.
4. A machine learning model is trained on the labeled data and Active
5. Learning is used to choose the next batch to be labelled.
6. This batch is labelled again in LabelStudio.
7. The machine learning model is trained on the updated labelled dataset.
8. Finally, the model is used to predict on completely new data.
