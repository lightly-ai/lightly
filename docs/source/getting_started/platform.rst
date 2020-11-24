
The Lightly Platform
===================================

The lightly framework itself allows you to use self-supervised learning
in a very simple way and even create embeddings of your dataset.
However, we can do much more than just train and embed datasets. The Python
framework acts as an interface to the Lightly data curation platform. 
Once you have an embedding of an unlabeled dataset you might still require
some labels to train a model. But which samples do you pick for labeling and 
training a model?

This is exactly where the platform comes in. The platform helps you analyze your
dataset and using various methods pick the relevant samples for your task.


.. _my-reference-label:

Authentication Token
-----------------------------------

In order to authenticate yourself on the platform when using the pip package
we provide you with an authentication token. You can retrieve
it when creating a new dataset or when clicking on your 
account (top right)-> preferences on the 
`application <https://app.lightly.ai>`_.

Keep the token for yourself and don't share it. Anyone with the
token could access your datasets!