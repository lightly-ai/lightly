"""

.. _lightly-tutorial-export-labelstudio:

Tutorial 10: Export to LabelStudio
=============================================

This tutorial shows how you can easily label all images of a tag in Lightly
using the open-source data labeling tool `LabelStudio <labelstud.io>`

What you will learn
--------------------

* Export a tag in the LabelStudio-compatible task description format
* Import the task description into LabelStudio

Requirements
-------------
You have a dataset in the Lightly Webapp and optionally already chosen
a subset of it and created a tag for it. Now you want to label all images
in this tag using LabelStudio.

If you don't have such a dataset yet, you can use any dataset (e.g. the
playground dataset) or follow one of the other tutorials to create one.
The `tutorial on how to diversify sunflowers <tutorial_sunflowers.rst>`
is particularly well-suited.

Launch LabelStudio
------------------
Follow the documentation to `install and start Labelstudio <https://labelstud.io/guide/index.html#Quick-start>`.
If you are successfull,

Export a tag in the LabelStudio format
------------------
Just head over to the *Download* tag to see the different download options.
Choose 'Export Reduced Dataset', an expiration duration giving you
enough time to label all image and choose the form 'json'.
The tasks include a url pointing to the real images, thus allowing everyone
with the link to access the images. This is needed for LabelStudio to access the
images without needing to login.
After clicking on 'Export to LabelStudio' tasks, they are downloaded
as a single json file to your PC


Launch LabelStudio
------------------

Now you can use this diverse subset for your machine learning project.
Just head over to the *Download* tag to see the different download options.
Apart from downloading the filenames or the images directly, you can also
use the lightly-download command to copy the files in the subset from your existing
to a new directory. The CLI command with prefilled arguments is already provided.


"""
