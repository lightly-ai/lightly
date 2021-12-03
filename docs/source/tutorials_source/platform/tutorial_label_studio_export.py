"""

.. _lightly-tutorial-export-labelstudio:

Tutorial 10: Export to LabelStudio
=============================================

This tutorial shows how you can easily label all images of a tag in Lightly
using the open-source data labeling tool `LabelStudio <labelstud.io>`_

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
The `tutorial on how to diversify your dataset <./tutorial_sunflowers.rst>`_
is particularly well-suited.

Launch LabelStudio
------------------
Follow the documentation to `install and start Labelstudio <https://labelstud.io/guide/index.html#Quick-start>`_.
Then create a new project and click on import. Now you should be in the
import screen.

.. figure:: ../../tutorials_source/platform/images/tutorial_export_labelstudio/labelstudio_import_dialog.jpg
    :align: center
    :alt: Import dialog of LabelStudio.

Export a tag in the LabelStudio format
------------------
Now go again to the tab with the Lightly webapp.
Just head over to the *Download* tag to see the different download options.
Choose 'Export Reduced Dataset', an expiration duration giving you
enough time to label all image and choose the form 'json'.
The tasks include a url pointing to the real images, thus allowing everyone
with the link to access the images. This is needed for LabelStudio to access the
images without needing to login.
After clicking on 'Export to LabelStudio Tasks', they are downloaded
as a single json file to your PC.


Import the tag into LabelStudio
------------------

Head over to the tab with LabelStudio open and import the file you just
downloaded. Either per drag-n-drop or the import dialogue. Then finish
the import.

.. figure:: ../../tutorials_source/platform/images/tutorial_export_labelstudio/labelstudio_imported_file.jpg
    :align: center
    :alt: Imported file into LabelStudio.

Start labeling
------------------

Now you can start labeling your images! To see them, you might need
to change the type of the image column to 'img'.

.. figure:: ../../tutorials_source/platform/images/tutorial_export_labelstudio/labelstudio_import_finished.jpg
    :align: center
    :alt: LabelStudio tasks fully imported and showing images.



"""
