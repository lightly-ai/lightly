"""

.. _lightly-tutorial-export-labelstudio:

Tutorial 10: Export to LabelStudio
=============================================

This tutorial shows how you can easily label all images of a tag from Lightly
using the open-source data labeling tool `LabelStudio <https://labelstud.io>`_.

What you will learn
--------------------

* Export a tag from Lightly in the `LabelStudio format <https://labelstud.io/guide/tasks.html#Basic-Label-Studio-JSON-format>`_.
* Import the tag into LabelStudio.

Requirements
------------
You have a dataset in the `Lightly Platform <https://app.lightly.ai>`_
and optionally already chosen a subset of it and created a tag for it.
Now you want to label all images of this tag using LabelStudio.

If you have not created your own dataset yet, you can use any dataset
(e.g. the playground dataset) or follow one of the other tutorials to create one.
The :ref:`lightly-tutorial-sunflowers`
is particularly well-suited.

Launch LabelStudio
------------------
Follow the documentation to `install and start Labelstudio <https://labelstud.io/guide/index.html#Quick-start>`_.
Then create a new project and click on import. Now you should be in the
import screen.

.. figure:: ../../tutorials_source/platform/images/tutorial_export_labelstudio/labelstudio_import_dialog.jpg
    :align: center
    :alt: Import dialog of LabelStudio.

Export from Lightly in the LabelStudio format
---------------------------------------------
Now open your dataset in the `Lightly Platform <https://app.lightly.ai>`_.
and select the tag you want to export.
Navigate to the *Download* page to see the different download options.
Within *Export Reduced Dataset*, select *LabelStudio Tasks* from the dropdown
of the list of supported export formats. Specify an expiration duration
giving you enough time to label all images.
The tasks include a url pointing to the real images, thus allowing everyone
with the link to access the images. This is needed for LabelStudio to access the
images without needing to login.
After clicking the button 'Export to LabelStudio Tasks', they are downloaded
as a single json file to your PC.
If you only want to export from a specific tag, just select the tag on the top
before exporting.


Import the tasks into LabelStudio
---------------------------------

Now head back to LabelStudio and import the file you just
downloaded. Either per drag-n-drop or browse your local files. Then finish
the import.

.. figure:: ../../tutorials_source/platform/images/tutorial_export_labelstudio/labelstudio_imported_file.jpg
    :align: center
    :alt: Imported file into LabelStudio.

Start labeling
--------------

Now you can start labeling your images! To see them, you might need
to change the type of the image column to 'img'.

.. figure:: ../../tutorials_source/platform/images/tutorial_export_labelstudio/labelstudio_import_finished.jpg
    :align: center
    :alt: LabelStudio tasks fully imported and showing images.



"""
