Meta Information
======================

Depending on your current setup one of the following topics might interest you:

- | You have a dataset but want lightly to "ignore" certain Samples.
  | --> `Mask Samples`_

- | You have an existing dataset and want to add only relevant new data.
  | --> `Use Pre-Selected Samples`_

- | You have your own (weak) labels. Can lightly use this information to improve
    the selection? 
  | --> `Custom Labels`_


Mask Samples
-----------------------------------

You can also add masking information to prevent certain samples from being
used to the .csv file. 

The following example shows a dataset in which the column "masked" is used
to prevent Lightly Docker from using this specific sample. In this example,
img-1.jpg is simply ignored and not considered for sampling. E.g. the sample
neither gets selected nor is it affecting the selection of any other sample.

.. list-table:: masked_embeddings.csv
   :widths: 50 50 50 50 50
   :header-rows: 1

   * - filenames
     - embedding_0
     - embedding_1
     - masked
     - labels
   * - img-1.jpg
     - 0.1
     - 0.5
     - 1
     - 0
   * - img-2.jpg
     - 0.2
     - 0.2
     - 0
     - 0
   * - img-3.jpg
     - 0.1
     - 0.9
     - 0
     - 0


Use Pre-Selected Samples
-----------------------------------
Very similar to masking samples we can also pre-select specific samples. This 
can be useful for semi-automated data selection processes. A human annotator
can pre-select some of the relevant samples and let Lightly Docker add only
additional samples that are enriching the existing selection.


.. list-table:: selected_embeddings.csv
   :widths: 50 50 50 50 50
   :header-rows: 1

   * - filenames
     - embedding_0
     - embedding_1
     - selected
     - labels
   * - img-1.jpg
     - 0.1
     - 0.5
     - 0
     - 0
   * - img-2.jpg
     - 0.2
     - 0.2
     - 0
     - 0
   * - img-3.jpg
     - 0.1
     - 0.9
     - 1
     - 0

.. note:: Pre-selected samples also count for the target number of samples.
          For example, you have a dataset with 100 samples. If you pre-select
          60 and want to sample 50, sampling would have no effect since there
          are already more than 50 samples selected.

Custom Labels
-----------------------------------

You can always add custom embeddings to the dataset by following the guide
here: :ref:`lightly-custom-labels`