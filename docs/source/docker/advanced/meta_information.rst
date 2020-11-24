Advanced
===================================
Here you learn more advanced usage patterns of Lightly Docker.

Depending on your current setup one of the following topics might interest you:

- I have a dataset but I want lightly to "ignore" certain Samples.
  --> Mask Samples

- I have an existing dataset and want to add only relevant new data.
  --> Use Pre-Selected Samples

- I have my own (weak) labels. Can lightly use this information to improve
  the selection? --> Custom Labels


Mask Samples
-----------------------------------

You can also add masking information to prevent certain samples from being
used to the .csv file. 

The following example shows a dataset in which the column "masked" is used
to prevent Lightly Docker from using this specific sample. In this example,
img-1.jpg is simply ignored and not considered for sampling. E.g. the sample
neither gets selected nor is it affecting selection of any other sample.

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
additional samples which are enriching the existing selection.


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


Custom Labels
-----------------------------------

You can always add custom embeddings to the dataset by following the guide
here: :ref:`lightly-custom-labels` 