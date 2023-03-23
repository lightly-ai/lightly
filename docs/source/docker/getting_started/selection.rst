.. _worker-selection:

Selection
=========

Lightly allows you to specify the subset to be selected based on several objectives.

E.g. you can specify that the images in the subset should be visually diverse, be images the model struggles with (active learning),
should only be sharp images, or have a certain distribution of classes, e.g. be 50% from sunny, 30% from cloudy and 20% from rainy weather.

Each of these objectives is defined by a `strategy`. A strategy consists of two parts:

- The :code:`input` defines which data the objective is defined on. This data is either a scalar number or a vector for each sample in the dataset.
- The :code:`strategy` itself defines the objective to apply on the input data.

Lightly allows you to specify several objectives at the same time. The algorithms try to fulfil all objectives simultaneously.

Lightly's data selection algorithms support four types of input:

- **Embeddings** computed using `our open source framework for self-supervised learning <https://github.com/lightly-ai/lightly>`_
- **Lightly metadata** are metadata of images like the sharpness and computed out of the images themselves by Lightly.
- (Optional)  :ref:`Model predictions <docker-datasource-predictions>` such as classifications, object detections or segmentations
- (Optional) :ref:`Custom metadata <docker-datasource-metadata>` can be anything you can encode in a json file (from numbers to categorical strings)

Prerequisites
-------------

In order to use the selection feature, you need to

- Start the Lightly Worker in worker mode. See :ref:`worker-register`.

- Set up a dataset in the Lightly Platform with a cloud storage as datasource. See :ref:`worker-creating-a-dataset`

Scheduling a Lightly Worker run with selection
----------------------------------------------

For scheduling a Lightly Worker run with a specific selection,
you can use the python client and its :py:meth:`schedule_compute_worker_run <lightly.api.api_workflow_client.ApiWorkflowClient.schedule_compute_worker_run>` method.
You specify the selection with the :code:`selection_config` argument.
See :ref:`worker-scheduling-a-job` for reference.

Here is an example for scheduling a Lightly worker run with a specific selection configuration:

.. literalinclude:: ../integration/examples/trigger_job.py



Selection Configuration
-----------------------

The configuration of a selection needs to specify both the maximum number of samples to select and the strategies:

.. code-block:: python

    {
        "n_samples": 50,
        "proportion_samples": 0.1
        "strategies": [
            {
                "input": {
                    "type": ...
                },
                "strategy": {
                    "type": ...
                }
            },
            ... more strategies
        ]
    }

The variable :code:`n_samples` must be a positive integer specifying the absolute number of samples which should be selected.
Alternatively to :code:`n_samples`, you can also set :code:`proportion_samples` to set the number of samples to be selected relative to the input dataset size.
E.g. set it to `0.1` to select 10% of all samples.
Please set either one or the other. Setting both or none of them will cause an error.

Each strategy is specified by a :code:`dictionary`, which is always made up of an :code:`input` and the actual :code:`strategy`.

.. code-block:: python

    {
        "input": {
            "type": ...
        },
        "strategy": {
            "type": ...
        }
    },


Selection Input
^^^^^^^^^^^^^^^^

The input can be one of the following:

.. tabs::

    .. tab:: EMBEDDINGS

        The `lightly OSS framework for self supervised learning <https://github.com/lightly-ai/lightly>`_ is used to compute the embeddings.
        They are a vector of numbers for each sample.
        
        You can define embeddings as input using:

        .. code-block:: python

            "input": {
                "type": "EMBEDDINGS"
            }

        You can also use embeddings from other datasets to create strategies such as 
        similarity search:

        .. code-block:: python

            "input": {
                "type": "EMBEDDINGS",
                "dataset_id": "DATASET_ID_OF_THE_QUERY_IMAGES",
                "tag_name": "TAG_NAME_OF_THE_QUERY_IMAGES" # e.g. "initial-tag"
            },

    .. tab:: SCORES

        They are a scalar number for each element. They are **specified by the prediction task and the scorer**:

        .. code-block:: python

            # using your own predictions
            "input": {
                "type": "SCORES",
                "task": "YOUR_TASK_NAME",
                "score": "uncertainty_entropy"
            }

            # using the lightly pretagging model
            "input": {
                "type": "SCORES",
                "task": "lightly_pretagging",
                "score": "uncertainty_entropy"
            }

        You can specify one of the tasks you specified in your datasource, see :ref:`docker-datasource-predictions` for reference.
        Alternatively, set the task to **lightly_pretagging** to use object detections created by the Lightly Worker itself.
        See :ref:`docker-pretagging` for reference.


    .. tab:: PREDICTIONS

        .. _worker-selection-predictions:

        The class distribution probability vector of predictions can be used as well. Here, three cases have to be distinguished:

            - **Image Classification**: The probability vector of each sample's prediction is used directly.

            - **Object Detection**: The probability vectors of the class predictions of all objects in an image are summed up.

            - **Object Detection** and using the :ref:`docker-object-level`: Each sample is a cropped object and has a single object prediction, whose probability vector is used.

        This input is **specified using the prediction task**. Furthermore, it should be remembered, which class names are used for this task, as they are needed in later steps.
        
        If you use your own predictions (see :ref:`docker-datasource-predictions`), the task name and class names are taken from the specification in the prediction `schema.json`.
        
        Alternatively, set the task to **lightly_pretagging** to use object detections created by the Lightly Worker itself.
        Its class names are specified here: :ref:`docker-pretagging`.


        .. code-block:: python
            
            # using your own predictions
            "input": {
                "type": "PREDICTIONS",
                "task": "my_object_detection_task",
                "name": "CLASS_DISTRIBUTION"
            }

            # using the lightly pretagging model
            "input": {
                "type": "PREDICTIONS",
                "task": "lightly_pretagging",
                "name": "CLASS_DISTRIBUTION"
            }

    .. tab:: METADATA

        Metadata is specified by the metadata key. It can be divided across two dimensions:

        - **Custom Metadata** vs. **Lightly Metadata**

            **Custom Metadata** must be specified when creating a datasource and you must have uploaded metadata to it.
            See :ref:`docker-datasource-metadata` for reference. An example configuration:

            .. code-block:: python

                "input": {
                    "type": "METADATA",
                    "key": "weather.temperature"
                }

            Use as key the “path” you specified when creating the metadata in the datasource.


            **Lightly Metadata**, is calculated by the Lightly Worker. It is specified by prepending :code:`lightly` to the key. 
            An example configuration:

            .. code-block:: python

                "input": {
                    "type": "METADATA",
                    "key": "lightly.sharpness"
                }

            Currently supported metadata are :code:`sharpness`, :code:`snr` (signal-to-noise-ratio) and :code:`sizeInBytes`.
            If your use case would profit from more metadata computed out of the image, please reach out to us.

        - **Numerical** vs. **Categorical** values

            Not all metadata types can be used in all selection strategies. Lightly differentiates between numerical and categorical metadata.

            **Numerical** metadata are numbers (int, float), e.g. `lightly.sharpness` or `weather.temperature`. It is usually real-valued.
            
            **Categorical** metadata is from a discrete number of categories, e.g. `video.location_id` or `weather.description`.
            It can be either an integer or a string.


Selection Strategy
^^^^^^^^^^^^^^^^^^^

There are several types of selection strategies, all trying to reach different objectives.

.. tabs::

    .. tab:: DIVERSITY

        Use this strategy to **select samples such that they are as different as possible from each other**.

        Can be used with **EMBEDDINGS**. 
        Samples with a high distance between their embeddings are 
        considered to be more *different* from each other than samples with a 
        low distance. The strategy is specified like this:

        .. code-block:: python

            "strategy": {
                "type": "DIVERSITY"
            }

        If you want to preserve a minimum distance between chosen samples, you 
        can specify it as an additional stopping condition. The selection process
        will stop as soon as one of the stopping criteria has been reached.

        .. code-block:: python
            :emphasize-lines: 3

            "strategy": {
                "type": "DIVERSITY",
                "stopping_condition_minimum_distance": 0.2
            }

        Setting :code:`"stopping_condition_minimum_distance": 0.2` will remove all samples which are
        closer to each other than 0.2. 
        This allows you to specify the minimum allowed distance between two images in the output dataset.
        If you use embeddings as input, this value should be between 0 and 2.0, as the embeddings are normalized to unit length.
        This is often a convenient method when working with different data sources and trying to combine them in a balanced way.
        If you want to use this stopping condition to stop the selection early,
        make sure that you allow selecting enough samples by setting :code:`n_samples` or :code:`proportion_samples` high enough.

        .. note:: Higher minimum distance in the embedding space results in more
                  diverse images being selected. Furthermore, increasing the
                  minimum distance will result in fewer samples being selected.

    .. tab:: WEIGHTS

        The objective of this strategy is to **select samples that have a high numerical value**.
        
        Can be used with **SCORES** and **numerical METADATA**. It can be specified with:

        .. code-block:: python

            "strategy": {
                "type": "WEIGHTS"
            }

    .. tab:: THRESHOLD

        The objective of this strategy is to only **select samples that have a numerical value fulfilling a threshold criterion**.
        E.g. they should be bigger or smaller than a certain value.

        Can be used with **SCORES** and **numerical METADATA**. It is specified as follows:

        .. code-block:: python

            "strategy": {
                "type": "THRESHOLD",
                "threshold": 20,
                "operation": "BIGGER_EQUAL"
            }

        This will keep all samples whose value (specified by the input) is >= 20 and remove all others.
        The allowed operations are :code:`SMALLER`, :code:`SMALLER_EQUAL`, :code:`BIGGER`, :code:`BIGGER_EQUAL`.

    .. tab:: BALANCE

        The objective of this strategy is to **select samples such that the distribution of classes in them is as close to a target distribution as possible**.

        E.g. the samples chosen should have 50% sunny and 50% rainy weather.
        Or, the objects of the samples chosen should be 40% ambulance and 60% buses.

        Can be used with **PREDICTIONS** and **categorical METADATA**.

        .. code-block:: python

            "strategy": {
                "type": "BALANCE",
                "target": {
                    "Ambulance": 0.4, # `Ambulance` should be a valid class in your `schema.json`
                    "Bus": 0.6
                }
            }

        If the values of the target do not sum up to 1, the remainder is assumed to be the target for the other classes.
        For example, if we would set the target to 20% ambulance and 40% bus, there is the implicit assumption, that the remaining 40% should come from any other class,
        e.g. from cars, bicycles or pedestrians.

        Note that not specified classes do not influence the selection process!

    .. tab:: SIMILARITY

        With this strategy you can use the input emebeddings from another dataset
        to **select similar images**. This can be useful if you are looking for more 
        examples of certain edge cases.

        Can be used with **EMBEDDINGS**.

        .. code-block:: python

            "strategy": {
                "type": "SIMILARITY",
            }


Configuration Examples
----------------------

Here are examples for the full configuration including the input for several objectives:

.. dropdown:: Visual Diversity (CORESET)

    Choosing 100 samples that are visually diverse equals diversifying samples based on their embeddings:

    .. code-block:: python

        {
            "n_samples": 100, # set to the number of samples you want to select
            "strategies": [
                {
                    "input": {
                        "type": "EMBEDDINGS"
                    },
                    "strategy": {
                        "type": "DIVERSITY"
                    }
                }
            ]
        }


.. dropdown:: Active Learning

    Active Learning equals weighting samples based on active learning scores:

    .. code-block:: python

        {
            "n_samples": 100, # set to the number of samples you want to select
            "strategies": [
                {
                    "input": {
                        "type": "SCORES",
                        "task": "my_object_detection_task", # change to your task
                        "score": "uncertainty_entropy" # change to your preferred score
                    },
                    "strategy": {
                        "type": "WEIGHTS"
                    }
                }
            ]
        }

    .. note:: This works as well for Image Classifciation or Segmentation!

.. dropdown:: Visual Diversity and Active Learning (CORAL)

    For combining two strategies, just specify both of them:

    .. code-block:: python

        {
            "n_samples": 100, # set to the number of samples you want to select
            "strategies": [
                {
                    "input": {
                        "type": "EMBEDDINGS"
                    },
                    "strategy": {
                        "type": "DIVERSITY"
                    }
                },
                {
                    "input": {
                        "type": "SCORES",
                        "task": "my_object_detection_task", # change to your task
                        "score": "uncertainty_entropy" # change to your preferred score
                    },
                    "strategy": {
                        "type": "WEIGHTS"
                    }
                }
            ]
        }

.. dropdown:: Metadata Thresholding

    This can be used to remove e.g. blurry images, which equals selecting
    samples whose sharpness is over a threshold:

    .. code-block:: python

        {
            "n_samples": 100, # set to the number of samples you want to select
            "strategies": [
                {
                    "input": {
                        "type": "METADATA",
                        "key": "lightly.sharpness"
                    },
                    "strategy": {
                        "type": "THRESHOLD",
                        "threshold": 20,
                        "operation": "BIGGER"
                    }
                }
            ]
        }

.. dropdown:: Object Balancing

    Use lightly pretagging to get the objects, then specify a target distribution of classes:

    .. code-block:: python

        {
            "n_samples": 100, # set to the number of samples you want to select
            "strategies": [
                {
                    "input": {
                        "type": "PREDICTIONS",
                        "task": "lightly_pretagging", # (optional) change to your task
                        "name": "CLASS_DISTRIBUTION"
                    },
                    "strategy": {
                        "type": "BALANCE",
                        "target": {
                            "car": 0.1,
                            "bicycle": 0.5,
                            "bus": 0.1,
                            "motorcycle": 0.1,
                            "person": 0.1,
                            "train": 0.05,
                            "truck": 0.05
                        }
                    }
                }
            ]
        }

    .. note:: To use the `lightly pretagging` you need to enable it using :code:`'pretagging': True` in the
              worker config. See :ref:`docker-pretagging` for reference.

.. dropdown:: Metadata Balancing

    Let’s assume you have specified metadata with the path `weather.description`
    and want your selected subset to have 20%  sunny, 40% cloudy and the rest other images:

    .. code-block:: python

        {
            "n_samples": 100, # set to the number of samples you want to select
            "strategies": [
                {
                    "input": {
                        "type": "METADATA",
                        "key": "weather.description"
                    },
                    "strategy": {
                        "type": "BALANCE",
                        "target": {
                            "sunny": 0.2,
                            "cloudy": 0.4
                        }
                    }
                }
            ]
        }

.. dropdown:: Similarity Search

    To perform simlarity search you need a dataset and tag
    consisting of the query images.

    We can then use the following configuration to find similar images from the
    input dataset. This example will select 100 images from the input dataset that 
    are the most similar to the images in the tag from the query dataset.

    .. code-block:: python

        {
            "n_samples": 100, # put your number here
            "strategies": [
                {
                    "input": {
                        "type": "EMBEDDINGS",
                        "dataset_id": "DATASET_ID_OF_THE_QUERY_IMAGES", 
                        "tag_name": "TAG_NAME_OF_THE_QUERY_IMAGES" # e.g. "initial-tag"
                    },
                    "strategy": {
                        "type": "SIMILARITY",
                    }
                }
            ]
        }

Application of Strategies
-------------------------

Generally, the order in which the different strategies were defined in the config does not matter.
In a first step, all the thresholding strategies are applied.
In the next step, all other strategies are applied in parallel.

.. note:: Note that different taskes can also be combined. E.g. you can use predictions 
          from "my_weather_classification_task" for one strategy combined with predictions from
          "my_object_detection_task" from another strategy.

The Lightly optimizer tries to fulfil all strategies as good as possible. 
**Potential reasons why your objectives were not satisfied:**

- **Tradeoff between different objectives.**
  The optimizer always has to tradeoff between different objectives.
  E.g. it may happen that all samples with high WEIGHTS are close together. If you also specified the objective DIVERSITY, then only a few of these high-weight samples
  may be chosen. Instead, also other samples that are more diverse, but have lower weights, are chosen.

- **Restrictions in the input dataset.**
  This applies especially for BALANCE: E.g. if there are only 10 images of ambulances in the input dataset and a total
  of 1000 images are selected, the output can only have a maximum of 1% ambulances. Thus a BALANCE target of having 20% ambulances cannot be fulfilled.

- **Too little samples to choose.**
  If the selection algorithm can only choose a small number of samples, it may not be possible to fulfil the objectives.
  You can solve this by increasing :code:`n_samples` or :code:`proportion_samples`.

Selection on object level
-------------------------

Lightly supports doing selection on :ref:`docker-object-level`.

While embeddings are fully available, there are some limitations regarding the usage of METADATA and predictions for SCORES and PREDICTIONS as input:

- When using the object level workflow, the object detections used to create the object crops out of the images are available and can be used for both the SCORES and PREDICTIONS input. However, predictions from other tasks are NOT available at the moment.

- Lightly metadata is generated on the fly for the object crops and can thus be used for selection. However, other metadata is on image level and thus NOT available at the moment.

If your use case would profit from using image-level data for object-level selection, please reach out to us.
