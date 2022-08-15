.. _worker-selection:

Selection
=========

Lightly allows you to specify the subset to be selected based on many different objectives.
E.g. you can specify that the images in the subset should be visually diverse, be images the model struggles with,
should only be sharp images, or have a certain distribution of classes, e.g. be 50% from sunny, 30% from cloudy and 20% from rainy weather.

Each of these objectives is defined by a `strategy`. A strategy consists of two parts:

- The input tells on which data the objective is defined on. It defines a scalar number or vector for each image in the dataset.
- The strategy itself defines the objective to apply on the input data.

Lightly allows you to specify many different objectives at the same time. The algorithms tries to fulfil all objectives simultaneously.

Prerequisites
-------------

For using the selection, you must have

- Started the Lightly Worker in worker mode. See :ref:`worker-register`.

- Set up a dataset in the Lightly Platform with a cloud storage as datasource. See :ref:`worker-creating-a-dataset`

Scheduling a Lightly Worker run with selection
----------------------------------------------

For scheduling a Lightly Worker run with a specific selection,
you can use the python client and its :py:meth:`lightly.api.ApiWorkflowClient.schedule_compute_worker_run` method.
You specify the selection with the `selection_config` argument.

Here is example for scheduling a Lightly worker run with a specific selection configuration:

.. literalinclude:: ../integration/examples/trigger_job.py



Selection Configuration
-----------------------

The configuration of a selection needs to specify both the maximum number of samples to select and the strategies:

.. code-block:: python

    DockerWorkerSelectionConfig(
        n_samples=50,
        strategies=[
            DockerWorkerSelectionConfigEntry(
                input=DockerWorkerSelectionConfigEntryInput(type=DockerWorkerSelectionInputType.CHANGEME, extra parameters),
                strategy=DockerWorkerSelectionConfigEntryStrategy(type=DockerWorkerSelectionStrategyType.CHANGEME, extra parameters)
            ),
            ... more entries
        ]
    )

The variable `n_samples` must be a positive integer specifying the absolute number of samples which should be selected.
Specifying a proportion of samples (e.g. 50%) is not supported at the moment, but will be added soon.

Each strategy is specified by a `DockerWorkerSelectionConfigEntry`, which is always made up of an input and the actual strategy.
The input or `DockerWorkerSelectionConfigEntryInput` can be one of the following:

.. tabs::

    .. tab:: EMBEDDINGS

        They are a vector of numbers for each element. The can be defined easily as input:

        .. code-block:: python

            input=DockerWorkerSelectionConfigEntryInput(type=DockerWorkerSelectionInputType.EMBEDDINGS)

    .. tab:: SCORES

        They are a scalar number for each element. They are specified by the prediction task and the scorer:

        .. code-block:: python

            input=DockerWorkerSelectionConfigEntryInput(type=DockerWorkerSelectionInputType.SCORES, task="lightly_pretagging", score="uncertainty_entropy")

        You can specify one of the tasks you specified in your datasource, see :ref:`docker-datasource-predictions` for reference.
        Alternatively, you can specify "lightly_pretagging" as the task to use object detections created by the Lightly worker itself.
        See :ref:`docker-pretagging` for reference.

    .. tab:: METADATA

        Metadata is specified by the metadata key. It can be divided across two dimensions:

        - Custom metadata vs. Lightly metadata

            Custom metadata must be specified when creating a datasource and you must have uploaded metadata to it.
            See :ref:`docker-datasource-metadata` for reference. An example configuration:

            .. code-block:: python

                input=DockerWorkerSelectionConfigEntryInput(type=DockerWorkerSelectionInputType.METADATA, key="weather.temperature")

            Use as key the “path” you specified when creating the metadata in the datasource.


            Lightly metadata, on the contrary, is calculated out of image data on the fly. It is specified by appending a `.lightly` to the key.

            An example configuration:

            .. code-block:: python

                input=DockerWorkerSelectionConfigEntryInput(type=DockerWorkerSelectionInputType.METADATA, key="lightly.sharpness")

        - Numerical metadata vs. categorical metadata

            Numerical metadata is a number, e.g. `lightly.sharpness` or `weather.temperature`. It is usually real-valued.
            Categorical metadata is from a discrete number of categories, e.g. `video.location_id` or `weather.description`.
            It can be either an integer or a string.

    .. tab:: PREDICTIONS

        .. _worker-selection-predictions:

        The class distribution probability vector of predictions can be used as well. Here, three case have to be distinguished:

            - The predictions are classifications → The probability vector of each sample's prediction is used directly.

            - The predictions are object detections → The probability vector of the class predictions of all object in an image are summed up.

            - The predictions are object detections AND the selection is on object level → Each sample is a cropped object and has a single object prediction, whose probability vector is used.

        This input is specified using the task name. Futhermore, it should be remembered, which class names are used for this task, as they are needid in later steps.
        If you use your own predictions (see :ref:`docker-datasource-predictions`), the task name and class names are taken from the specification in the prediction `schema.json`.
        Alternatively, you use the Lightly pretagging and the class names are specified here: :ref:`docker-pretagging`. In that case the task name is `lightly_pretagging`.


        .. code-block:: python

            input=DockerWorkerSelectionConfigEntryInput(type=DockerWorkerSelectionInputType.PREDICTIONS, task="my_object_detection_task", name=DockerWorkerSelectionInputPredictionsName.CLASS_DISTRIBUTION))

There are several types of selection strategies, all trying to reach different objectives:

.. tabs::

    .. tab:: DIVERSIFY

        The objective of this strategy is to choose samples such that they have a high distance from each other.
        This strategy requires the input to be a NxD matrix of numbers.
        This applies to EMBEDDINGS, but also to SCORES and numerical METADATA. It is specified easily:

        .. code-block:: python

            strategy=DockerWorkerSelectionConfigEntryStrategy(type=DockerWorkerSelectionStrategyType.DIVERSIFY)

        If you want to preserve a minimum distance between chosen samples, you can specify it as a stopping condition:

        .. code-block:: python

            strategy=DockerWorkerSelectionConfigEntryStrategy(type=DockerWorkerSelectionStrategyType.DIVERSIFY, stopping_condition_minimum_distance=0.2)

        Setting the stopping_condition_minimum_distance to 0.2 will remove all samples which are
        closer to each other than 0.2. This allows you to specify the minimum allowed distance between two image
        embeddings in the output dataset. After normalizing the input embeddings
        to unit length, this value should be between 0 and 2.0.
        This is often a convenient method when working with different data sources and trying to combine them in a balanced way.
        If you want to use this stopping condition to stop the selection early, make sure that you allow selecting enough samples by setting `n_samples` high enough.

    .. tab:: WEIGHTS

        The objective of this strategy is to choose samples that have a high numerical value. It requires the input to be a Nx1 matrix of numbers,
        which applies to SCORES and numerical METADATA. It can be specified easily:

        .. code-block:: python

            strategy=DockerWorkerSelectionConfigEntryStrategy(type=DockerWorkerSelectionStrategyType.WEIGHTS)

    .. tab:: THRESHOLD

        The objective of this strategy is to only choose samples that have a numerical value fulfilling a threshold criterion.
        E.g. they should be bigger or smaller than a certain value. Like for weighting, this strategy requires the input to be a Nx1 matrix of numbers,
        which applies to SCORES and numerical METADATA. It is specified as follows:

        .. code-block:: python

            strategy=DockerWorkerSelectionConfigEntryStrategy(type=DockerWorkerSelectionStrategyType.THRESHOLD, threshold=20, operation=DockerWorkerSelectionStrategyThresholdOperation.BIGGER_EQUAL)

        The allowed operations are SMALLER, SMALLER_EQUAL, BIGGER, BIGGER_EQUAL.

    .. tab:: BALANCE

        The objective of this strategy to choose samples such the distribution of classes in them is as close to a target distribution as possible.
        E.g. the samples chosen should have 50% sunny and 10% rainy weather.
        Or the objects of the samples chosen should be 20% ambulance and 40% buses.

        .. code-block:: python

            strategy=DockerWorkerSelectionConfigEntryStrategy(type=DockerWorkerSelectionStrategyType.BALANCE, target={"Ambulance": 0.2, "Bus": 0.4})

        If the values of the target do not sum up to 1, the remainder is assumed to be the target for the other classes.
        In this example with 20% ambulance and 40% bus, there is the implicit assumption, that the remaining 40% should come from any other class,
        e.g. from cars, bicycles or pedestrians.

        The keys of the target must correspond to the class names. The input to this selection strategy can be

        - PREDICTIONS

            In this case, the class names specified in the target must be the same as specified in the predictions.
            See :ref:`worker-selection-predictions` for more details.

        - categorical METADATA

            In this case, the class names specified in the target must be found at least once in the metadata.

Configuration Examples
----------------------

Here are examples for the full configuration including the input for several objectives:

.. tabs::

    .. tab:: Visual Diversity (CORESET)

        Choosing samples that are visually diverse equals diversifying samples based on their embeddings:

        .. code-block:: python

            DockerWorkerSelectionConfigEntry
                input=DockerWorkerSelectionConfigEntryInput(type=DockerWorkerSelectionInputType.EMBEDDINGS),
                strategy=DockerWorkerSelectionConfigEntryStrategy(type=DockerWorkerSelectionStrategyType.DIVERSIFY)
            )

    .. tab:: Active Learning

        Active Learning equals weighting samples based on active learning scores:

        .. code-block:: python

            DockerWorkerSelectionConfigEntry
                input=DockerWorkerSelectionConfigEntryInput(type=DockerWorkerSelectionInputType.SCORES, task="my_object_detection_task", score="uncertainty_entropy"),
                strategy=DockerWorkerSelectionConfigEntryStrategy(type=DockerWorkerSelectionStrategyType.WEIGHTS)
            )

    .. tab:: Visual Diversity and Active Learning (CORAL)

        For combining two strategies, just specify both of them:

        .. code-block:: python

            [
                DockerWorkerSelectionConfigEntry(
                    input=DockerWorkerSelectionConfigEntryInput(type=DockerWorkerSelectionInputType.EMBEDDINGS),
                    strategy=DockerWorkerSelectionConfigEntryStrategy(type=DockerWorkerSelectionStrategyType.DIVERSIFY)
                ),
                DockerWorkerSelectionConfigEntry(
                    input=DockerWorkerSelectionConfigEntryInput(type=DockerWorkerSelectionInputType.SCORES, task="my_object_detection_task", score="uncertainty_entropy"),
                    strategy=DockerWorkerSelectionConfigEntryStrategy(type=DockerWorkerSelectionStrategyType.WEIGHTS)
                )
            ]

    .. tab:: Metadata Thresholding

        This can be used e.g. to remove blurry images, which equals choosing samples whose sharpness is over a threshold:

        .. code-block:: python

            DockerWorkerSelectionConfigEntry(
                input=DockerWorkerSelectionConfigEntryInput(type=DockerWorkerSelectionInputType.METADATA, key="lightly.sharpness"),
                strategy=DockerWorkerSelectionConfigEntryStrategy(type=DockerWorkerSelectionStrategyType.THRESHOLD, threshold=20, operation=DockerWorkerSelectionStrategyThresholdOperation.BIGGER)
            )

    .. tab:: Score Thresholding

        You can specify to only choose images that have at least 2 objects in them:

        .. code-block:: python

            DockerWorkerSelectionConfigEntry(
                input=DockerWorkerSelectionConfigEntryInput(type=DockerWorkerSelectionInputType.SCORES, task="my_object_detection_task", score="object_frequency"),
                strategy=DockerWorkerSelectionConfigEntryStrategy(type=DockerWorkerSelectionStrategyType.THRESHOLD, threshold=2, operation=DockerWorkerSelectionStrategyThresholdOperation.BIGGER_EQUAL)
            )

    .. tab:: Object Balancing

        Use lightly pretagging to get the objects, then specify a target distribution of classes:

        .. code-block:: python

            DockerWorkerSelectionConfigEntry(
                input=DockerWorkerSelectionConfigEntryInput(type=DockerWorkerSelectionInputType.PREDICTIONS, task="lightly_pretagging", name=DockerWorkerSelectionInputPredictionsName.CLASS_DISTRIBUTION)),
                strategy=DockerWorkerSelectionConfigEntryStrategy(type=DockerWorkerSelectionStrategyType.BALANCE, target={"car": 0.1, "bicycle": 0.5, "bus": 0.1, "motorcycle": 0.1, "person": 0.1, "train": 0.05, "truck": 0.05})
            )

    .. tab:: Metadata Balancing

        Let’s assume you have specified metadata with the path `weather.description` and want your selected subset to have 20%  sunny, 40% cloudy and the rest other images:

        .. code-block:: python

            DockerWorkerSelectionConfigEntry(
                input=DockerWorkerSelectionConfigEntryInput( type=DockerWorkerSelectionInputType.METADATA, key="weather.description"),
                strategy=DockerWorkerSelectionConfigEntryStrategy( type=DockerWorkerSelectionStrategyType.BALANCE, target={"sunny": 0.2, "cloudy": 0.4} )
            )

Application of strategies
-------------------------

Generally, the order in which the different strategies were defined in the config does not matter.
In a first steps, all the thresholding strategies are applied.
In the next step, all other strategies are applied in parallel.

The Lightly optimizer tries to fulfil all strategies as good as possible. If it does not, there can be several reasons for it:

- **Tradeoff between different objectives.**
  The optimizer always has to tradeoff between different objectives.
  E.g. it may happen that all samples with high WEIGHTS are close together. If you also specified the objective DIVERSIFY, then only a few of these high-weight samples
  may be chosen. Instead, also other sample that are visually more diverse, but have lower weights, are chosen.

- **Restrictions in the input dataset.**
  This applies especially for BALANCE: E.g. if there are only 10 images of ambulances in the input dataset and a total
  of 1000 images are selected, the output can only have a maximum of 1% ambulances. Thus a BALANCE target of having 20% ambulances cannot be fulfilled.

- **Too little samples to choose*.*
  If the selection algorithm can only choose a small number of samples, it may not be possible to fulfil the objectives. You can solve this by increasing `n_samples`.

Selection on object level
-------------------------

Lightly supports doing selection on object level, see :ref:`docker-object-level`.

While embeddings are fully available, there are some limitations regarding the usage of METADATA and predictions for SCORES and PREDICTIONS as input:

- When using the object level workflow, the object detections used to create the object crops out of the images are available and can be used for both the SCORES and PREDICTIONS input. However, predictions from other tasks are NOT available at the moment.

- Lightly metadata is generated on the fly for the object crops and can thus be used for selection. However, other metadata is on image level and thus NOT available at the moment.

If your use case would profit from using image-level data for object-level selection, please reach out to us.






