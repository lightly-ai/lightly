.. _lightly-active-learning:

Active Learning
===================
Lightly enables active learning with only a few lines of additional code. Learn 
here, how to get the most out of your data by maximizing the available information
in your annotated dataset.


Introcution to Pool-based Active Learning
------------------------------------------
TODO


Active Learning with Lightly
-----------------------------

Preparations
~~~~~~~~~~~~~~~~~
Before you read on, make sure you have read the section on the :ref:`lightly-platform`. 
In particular, you should know how to create a dataset in the `web-app <https://app.lightly.ai>`_.
and how to upload images and embeddings to it. To do active learning, you will 
need such a dataset with embeddings (don't worry, it's free!).


Concepts
~~~~~~~~~~~~~~~~~
Lightly makes use of the following concepts for active learning:

* **ApiWorkflowClient:** :py:class:`lightly.api.api_workflow_client.ApiWorkflowClient`
   The `ApiWorkflowClient` is used to connect to our API. The API handles the 
   selection of the images based on embeddings and uncertainty scores. To initialize
   the `ApiWorkflowClient`, you will need the `datasetId` and the `token` from the 
   :ref:`lightly-platform`.
   
* **ActiveLearningAgent:** :py:class:`lightly.active_learning.agents.agent.ActiveLearningAgent`
   The `ActiveLearningAgent` builds the client interface of our active learning 
   framework. It allows to indicate which images are preselected and which ones
   to sample from. Furthermore, one can query it to get a new batch of images.
   To initialize an `ActiveLearningAgent` you need an `ApiWorkflowClient`.
   
* **SamplerConfig:** :py:class:`lightly.active_learning.config.sampler_config.SamplerConfig`
   The `SamplerConfig` allows the configuration of a sampling request. In particular,
   you can set number of samples, the name of the resulting selection, and the `SamplingMethod`.
   Currently, you can set the `SamplingMethod` to one of the following:

   * Random: Selects samples uniformly at random.
   * Coreset: Greedily selects samples which are diverse.
   * Coral: Combines Coreset with uncertainty scores to do active learning.
   
* **Scorer:** :py:class:`lightly.active_learning.scorers.scorer.Scorer`
   The `Scorer` takes as input the predictions of a pre-trained model on the set
   of unlabeled images. It evaluates different scores based on how certain the model
   is about the images and passes them to the API so the sampler can use them with
   Coral.
   

Continue reading to see how these components interact and how active learning is
done with Lightly.


Initial Selection
~~~~~~~~~~~~~~~~~~~
The goal of making an initial selection is to get a subdataset on which you can train
an initial model. The output of the model can then be used to select new samples. That way,
the model can be iteratively improved.

To make an initial selection, start off by adding your raw, *unlabeled* data and the according
image embeddings to a dataset in the Lightly `web-app <https://app.lightly.ai>`_. A simple way to do so
is to use `lightly-magic` from the command-line.

.. code-block:: bash

   # use trainer.max_epochs=0 to skip training
   lightly-magic input_dir='path/to/raw/dataset' dataset_id='xyz' token='123' trainer.max_epochs=0

Next, you will need to initialize the `ApiWorkflowClient` and the `ActiveLearningAgent`

.. code-block:: Python

    import lightly
    from lightly.api import ApiWorkflowClient
    from lightly.active_learning import ActiveLearningAgent

    api_client = ApiWorkflowClient(dataset_id='xyz', token='123')
    al_agent = ActiveLearningAgent(api_client) 


.. note::

   Note about query_tag. TODO.

Let's configure the sampling request and request an initial selection next:

.. code-block:: Python

   from lightly.active_learning import SamplerConfig
   from lightly.openapi_generated.swagger_client import SamplingMethod

   # we want an initial pool of 100 images
   config = SamplerConfig(n_samples=100, method=SamplingMethod.CORESET, name='initial-selection')
   initial_selection = al_agent.query(sampler_config)

   assert len(initial_selection) == 100

The query returns the list of filenames corresponding to the initial selection. Additionally, you
will find that a tag has been created in the web-app under the name "initial-selection".
Head there to scroll through the samples and download the selected images before annotating them.


Active Learning Step
~~~~~~~~~~~~~~~~~~~~~~

After you have annotated your initial selection of images, you can train a model
on them. The trained model can then be used to figure out, with which images it 
has problems. These images can then be added to the labeled dataset.

To do active learning with Lightly, you will need the `ApiWorkflowClient` and `ActiveLearningAgent` from before.
If you have to re-initialize them, make sure to set the `pre_selected_tag_name` to your
current selection (if this is the first iteration, this is the name you have passed 
to the sampler config when doing the initial selection). Note, that if you don't 
have to re-initialize them, the tracking of the tags is taken care of for you.

.. code-block:: Python

   # re-initializing the ApiWorkflowClient and ActiveLearningAgent
    api_client = ApiWorkflowClient(dataset_id='xyz', token='123')
    al_agent = ActiveLearningAgent(api_client, preselected_tag_name='initial-selection')

TODO text here about getting the scorer, normalize!

.. code-block:: Python

    # TODO getting the scorer

TODO text here about starting the next query 

.. code-block:: Python

   # we want a total of 200 images after the first iteration
   config = SamplerConfig(n_samples=200, method=SamplingMethod.CORAL, name='al-iteration-1')
   al_iteration_1 = al_agent.query(sampler_config, scorer)

   assert len(al_iteration_1) == 200


TODO text here
required:
- ApiWorkflowClient from before
- AL agent (from before, otherwise need to set preselected tag)
- SamplerConfig (n_smaples, name, CORAL)
- Scorer (predictions -> normalized)

-> show code