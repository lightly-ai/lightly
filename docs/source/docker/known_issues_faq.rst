.. _rst-docker-known-issues-faq:

Known Issues and FAQ
===================================


Shared Memory Error when running Lightly Docker
-----------------------------------------------

The following error message appears when the docker runtime has not enough
shared memory. By default Docker uses 64 MBytes. However, when using multiple 
workers for data fetching `lightly.loader.num_workers` there might be not enough.

.. code-block:: console

    ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).                                                                                                
    Traceback (most recent call last):                                                                                                                                                                          
    File "/opt/conda/envs/env/lib/python3.7/multiprocessing/queues.py", line 236, in _feed                                                                                                                    
        obj = _ForkingPickler.dumps(obj)                                                                                                                                                                        
    File "/opt/conda/envs/env/lib/python3.7/multiprocessing/reduction.py", line 51, in dumps                                                                                                                  
        cls(buf, protocol).dump(obj)                                                                                                                                                                            
    File "/opt/conda/envs/env/lib/python3.7/site-packages/torch/multiprocessing/reductions.py", line 321, in reduce_storage                                                                                   
        fd, size = storage._share_fd_()                                                                                                                                                                         
    RuntimeError: unable to write to file </torch_31_1030151126> 

To solve this problem we need to increase the shared memory for the docker runtime.

You can change the shared memory to 512 MBytes by adding `--shm-size="512m"` to 
the docker run command:

.. code-block:: console

    # example of docker run with setting shared memory to 512 MBytes
    docker run --shm-size="512m" --gpus all