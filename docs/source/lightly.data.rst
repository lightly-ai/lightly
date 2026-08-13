lightly.data
===================

.. automodule:: lightly.data

.dataset
---------------
.. automodule:: lightly.data.dataset
   :members:

.multi_view_collate
-------------------
.. autoclass:: lightly.data.multi_view_collate.MultiViewCollate
   :members:
   :special-members: __call__

.ijepa_collate
--------------

.. note::
   ``IJEPAMaskCollator`` used to live in ``lightly.data.collate``. That import path
   still works but warns, and is removed in v1.7.0. Use
   ``from lightly.data import IJEPAMaskCollator``.

.. autoclass:: lightly.data.ijepa_collate.IJEPAMaskCollator
   :members:
   :special-members: __call__

.collate:
---------
.. automodule:: lightly.data.collate
   :members:
