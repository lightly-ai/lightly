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
   ``IJEPAMaskCollator`` used to live in ``lightly.data.collate``, which has been
   removed. Use ``from lightly.data import IJEPAMaskCollator``.

.. autoclass:: lightly.data.ijepa_collate.IJEPAMaskCollator
   :members:
   :special-members: __call__
