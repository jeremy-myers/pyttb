``tenmat``
----------------

Data members
^^^^^^^^^^^^
+-----------------+----------------------+------------------------------------------------------------------------+
| MATLAB name     | ``pyttb`` name       | Calling convention                                                     |
+=================+======================+========================================================================+
| ``tsize``       | ``tshape``           | ``X.tshape``                                                           |
+-----------------+----------------------+------------------------------------------------------------------------+

Methods
^^^^^^^
+-----------------+----------------------+------------------------------------------------------------------------+
| MATLAB name     | ``pyttb`` name       | Calling convention                                                     |
+=================+======================+========================================================================+
|                 | ``from_data``        | ``B = ttb.tenmat.from_data(A.data, A.rindices, A.cindices, A.tshape)`` |
| ``tenmat``      +----------------------+------------------------------------------------------------------------+
|                 | ``from_tensor_type`` | ``A = ttb.tenmat.from_tensor_type(X, np.array([1]))``                  |
+-----------------+----------------------+------------------------------------------------------------------------+
| ``tensor``      | ``to_tensor``        | ``X.to_tensor()``                                                      |
+-----------------+----------------------+------------------------------------------------------------------------+