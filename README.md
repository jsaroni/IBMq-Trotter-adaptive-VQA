# Time evolution of quantum states via adaptive variational quantum algorithm

An Adaptive Variational Quantum Dynamics Simulation (AVQDS) automatically generates a variational ansatz and adaptively expands it along the time evolution path. The ansatz closely matches that from exact diagonalization time evolution and the circuits require less number of gates than Trotter simulations up to the final time given.


## Requirement

After executong the code, one should preinstall h5py, qiskit, qutip.

## Files in AVQDS_heis

There are two type of files, python and ipython (Jupyter notebook)

### ipython (Jupyter notebook)

**Mini_Heisenberg_model_best_basis_gate_57.ipynb** is the notebook use the best Heisenberg trotter decomposition using only 3-cnot gates. The best result is 57 %.

**Mini_Heisenberg_model_variational_gate_93__U.ipynb** is the notebook use UAVQDS (π). The best result is 93 %.

**Mini_Heisenberg_model_variational_gate_U4_70.ipynb** is the notebook use UAVQDS (π/4). The best result is 70 %.

### python

**ansatz.py**

**avaridyn.py**

**model.py**

**plot.py**

**run.py**

**timing.py**

## Files in Results 

In the results folder, there are 4 files. **params_trace_pi.dat** and **ansatz_pi.h5**
denote the parameters and operator sequence of UAVQDS (π). **params_trace_pi4.dat**
and **ansatz_pi4.h5** denote the parameters and operator sequence of UAVQDS (π/4).



