# Time evolution of quantum states via adaptive variational quantum algorithm

An Adaptive Variational Quantum Dynamics Simulation (AVQDS) automatically generates a variational ansatz and adaptively expands it along the time evolution path. The ansatz closely matches that from exact diagonalization time evolution and the circuits require less number of gates than Trotter simulations up to the final time given.


## Requirement

Before executing the AVQDS code, one should install h5py, qiskit, qutip, numpy and the latest version of scipy.

## Files in AVQDS_heis

There are two types of files, python and ipython (Jupyter notebook)

### ipython file (Jupyter notebook)

**Mini_Heisenberg_model_best_basis_gate_57.ipynb** is the notebook using the best Heisenberg trotter decomposition using only 3-cnot gates. The best result is 57 %.

**Mini_Heisenberg_model_variational_gate_93__U.ipynb** is the notebook using UAVQDS (π). The best result is 93 %.

**Mini_Heisenberg_model_variational_gate_U4_70.ipynb** is the notebook using UAVQDS (π/4). The best result is 70 %.

### python file

**ansatz.py**
Finds the variational wave-function using a pool of operators that construct the Hamiltonian and corresponding variational parameters that minimize the McLachlan distance.

**avaridyn.py**
Stores records of desired quantities.

**model.py**
The Heisenberg model Hamiltonian with open boundary conditions is defined in this module.

**plot.py**
To plot quantities of interest like the Loschmidt echo.

**run.py**
Runs the AVQDS simulation to find the optimal parameters and operators for the time evovled variational ansatz as the classical component of the hybrid algorithm optimization.

**timing.py**

## Files in Results 

In the results folder, there are 4 files. **params_trace_pi.dat** and **ansatz_pi.h5**
denote the parameters and operator sequence of UAVQDS (π). **params_trace_pi4.dat**
and **ansatz_pi4.h5** denote the parameters and operator sequence of UAVQDS (π/4).



