# Time evolution of quantum states via adaptive variational quantum algorithm

An Adaptive Variational Quantum Dynamics Simulation (AVQDS)[1](https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.030307)[2](https://quantum-journal.org/papers/q-2019-10-07-191/)automatically generates a variational ansatz and adaptively expands it along the time evolution path. The ansatz closely matches that from exact diagonalization time evolution and the circuits require less number of gates than Trotter simulations up to the final time given. ([More details](https://github.com/jsaroni/IBMq-Trotter-adpative-VQA/blob/main/IBMq_Trotter_adaptive_VQA.pdf)) 


## Requirement

Before executing the AVQDS code, one should install h5py, qiskit, qutip, numpy and the latest version of scipy.

## Files in AVQDS_heis

There are two types of files, python and ipython (Jupyter notebook)

### ipython file (Jupyter notebook)

[Mini_Heisenberg_model_best_basis_gate_57.ipynb](https://github.com/jsaroni/IBMq-Trotter-adpative-VQA/blob/main/AVQDS_heis/Mini_Heisenberg_model_best_basis_gate_57.ipynb) is the notebook using the best Heisenberg trotter decomposition using only 3-cnot gates for a one trotter step. The best result is 57 %.

[Mini_Heisenberg_model_variational_gate_93__U.ipynb](https://github.com/jsaroni/IBMq-Trotter-adpative-VQA/blob/main/AVQDS_heis/Mini_Heisenberg_model_variational_gate_93__U.ipynb) is the notebook using UAVQDS (π). The best result is 93 %.

[Mini_Heisenberg_model_variational_gate_U4_70.ipynb](https://github.com/jsaroni/IBMq-Trotter-adpative-VQA/blob/main/AVQDS_heis/Mini_Heisenberg_model_variational_gate_U4_70.ipynb) is the notebook using UAVQDS (π/4). The best result is 70 %.

### python file

[ansatz.py](https://github.com/jsaroni/IBMq-Trotter-adpative-VQA/blob/main/AVQDS_heis/ansatz.py)
Finds the variational wave-function using a pool of operators that construct the Hamiltonian and corresponding variational parameters that minimize the McLachlan distance.

[avaridyn.py](https://github.com/jsaroni/IBMq-Trotter-adpative-VQA/blob/main/AVQDS_heis/avaridyn.py)
Stores records of desired quantities.

[model.py](https://github.com/jsaroni/IBMq-Trotter-adpative-VQA/blob/main/AVQDS_heis/model.py)
The Heisenberg model Hamiltonian with open boundary conditions is defined in this module.

[plot.py](https://github.com/jsaroni/IBMq-Trotter-adpative-VQA/blob/main/AVQDS_heis/plot.py)
To plot quantities of interest like the Loschmidt echo.

[run.py](https://github.com/jsaroni/IBMq-Trotter-adpative-VQA/blob/main/AVQDS_heis/run.py)
Used to run the AVQDS simulation to find the optimal parameters and operators for the time evovled variational ansatz. Since the process of searching the optimized parameters requires bunch of circuits, we use the classical simulation help us to calculate McLachlan distance directly.

[timing.py](https://github.com/jsaroni/IBMq-Trotter-adpative-VQA/blob/main/AVQDS_heis/timing.py)
Used to count the execution time of the program.

## Files in Results 

In the results folder, there are 4 files. [params_trace_pi.dat](https://github.com/jsaroni/IBMq-Trotter-adpative-VQA/blob/main/results/params_trace_pi.dat) and [ansatz_pi.h5](https://github.com/jsaroni/IBMq-Trotter-adpative-VQA/blob/main/results/ansatz_pi.h5)
denote the parameters and operator sequence of UAVQDS (π). [params_trace_pi4.dat](https://github.com/jsaroni/IBMq-Trotter-adpative-VQA/blob/main/results/params_tracepi4.dat)
and [ansatz_pi4.h5](https://github.com/jsaroni/IBMq-Trotter-adpative-VQA/blob/main/results/ansatz_pi4.h5) denote the parameters and operator sequence of UAVQDS (π/4).



