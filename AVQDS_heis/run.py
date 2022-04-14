import numpy as np
import os
import sys
import time
from model import zzxzmodel
from model import heis
from ansatz import ansatz
from avaridyn import avaridynIsing
from avaridyn import avaridynHeis
from ansatz import generate_op_pools
from qiskit import QuantumCircuit, QuantumRegister, execute
from qiskit.opflow import Zero, One, I, X, Y, Z


N=3


start = time.time()


nsite = N

# init_state = ([state for supersposition],[phase]) : phase e^(i\phi)
init_state_1 = ([3],[1]) #Create initial state 0-> [down,down,down], 1-> [up,down,down].....

# U_{\pi/4} the init_state_1 should be ([0,4,6,7],[1,1,1,1])

# Make initial state at ref_state at t=0
ref_state = np.zeros((2**nsite),dtype=complex)

if len(init_state_1[0])==1:
    
    ref_state[init_state_1[0][0]] = 1
    
else:
    
    for i in range(len(init_state_1[0])):
        
        ref_state[init_state_1[0][i]]=init_state_1[1][i]/np.sqrt(len(init_state_1[0]))
        

ans = ansatz(nsite, ref_state = ref_state, pool='Heis', pthcut=9000) # Create Heisenberg operator pool: Rzz, Rxx, Ryy

# builds up the Hamiltonian model (see model.py)
model = heis(nsite=nsite, T=np.pi, Jzz_init = np.ones(nsite), Jxx_init= np.zeros(nsite), Jyy_init = np.zeros(nsite), hs_init = np.zeros(nsite), Jxx=1.0, Jyy=1.0, Jzz=1.0)
# T: the total evolution time. If one wan to get U(\pi/4), one can set it as np.pi/4


# The set up for time evolution with Hamiltonian
dyn = avaridynHeis(model, ans, quench_type = 1, init_state = init_state_1, dtmax=0.001, dthmax=0.01) 

dyn.run() # Run avqds simulator to find best parameters and operators for pseudo-Trotter time evolution.
  
target_t=np.pi # the evolution time we want to get. If one wan to get U(\pi/4), one can set it as np.pi/4
f = open("params_trace.DAT", "r") # open the variational parameters file
num_string = f.readline()
pi_closer = 0
del_num = 1000
while num_string != '': 
    num = num_string.split()
    time = float(num[0])

    if min(del_num,np.abs(time-target_t))==del_num:
        pass
    else:
        del_num=np.abs(time-target_t)
        pi_closer=time
        params=num[1:]
    num_string = f.readline()

print(params) # The variational parameters

qc = QuantumCircuit(nsite) # create quantum circuit
qc.x([1,2])
for _ in range(1): # if one want to repeat 4 times, set it as 4
    for i,op in enumerate(ans._ansatz[1]):
        if op[1]==3:                             # op[1]=3: means Rzz gate
            qc.rzz(float(params[i]),op[0],op[2])
        elif op[1]==2:                           # op[1]=2: means Ryy gate
            qc.ryy(float(params[i]),op[0],op[2])
        elif op[1]==1:                           # # op[1]=1: means Rxx gate
            qc.rxx(float(params[i]),op[0],op[2])
    
        

###################### State tomography ############################

from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import state_fidelity



st_qcs = state_tomography_circuits(qc,[0,1,2]) # create circuits for state tomography
backend = QasmSimulator(method='statevector') # Use the ideal simulator

reps=4
jobs = []
shots=8192
for _ in range(reps):
    # execute
    job = execute(st_qcs, backend, shots=shots)
    jobs.append(job)

# Returns the matrix representation of the XXX Heisenberg model for 3 spin-1/2 particles in a line
def H_heis3():
    # Interactions (I is the identity matrix; X, Y, and Z are Pauli matricies; ^ is a tensor product)
    XXs = (I^X^X) + (X^X^I)
    YYs = (I^Y^Y) + (Y^Y^I)
    ZZs = (I^Z^Z) + (Z^Z^I)
    
    # Sum interactions
    H = XXs + YYs + ZZs
    
    # Return Hamiltonian
    return H

# Returns the matrix representation of U_heis3(t) for a given time t assuming an XXX Heisenberg Hamiltonian for 3 spins-1/2 particles in a line
def U_heis3(t):
    # Compute XXX Hamiltonian for 3 spins in a line
    H = H_heis3()
    
    # Return the exponential of -i multipled by time t multipled by the 3 spin XXX Heisenberg Hamilonian 
    return (t * H).exp_i()    

# Compute the state tomography based on the st_qcs quantum circuits and the results from those ciricuits
def state_tomo(result, st_qcs): #, time):
    # The expected final state; necessary to determine state tomography fidelity
    #target_state = (One^One^Zero).to_matrix()
    # Fit state tomography results Zero
    
    initial_state = One^One^Zero
    
    time=4*np.pi/4

    target_state = (U_heis3(float(time)) @ initial_state).eval().to_matrix()


    tomo_fitter = StateTomographyFitter(result, st_qcs)

    rho_fit = tomo_fitter.fit(method='lstsq')
    # Compute fidelity
    fid = state_fidelity(rho_fit, target_state)
    return fid

# Compute state tomography fidelities for each repetition
fids = []
for job in jobs:
    fid = state_tomo(job.result(), st_qcs)#, target_time)
    fids.append(fid)

for i in range(len(params)):
    print('The '+str(i)+'th variational parameter are:', round(float(params[i]),4))
    
    if ans._ansatz[1][i][1] == 1:
    
        print('The corresponding Operator is XX('+str(ans._ansatz[1][i][0])+','+str(ans._ansatz[1][i][2])+')')
    
    elif ans._ansatz[1][i][1] == 2:
        
        print('The corresponding Operator is YY('+str(ans._ansatz[1][i][0])+','+str(ans._ansatz[1][i][2])+')')
        
    elif ans._ansatz[1][i][1] == 3:
        
        print('The corresponding Operator is ZZ('+str(ans._ansatz[1][i][0])+','+str(ans._ansatz[1][i][2])+')')

print('state tomography fidelity = {:.4f} \u00B1 {:.4f}'.format(np.mean(fids), np.std(fids)))