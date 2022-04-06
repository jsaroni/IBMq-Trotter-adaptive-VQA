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
from qiskit.opflow import Zero, One


N=3


start = time.time()


nsite = N

# init_state = ([state for supersposition],[phase]) : phase e^(i\phi)
init_state_1 = ([3],[1])
ref_state = np.zeros(2**nsite)

if len(init_state_1[0])==1:
    
    ref_state[init_state_1[0][0]] = 1
    
else:
    
    for i in range(len(init_state_1[0])):
        
        ref_state[init_state_1[0][i]]=init_state_1[1][i]/np.sqrt(len(init_state_1[0]))
        

ans = ansatz(nsite, ref_state = ref_state, pool='Heis', pthcut=9000)

model = heis(nsite=nsite, T=np.pi, Jzz_init = np.ones(nsite), Jxx_init= np.zeros(nsite), Jyy_init = np.zeros(nsite), hs_init = np.zeros(nsite), Jxx=1.0, Jyy=1.0)
dyn = avaridynHeis(model, ans, quench_type = 1, init_state = init_state_1, dtmax=0.001, dthmax=0.01) # dtmax=0.1, dthmax=0.2, dtmax=0.001, dthmax=0.01
dyn.run()

  
target_t=np.pi
f = open("params_trace.DAT", "r")
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
    num_string = f.readline()

print(pi_closer)
#print qiskit circuit

target_t=np.pi
f1= open("params_trace.DAT", "r")
num_string = f1.readline()
num=num_string.split()
while float(num[0]) != pi_closer:
    num_string = f1.readline()
    num=num_string.split()
    
params=num[1:]

print(params)

qc = QuantumCircuit(nsite)
qc.x([1,2])
#for _ in range(4):
for i,op in enumerate(ans._ansatz[1]):
    if op[1]==3:
        qc.rzz(float(params[i]),op[0],op[2])
    elif op[1]==2:
        qc.ryy(float(params[i]),op[0],op[2])
    elif op[1]==1:
        qc.rxx(float(params[i]),op[0],op[2])
        



from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.providers.aer import QasmSimulator
from qiskit.quantum_info import state_fidelity

st_qcs = state_tomography_circuits(qc,[0,1,2])
backend = QasmSimulator(method='statevector') #jakarta

reps=4
jobs = []
shots=8192
for _ in range(reps):
    # execute
    job = execute(st_qcs, backend, shots=shots)
    jobs.append(job)
    

# Compute the state tomography based on the st_qcs quantum circuits and the results from those ciricuits
def state_tomo(result, st_qcs): #, time):
    # The expected final state; necessary to determine state tomography fidelity
    target_state = (One^One^Zero).to_matrix()  # DO NOT MODIFY (|q_5,q_3,q_1> = |110>)
    # Fit state tomography results

    #target_state = (U_heis3(float(time)) @ initial_state).eval().to_matrix()

    

    tomo_fitter = StateTomographyFitter(result, st_qcs)

    rho_fit = tomo_fitter.fit(method='lstsq')
    # Compute fidelity
    fid = state_fidelity(rho_fit, target_state)
    return fid

# Compute tomography fidelities for each repetition

fids = []
#target_time=trotter_steps * np.pi/10
for job in jobs:
    fid = state_tomo(job.result(), st_qcs)#, target_time)
    fids.append(fid)

print('state tomography fidelity = {:.4f} \u00B1 {:.4f}'.format(np.mean(fids), np.std(fids)))