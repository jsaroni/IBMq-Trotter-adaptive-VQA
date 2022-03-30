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


N=2


start = time.time()


nsite = N


init_state = (1, 1, 1)
ref_state = np.zeros(2**nsite)
if init_state[0] == init_state[1]:
    ref_state[init_state[0]] += 1.
else:
    ref_state[init_state[0]] += 1./np.sqrt(2.)
    ref_state[init_state[1]] += np.sign(init_state[2])/np.sqrt(2.)

ans = ansatz(nsite, ref_state = ref_state, pool='Heis', pthcut=9000)

model = heis(nsite=nsite, T=6, Jzz_init = np.ones(nsite), Jxx_init= np.zeros(nsite), Jyy_init = np.zeros(nsite), hs_init = np.zeros(nsite), Jxx=1.0, Jyy=1.0)
dyn = avaridynHeis(model, ans, quench_type = 1, init_state = (init_state[0], init_state[1], init_state[2]), dtmax=0.001, dthmax=0.01) # dtmax=0.1, dthmax=0.2, dtmax=0.001, dthmax=0.01
dyn.run()

    

print(generate_op_pools(3, 2, None))








