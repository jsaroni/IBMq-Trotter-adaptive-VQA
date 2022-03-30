# author: Yongxin Yao (yxphysice@gmail.com)
import h5py, numpy, pickle, os
from qutip import Qobj


class avaridynBase:
    '''
    Base class for adaptive variational quantum dynamics simulation.
    defines commom procedures.
    '''
    def __init__(self,
                 model,  # the model
                 ansatz,  # the ansatz
                 quench_type = 1,  # 0 = quench from GS of initial Hamiltonian, 1 = time-evolve from specified (superposition ) of up to two product states
                 init_state = (0, 0, 1),  # tuple of two product states and relative phase: [i,j, relative phase] with i,j \in {0, ..., 2^(nsite - 1)} and relative phase \theta. If quench_type = 1, allow for an initial state of the form (|i> + exp(i\theta) |j>)/sqrt{2}, where |i>, |j> are arbitrary Z basis product states (given in integer representation of bit string).
                 dtmax=0.05,  # maximal allowed time step size
                 dthmax=0.02,  # maximal allowed theta step size
                 checkpoint = False
                 ):
        # assign various inner-class properties.
        self._model = model
        self._ansatz = ansatz
        self._quench_type = quench_type
        self._init_state = init_state
        self._dtmax = dtmax
        self._dthmax = dthmax
        self._checkpoint = checkpoint

    def run(self,
            path="/0/",  # path to save final results.
            mode=1,  # 1: set ground state of the initial Hamiltonian
            #    as the reference state for the ansatz.
            # others: default in ansatz class.
            tmore=0.01,  # additional time for further simulations
            ):
        # set initial state for dynamics simulations
        self.set_initial_state(mode=mode)

        if self._quench_type == 0:
            # ansatz adaptive optimization for ground state of the initial H
            self._ansatz.additive_optimize(self._model.get_h(0))
        elif self._quench_type == 1:
            pass #: don't do anything as ansatz is already set in function set_initial_state
        else:
            raise Exception("Wrong value of quench type: only 0 and 1 are allowed.")
        # get the current ansatz state
        vec = self._ansatz.get_state()
        #print(f"state =  {vec}")

        # get the H expectation value of the ansatz state.
        e_ansatz = self._model.get_h_expval(0, vec)
        print(f"ansatz energy: {e_ansatz:.5f}")
        if self._ansatz._opdense:
            # if operator is in dense matrix representation.
            ovlp = abs(numpy.vdot(vec, self._vec_i))
        else:
            # Q obj format
            ovlp = abs(vec.overlap(self._vec_i))
        print(f"initial state overlap: {ovlp:.10f}")

        if self.restart():
            t = self._t_list[-1] + self._dt
            self._ansatz.update_state()
        else:
            # initialize the record set
            self.init_records()
            # (noniniform) time mesh
            self._t_list = []
            t = 0.

        # file to record the parameter values
        with open("params_trace.dat", "w") as f:
            nsteps = 0
            print(f"t = {t}")
            print(f"T = {self._model._T}")
            while t < self._model._T + tmore:
                # get the current H
                h = self._model.get_h(t)
                # evolve the state for one step, return the time step
                self._dt = self._ansatz.one_step(h,
                        self._dtmax,
                        self._dthmax,
                        )
                # update the time mesh
                self._t_list.append(t)
                # record the quantitis of interest of the current step
                self.update_records(t)
                # record parameters
                f.write(f"{t:.6f}  " + "  ".join(f"{th:.12f}" \
                        for th in self._ansatz._params) + "\n")
                t += self._dt
                nsteps += 1
                if nsteps == 100:
                    self.snapshot()
                    nsteps = 0

        self.snapshot()
        # some limits of the equation of motion variables to check.
        print(f"pthmaxt = {self._ansatz._pthmaxt:.2e}")
        #print(f"m_max = {numpy.max(numpy.abs(self._ansatz._mmat))}")
        #print(f"v_max = {numpy.max(numpy.abs(self._ansatz._vvec))}")
        # save records
        self.save_records(path)
        # save the final ansatz.
        self._ansatz.save_ansatz()
        # save final state
        self._ansatz.save_state(self._t_list[-1])

    def set_initial_state(self, mode):
        raise NotImplementedError("set_initial_state not implemented.")

    def init_records(self):
        self._records = {}

    def update_records(self):
        raise NotImplementedError("update_records not implemented.")

    def save_records(self, path):
        '''
        save records to the path.
        '''
        with h5py.File("results_asim" + ".h5", "a") as f:
            if path in f:
                del f[path]
            f[f"{path}/t"] = self._t_list
            for key, val in self._records.items():
                f[f"{path}/{key}"] = val
                
                



    def snapshot(self):
        '''
        for continuous run, save the info of the current step.
        '''
        if self._checkpoint:
            with open("snapshot.pickle", "wb") as f:
                pickle.dump([self._dt,
                        self._t_list, self._records, self._ansatz._params,
                        self._ansatz._ansatz, self._ansatz._ngates],
                        f, pickle.HIGHEST_PROTOCOL)

    def restart(self):
        '''
        check restart file.
        '''
        if os.path.isfile("snapshot.pickle"):
            with open("snapshot.pickle", "rb") as f:
                self._dt, self._t_list, self._records, self._ansatz._params, \
                        self._ansatz._ansatz, self._ansatz._ngates \
                        = pickle.load(f)
            return True
        else:
            return False                
                
                
                


class avaridynXYZ(avaridynBase):
    '''
    class for avqds of the XYZ model.
    '''
    def init_records(self):
        '''
        quantities of interest.
        '''
        self._records = {
                "e": [],           # instantaneous energy
                "state": [],       # state vectors
                "ov": [],          # overlap with the initial and final gs.
                "dist": [],        # McLachlan distance
                "spincorr": [],    # spin correlations
                "ngates": [],      # list of numbers of
                                   # (multi-qubit) rotation gates.
                }

    def set_initial_state(self,
            mode=1,   # 1: set ground state of the initial Hamiltonian
                      #    as the referece state for the ansatz.
                      # others: default in ansatz class.
            ):
        # get the lowest energy states of t=0
        #if mode == 1:
        w, v = self._model.get_loweste_states(0)
        # choose the ground state as the intial state
        print(f"lowest energy states: {' '.join(f'{w1:.5f}' for w1 in w[:2])}")
        self._vec_i = v[0]
        if mode == 1:
            # set ground state as the reference state for the ansatz.
            self._ansatz.set_ref_state(self._vec_i)
        #if mode == 2:
            # set arbitrary product state
        # get the final ground state
        w, v = self._model.get_loweste_states(self._model._T)
        print(f"final states energy: {' '.join(f'{w1:.5f}' for w1 in w[:2])}")
        self._vec_f = v[0]

    def update_records(self, t):
        '''
        calculate various quantities for analysis
        '''
        # get the current ansatz state
        vec = self._ansatz.state
        # get the energy
        e = self._model.get_h_expval(t, vec)
        self._records["e"].append(e)
        # get the overlap
        if self._ansatz._opdense:
            ovi = abs(numpy.vdot(vec, self._vec_i))
            ovf = abs(numpy.vdot(vec, self._vec_f))
        else:
            ovi = abs(vec.overlap(self._vec_i))
            ovf = abs(vec.overlap(self._vec_f))
        print(f"t = {t:.5f} e = {e:.6f} ov_initial_GS = {ovi:.6f} ov_final_GS {ovf:.6f}")
        self._records["ov"].append([ovi, ovf])

        # get the spin correlations
        if self._ansatz._opdense:
            vec = Qobj(vec)
        corrs = [op.matrix_element(vec, vec).real  \
                for op in self._model.corr_ops]

        # keep the records
        if isinstance(vec, Qobj):
           vec = vec.full().reshape(-1)
        self._records["state"].append(vec)
        self._records["spincorr"].append(corrs)
        dist = self._ansatz.get_dist()
        self._records["dist"].append(dist)
        ngates = self._ansatz.ngates
        self._records["ngates"].append(ngates)
        # check number of gates on fly
        print(f"ngates: {ngates}")


class avaridynIsing(avaridynBase):
    '''
    avqds class for transverse field Ising model.
    '''
    def init_records(self):
        self._records = {
                "e": [],          # instantaneous energy
                "mag_s": [],      # staggered magnetization
                "sz": [],         # local expectation values of Z_i. Returns array for all sites i.
                "state": [],      # state vectors
                "estate": [],     # exact state vectors
                "ov": [],         # overlaps with the initial
                                  # 2-fold degenerate gs.
                "dist": [],       # McLachlan distance
                "ngates": [],     # list of numbers of
                                  # (multi-qubit) rotation gates.
                "fidelity": [],   # ansatz state fidelity
                }

    def set_initial_state(self, mode=1):
        '''
        set initial state for time evolution. This is only used to compute the fidelity in comparison to the exact time evolution.
        '''
        # get lowest energy states for purpose of checking.
        w, v = self._model.get_loweste_states(0)
        print(f"lowest energy states: {' '.join(f'{w1:.5f}' for w1 in w[:2])}")
        print(f"quench_type = {self._quench_type}")
        if self._quench_type == 0: # initial state is GS of initial H
            # set the initial state |00..0>: the first state in the Hilbert space.
            self._vec_i = numpy.zeros(v[0].shape[0])
            self._vec_i[0] = 1. # define the initial state to be (1,0,0...) (length of vec_i is = 2^nsite). This
            # corresponds to |0000000> (according to Qutip convention).
        elif self._quench_type == 1: # initial state is superposition of two Z basis eigenstates
            self._vec_i = numpy.zeros(v[0].shape[0])
            print(f"init_state[0] = {self._init_state[0]}, init_state[1] = {self._init_state[1]}, init_state[2] = {self._init_state[2]}")
            if self._init_state[0] == self._init_state[1]:
                self._vec_i[self._init_state[0]] = 1.
            else:
                self._vec_i[self._init_state[0]] = 1./numpy.sqrt(2.)
                self._vec_i[self._init_state[1]] = numpy.sign(self._init_state[2])*1./numpy.sqrt(2.)
            print(f"self._vec_i = {self._vec_i}")
            # numpy.exp(1.j * self._init_state[2])
            # if self._init_state[1] is None:
            #     if self._init_state[0] is None:
            #         raise Exception("Did not specify the initial product state for quench.")
            #     else:
            #         self._vec_i[self._init_state[0]] = 1.
            # else:
            #     if self._init_state[0] is None:
            #         self._vec_i[self._init_state[1]] = 1.
            #     else:
            #         self._vec_i[self._init_state[0]] = 1./numpy.sqrt(2)
            #         self._vec_i[self._init_state[1]] = numpy.exp(j*self._init_state[2])/numpy.sqrt(2)
        else:
            raise Exception("quench type different from 0 or 1 is not supported.")

    def update_records(self, t):
        '''
        calcalate various quantities for analysis
        '''
        # get the current ansatz state
        vec = self._ansatz.state
        #print(f"t = {t}")
        #print(f"vec = {vec}")
        # get the energy
        e = self._model.get_h_expval(t, vec)
        self._records["e"].append(e)
        mag_s = self._model.get_staggered_magnetization(vec)
        #print(f"mag_s = {mag_s}")
        self._records["mag_s"].append(mag_s)
        sz= self._model.get_sz(vec)
        self._records["sz"].append(sz)

        if isinstance(vec, Qobj):
            vec = vec.full().reshape(-1)
        self._records["state"].append(vec)
        # overlap with the 2-fold degenerate ground state
        # (first and last states)
#        ov = abs(numpy.asarray([vec[0], vec[-1]]))
        ov = numpy.vdot(vec, self._vec_i)
        #print(f"t = {t:.5f} e = {e:.6f} ov = {ov[0]:.6f} {ov[1]:.6f}\n")
        print(f"t = {t:.5f} e = {e:.6f} ov_initial_state = {ov:.6f}\n")
        self._records["ov"].append(ov)
        # McLachlan distance
        dist = self._ansatz.get_dist()
        self._records["dist"].append(dist)

        ngates = self._ansatz.ngates
        self._records["ngates"].append(ngates)
        print(f"ngates: {ngates}")
        # fidelity
        state = self._ansatz._h.expm(-1j*t).dot(self._vec_i)
        ov2 = abs(state.dot(vec.conj()))**2
        self._records["fidelity"].append(ov2)
        self._records["estate"].append(state)
        print(f"fidelity: {ov2}")











class avaridynHeis(avaridynBase):
    '''
    avqds class for transverse field Ising model.
    '''
    def init_records(self):
        self._records = {
                "e": [],          # instantaneous energy
                "mag_s": [],      # staggered magnetization
                "sz": [],         # local expectation values of Z_i. Returns array for all sites i.
                "state": [],      # state vectors
                "estate": [],     # exact state vectors
                "ov": [],         # overlaps with the initial
                                  # 2-fold degenerate gs.
                "dist": [],       # McLachlan distance
                "ngates": [],     # list of numbers of
                                  # (multi-qubit) rotation gates.
                "fidelity": [],   # ansatz state fidelity
                "params": [],     # variational parameters
                "operators": [],  # operators from operator pool
                
                }

    def set_initial_state(self, mode=1):
        '''
        set initial state for time evolution. This is only used to compute the fidelity in comparison to the exact time evolution.
        '''
        # get lowest energy states for purpose of checking.
        w, v = self._model.get_loweste_states(0)
        print(f"lowest energy states: {' '.join(f'{w1:.5f}' for w1 in w[:2])}")
        print(f"quench_type = {self._quench_type}")
        if self._quench_type == 0: # initial state is GS of initial H
            # set the initial state |00..0>: the first state in the Hilbert space.
            self._vec_i = numpy.zeros(v[0].shape[0])
            self._vec_i[0] = 1. # define the initial state to be (1,0,0...) (length of vec_i is = 2^nsite). This
            # corresponds to |0000000> (according to Qutip convention).
        elif self._quench_type == 1: # initial state is superposition of two Z basis eigenstates
            self._vec_i = numpy.zeros(v[0].shape[0])
            print(f"init_state[0] = {self._init_state[0]}, init_state[1] = {self._init_state[1]}, init_state[2] = {self._init_state[2]}")
            if self._init_state[0] == self._init_state[1]:
                self._vec_i[self._init_state[0]] = 1.
            else:
                self._vec_i[self._init_state[0]] = 1./numpy.sqrt(2.)
                self._vec_i[self._init_state[1]] = numpy.sign(self._init_state[2])*1./numpy.sqrt(2.)
            print(f"self._vec_i = {self._vec_i}")
            # numpy.exp(1.j * self._init_state[2])
            # if self._init_state[1] is None:
            #     if self._init_state[0] is None:
            #         raise Exception("Did not specify the initial product state for quench.")
            #     else:
            #         self._vec_i[self._init_state[0]] = 1.
            # else:
            #     if self._init_state[0] is None:
            #         self._vec_i[self._init_state[1]] = 1.
            #     else:
            #         self._vec_i[self._init_state[0]] = 1./numpy.sqrt(2)
            #         self._vec_i[self._init_state[1]] = numpy.exp(j*self._init_state[2])/numpy.sqrt(2)
        else:
            raise Exception("quench type different from 0 or 1 is not supported.")

    def update_records(self, t):
        '''
        calcalate various quantities for analysis
        '''
        # get the current ansatz state
        vec = self._ansatz.state
        #print(f"t = {t}")
        #print(f"vec = {vec}")
        # get the energy
        e = self._model.get_h_expval(t, vec)
        self._records["e"].append(e)
        mag_s = self._model.get_staggered_magnetization(vec)
        #print(f"mag_s = {mag_s}")
        self._records["mag_s"].append(mag_s)
        sz= self._model.get_sz(vec)
        self._records["sz"].append(sz)

        if isinstance(vec, Qobj):
            vec = vec.full().reshape(-1)
        self._records["state"].append(vec)
        # overlap with the 2-fold degenerate ground state
        # (first and last states)
#        ov = abs(numpy.asarray([vec[0], vec[-1]]))
        ov = numpy.vdot(vec, self._vec_i)
        #print(f"t = {t:.5f} e = {e:.6f} ov = {ov[0]:.6f} {ov[1]:.6f}\n")
        print(f"t = {t:.5f} e = {e:.6f} ov_initial_state = {ov:.6f}\n")
        self._records["ov"].append(ov)
        # McLachlan distance
        dist = self._ansatz.get_dist()
        self._records["dist"].append(dist)

        ngates = self._ansatz.ngates
        self._records["ngates"].append(ngates)
        print(f"ngates: {ngates}")
        # fidelity
        state = self._ansatz._h.expm(-1j*t).dot(self._vec_i)
        ov2 = abs(state.dot(vec.conj()))**2
        self._records["fidelity"].append(ov2)
        self._records["estate"].append(state)
        #theta_mu = self._ansatz.params
        #self._records["params"].append(theta_mu)
        
        
        print(f"fidelity: {ov2}")
















