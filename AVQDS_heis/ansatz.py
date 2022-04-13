# author: Yongxin Yao (yxphysice@gmail.com)
import itertools, numpy, h5py, os, multiprocessing, warnings
import scipy.optimize, scipy.linalg, scipy.sparse
from timing import timeit




class spectra_data:
    '''
    data type for more efficient expm calculation.
    store the dense matrix and spectral representation
    '''
    def __init__(self,
            qobj,   # qobj from qutip
            ):
        # save full dense matrix
        self._a = qobj.full()
        # save the spectral representation
        self._w, self._v = numpy.linalg.eigh(self._a)

    def expm(self, alpha):
        # calculate exp(alpha*a)
        return (self._v*numpy.exp(alpha*self._w)).dot(self._v.T.conj())

    def dot(self, b):
        '''
        dot product.
        '''
        #print(f"a={self._a}, b={b}")
        return self._a.dot(b)

    def matrix_element(self, v, vp):
        return v.conj().dot(self._a.dot(vp))


class ansatz:
    '''
    define the main procedures of one step of avqds calculation.
    set up: 1) default reference state of the ansatz.
            2) operator pool
    '''
    def __init__(self,
            nq,              # number of qubits
            ref_state=None,  # reference state of the ansatz
            order=2,         # operator pool order (pauli string length)
            rcut=1.e-3,      # McLachlan distance cut-off
            pthcut=7000,     # maximal allowed \frac{\par theta}{\par t}
            pool=None,       # pool generation option
            opdense=True,    # operator in dense matrix format
            ncpus=1,         # multiprocess, not recommended for inefficiency
            err_ratio=0.,    # relative error percentage to check noise effect
            delta=0,     # Tikhonov regularization parameter
            max_add=20,      # maximal number of new pauli rotation gates to be added
            ):
        # variational parameters.
        self._params = []
        # initial parameter values.
        self._params_init = self._params[:]
        # list of unitaries interms of operators and index labels
        self._ansatz = [[], []]
        self._pool = pool
        self.set_order(order)
        # initialize the number of gates array.
        self._ngates = [0]*self._order
        self._opdense = opdense
        self._nq = nq
        # set reference state
        self.set_ref_state(ref_state)
        self._state = self._ref_state
        self._rcut = rcut
        self._pthcut = pthcut
        # diagonistic quantity to monitor.
        self._pthmaxt = 0  # max \frac{\par theta}{\par t}
        self._ncpus = ncpus
        self._delta = delta
        self._max_add = max_add

        self._err_ratio = err_ratio
        # if nonzero error ratio, initialize the random number generator.
        if abs(err_ratio) > 1.e-12:
            numpy.random.seed()

        # generate operator
        self.generate_op_pools()
        # option to initialize ansatz from file.
        self.h5init()

    def set_order(self, order):
        '''
        set operator order.
        '''
        if self._pool is None:
            self._order = order
        elif self._pool in ["XYZ", "-XYZ", "Ising", "-Ising", "ZZXZ", "-ZZXZ", "Heis", "-Heis"]:
            self._order = 2
        else:
            raise ValueError(f"pool = {self._pool} not available.")

    def h5init(self):
        '''
        initialize ansatz from file.
        '''
        if os.path.isfile("ansatzi.h5"):
            with h5py.File("ansatzi.h5", "r") as f:
                self._ref_state = f["/ref_state"][()]
                if "/ansatz" in f:
                    self._ansatz[0] = f["/ansatz"][()]
                    self._ansatz[1] = f["/ansatz_code"][()]
                    self._params = f["/params"][()]

    def add_error(self, v):
        if abs(self._err_ratio) > 1.e-12:
            # finite error ratio, draw random number from normal distribution
            # assume real entries.
            dv = abs(v)*self._err_ratio  # standard deviations
            dv = numpy.asarray(dv)
            err = dv*numpy.random.randn(*dv.shape)
            return v + err
        else:
            return v

    @timeit
    def additive_optimize(self, h):
        # change the representation of H if necessary.
        if self._opdense:
            self._h = spectra_data(h)
        else:
            self._h = h
        # (adaptively) optimize the ansatz to represent the ground state
        # of the initial Hamiltonian.
        self.optimize_params()
        print(f"initial cost = {self._cost:.6f}")
        iloop = 0
        while True:
            # one iteration of qubit-adapt-vqe.
            added = self.add_op()
            if not added:
                # reaching convergence.
                break
            self.optimize_params()
            self.update_state()
            print(f"iter {iloop}: cost = {self._cost:.6f}")
            iloop += 1
        if len(self._params_init) == 0:
            self._params_init = self._params[:]
        else:
            for i in range(len(self._params) - len(self._params_init)):
                self._params_init.append(0.)

    @property
    def state(self):
        return self._state
    
    @property
    def params(self):
        return self._params

    @property
    def ngates(self):
        return self._ngates[:]

    def get_cost(self):
        '''
        get the value of the cost function (Hamiltonian expectation value).
        '''
        # have the state vector updated.
        self.update_state()
        # expectation value of the Hamiltonian
        res = self._h.matrix_element(self._state, self._state)
        return res.real

    #@timeit
    def update_state(self):
        self._state = self.get_state()

    def generate_op_pools(self):
        '''
        generate operator pool.
        '''
        self._op_pools = generate_op_pools(self._nq,
                self._order,
                self._pool,
                )
        if self._opdense:
            # convert to more efficient data structure if needed.
            for i, op_pool in enumerate(self._op_pools):
                self._op_pools[i][0] = [spectra_data(op) for op in op_pool[0]]

    def set_ref_state(self, ref_state):
        if ref_state is None:
            '''
            default (first) reference state.
            '''
            self._ref_state = numpy.zeros((2**self._nq), dtype=numpy.complex)
            self._ref_state[0] = 1.
        else:
            if not isinstance(ref_state, numpy.ndarray) and self._opdense:
                ref_state = ref_state.full().reshape(-1)
            self._ref_state = ref_state

    #@timeit
    def optimize_params(self):
        if len(self._params) > 0:
            # full reoptimization of the ansatz given the initial point.
            res = scipy.optimize.minimize(fun_cost,
                    self._params,               # starting parameter point
                    args=(self._ansatz[0],      # only need the ansatz
                                                # operators, not the indices.
                            self._ref_state,    # reference state
                            self._h,            # Hamiltonian
                            ),
                    method='CG',                # conjugate gradient
                    jac=fun_jac,                # analytical jacobian function
                    )
            if res.success:
                # save parameters and cost.
                self._params = res.x.tolist()
                self._cost = res.fun
            
                
            else:
                print(res.message)
        else:
            # no parameters to optimize, directly evaluate.
            self._cost = self.get_cost()
            

    #@timeit
    def add_op(self,
            tol=1.e-4,
            ):
        '''
        adding a initary  in the adapt-vqe.
        '''
        scores = self.get_pool_scores()
        ids = numpy.argsort(abs(scores))
        iadd = ids[-1]
        print("top 3 scores: "+ \
                f"{' '.join(f'{scores[i]:.2e}' for i in ids[-3:])}")
        if len(self._ansatz[1]) > 0  \
                and numpy.allclose(self._op_pools[0][1][iadd],  \
                self._ansatz[1][-1]):
            # no further improvement
            print(f"abort: pauli ops {self._ansatz[1][-1]}" + \
                    f" vs {self._op_pools[0][1][iadd]}")
            return False
        elif abs(scores[iadd]) < tol:
            print(f"converge: gradient = {scores[iadd]:.2e} too small.")
            return False
        else:
            self._ansatz[0].append(self._op_pools[0][0][iadd])
            # label
            self._ansatz[1].append(self._op_pools[0][1][iadd])
            self.update_ngates()
            print(f"op {self._ansatz[1][-1]} appended.")
            self._params.append(0.)
            return True

    def update_ngates(self):
        '''
        update gate counts.
        '''
        iorder = numpy.count_nonzero(self._ansatz[1][-1][1::2])
        self._ngates[iorder-1] += 1

    #@timeit
    def add_ops_dyn(self):
        '''
        ansatz adaptively expanding procedure in avqds.
        '''
        # H |vec>
        if self._opdense:
            hvec = self._h.dot(self._state)
        else:
            hvec = self._h*self._state
        icyc = 0
        # energy variance
        hvar = self._e2 - self._e**2
        for _ in range(self._max_add):
            # number of parameters
            np = len(self._params)
            print("the number of parameters is = ", np )
            # M matrix
            mmat = numpy.zeros((np+1, np+1))
            # block for the existing parameters stay the same.
            mmat[:np, :np] = self._mmat
            # V vector
            vvec = numpy.zeros((np+1))
            vvec[:np] = self._vvec
            # McLachlan distance without energy variance
            val_max = self._distp
            # M^-1 V max elements, limits the step size of thetas.
            pth_max = max(self._pthmax, self._pthcut)
            ichoice = None
            mmat_ch = None
            # <partial_vec|vec_cur>
            if self._opdense:
                vpv_list = [numpy.vdot(v, self._state)  \
                        for v in self._vecp_list]
            else:
                vpv_list = [v.overlap(self._state) for v in self._vecp_list]
            for i, op, label in zip(itertools.count(),
                    self._op_pools[-1][0],
                    self._op_pools[-1][1]):
                # same op is the end one, skip
                if len(self._ansatz[1]) > 0 and  \
                        numpy.allclose(label, self._ansatz[1][-1]):
                    continue
                # mmat addition
                if self._opdense:
                    vecp_add = -0.5j*op.dot(self._state)
                else:
                    vecp_add = -0.5j*op*self._state
                if self._opdense:
                    vpv_add = numpy.vdot(vecp_add, self._state)
                    acol = [numpy.vdot(v, vecp_add) + vpv*vpv_add  \
                            for v, vpv in zip(self._vecp_list, vpv_list)]
                    acol.append(numpy.vdot(vecp_add, vecp_add) +  \
                            vpv_add*vpv_add)
                    # vvec addition
                    zes = numpy.vdot(vecp_add, hvec)

                else:
                    vpv_add = vecp_add.overlap(self._state)
                    acol = [v.overlap(vecp_add) + vpv*vpv_add  \
                            for v, vpv in zip(self._vecp_list, vpv_list)]
                    acol.append(vecp_add.overlap(vecp_add) +  \
                            vpv_add*vpv_add)
                    # vvec addition
                    zes = vecp_add.overlap(hvec)

                mmat[:, -1] = numpy.asarray(acol).real
                # add noise
                mmat[:, -1] = self.add_error(mmat[:, -1])
                # upper symmetric part
                mmat[-1, :] =mmat[:, -1]

                vvec[-1] = zes.imag
                vvec[-1] -= (vpv_add*self._e).imag
                # add noise
                vvec[-1] = self.add_error(vvec[-1])

                # m_inv = numpy.linalg.pinv(mmat)
                m_inv = get_minv(mmat, delta=self._delta)
                dist_p = (vvec.conj().dot(m_inv).dot(vvec)).real
                pthvec = m_inv.dot(vvec)
                pthmax = numpy.max(numpy.abs(pthvec))

                if dist_p > val_max + 1e-8 or \
                        (abs(hvar - dist_p) < self._rcut and  \
                        pthmax < pth_max):
                    # dist drop, larger time step, better condition
                    ichoice = i
                    val_max = dist_p
                    pth_max = pthmax
                    mmat_ch = mmat.copy()
                    vvec_ch = vvec.copy()
                    minv_ch = m_inv.copy()
                    vecp_add_ch = vecp_add.copy()

            dist = hvar - val_max
            diff = val_max - self._distp
            if ichoice is None or  \
                    (diff < 1.e-8 and pth_max - self._pthmax < 1.e-6):
                warnings.warn("dynamic ansatz cannot further improve.")
                break

            self._distp = val_max
            self._pthmax = pth_max
            self._mmat = mmat_ch
            self._minv = minv_ch
            self._vvec = vvec_ch
            self._ansatz[0].append(self._op_pools[-1][0][ichoice])
            self._ansatz[1].append(self._op_pools[-1][1][ichoice])
            self.update_ngates()
            self._vecp_list.append(vecp_add_ch)
            self._params.append(0)
            self._params_init.append(0.)

            print(f"pth_max: {pth_max:.2f}")
            print(f"add op: {self._ansatz[1][-1]}")
            print(f"icyc = {icyc}, dist = {dist:.2e}, improving {diff:.2e}")

            if (dist < self._rcut and pth_max < self._pthcut) \
                    or dist < 0:  # noisy case
                break
            icyc += 1


    def get_state(self):
        return get_ansatz_state(self._params,
                        self._ansatz[0],
                        self._ref_state,
                        )

    def get_pool_scores(self):
        '''-0.5j <[h,op]> = im(<h op>)
        '''
        scores = []
        if self._opdense:
            h_vec = self._h.dot(self._state)
        else:
            h_vec = self._h*self._state
        for op in self._op_pools[0][0]:
            if self._opdense:
                ov = op.dot(self._state)
                zes = numpy.vdot(h_vec, ov)
            else:
                ov = op*self._state
                zes = h_vec.overlap(ov)
            scores.append(zes.imag)
        return numpy.array(scores)

    #@timeit
    def one_step(self, h, dtmax, dthmax, dt=0.01):
        # hamiltonian
        if self._opdense:
            self._h = spectra_data(h)
        else:
            self._h = h
        self.set_par_states()
        amat = self.get_amat()
        cvec, mp, vp, e, e2 = self.get_cvec_phase()
        # McLachlan's principle, including global phase contribution
        m = amat.real + numpy.asarray(mp).real
        v = cvec.imag - numpy.asarray(vp).imag
        # add moise
        m = self.add_error(m)
        v = self.add_error(v)

        # check global phase contribution
        if len(v) > 0:
            res = numpy.max(numpy.abs(numpy.asarray(mp).real))
            print(f"global phase contribution for m: {res}")
            res = numpy.max(numpy.abs(numpy.asarray(vp).imag))
            print(f"global phase contribution for v: {res}")
        self._mmat = m
        self._vvec = v
        self._e, self._e2 = e, e2
        # m_inv = numpy.linalg.pinv(m)
        m_inv = get_minv(m, delta=self._delta)
        # McLachlan distance
        dist_h2 = e2 - e**2
        dist_p = (v.conj().dot(m_inv).dot(v)).real
        dist = dist_h2 - dist_p
        self._distp = dist_p
        self._minv = m_inv
        p_params = m_inv.dot(v)
        if len(p_params) > 0:
            pthmax = numpy.max(numpy.abs(p_params))
        else:
            pthmax = 0
        self._pthmax = pthmax
        print(f"initial mcLachlan distance: {dist:.2e} pthmax: {pthmax:.2f}")
        print("params =", self.params )
        np = len(self._params)
        
        print("length = ", np)
        
        if dist > self._rcut or pthmax > self._pthcut:
            self.add_ops_dyn()
        p_params = self._minv.dot(self._vvec)
        if len(p_params) > 0:
            pthmax = numpy.max(numpy.abs(p_params))
            self._pthmaxt = max(pthmax, self._pthmaxt)
            print(f"max element in p_params: {pthmax:.2f}")
            if pthmax > 0:
                dt = min(dtmax, dthmax/pthmax)

        self._params = [p + pp*dt for p, pp in zip(self._params, p_params)]
        self.update_state()
        return dt

    def condition_check(self, msg, tol=1.e-6):
        if minv_error(self._mmat, self._minv) > tol:
            raise ValueError(f"{msg} M matrix inversion error.")

    def get_dist(self):
        return self._e2 - self._e**2 - self._distp

    #@timeit
    def set_par_states(self):
        ''' d |vec> / d theta.
        '''
        np = len(self._params)
        vecp_list = []
        vec_i = self._ref_state
        if self._opdense:
            for i in range(np):
                th, op = self._params[i], self._ansatz[0][i]
                # factor of 0.5 difference from the main text.
                vec_i = op.expm(-0.5j*th).dot(vec_i)
                vec = -0.5j*op.dot(vec_i)
                for th, op in zip(self._params[i+1:], self._ansatz[0][i+1:]):
                    vec = op.expm(-0.5j*th).dot(vec)
                vecp_list.append(vec)
        else:
            for i in range(np):
                th, op = self._params[i], self._ansatz[0][i]
                opth = -0.5j*th*op
                vec_i = opth.expm()*vec_i
                vec = -0.5j*op*vec_i
                for th, op in zip(self._params[i+1:], self._ansatz[0][i+1:]):
                    opth = -0.5j*th*op
                    vec = opth.expm()*vec
                vecp_list.append(vec)
        self._vecp_list = vecp_list

    #@timeit
    def set_par_states_notefficient(self):
        ''' d |vec> / d theta.
        '''
        np = len(self._params)
        ps_pool = multiprocessing.Pool(self._ncpus)
        args = [(self, i) for i in range(np)]
        vecp_list = ps_pool.map(set_par_state, args)
        ps_pool.close()
        self._vecp_list = vecp_list

    def get_amat(self):
        np = len(self._params)
        amat = numpy.zeros((np, np), dtype=numpy.complex)
        for i in range(np):
            for j in range(i, np):
                if self._opdense:
                    zes = numpy.vdot(self._vecp_list[i], self._vecp_list[j])
                else:
                    zes = self._vecp_list[i].overlap(self._vecp_list[j])
                amat[i, j] = zes
                if i != j:
                    # hermitian component
                    amat[j, i] = numpy.conj(zes)
        return amat

    #@timeit
    def get_cvec_phase(self):
        np = len(self._params)
        cvec = numpy.zeros(np, dtype=numpy.complex)
        # h |vec>
        if self._opdense:
            hvec = self._h.dot(self._state)
            for i in range(np):
                cvec[i] = numpy.vdot(self._vecp_list[i], hvec)
            # energy
            e = numpy.vdot(self._state, hvec).real
            e2 = numpy.vdot(hvec, hvec).real
            # <partial_vec|vec_cur>
            vpv_list = [numpy.vdot(vecp, self._state)  \
                    for vecp in self._vecp_list]
        else:
            hvec = self._h*self._state
            for i in range(np):
                cvec[i] = self._vecp_list[i].overlap(hvec)
            # energy
            e = (self._state.overlap(hvec)).real
            e2 = (hvec.overlap(hvec)).real
            # <partial_vec|vec_cur>
            vpv_list = [vecp.overlap(self._state)  \
                    for vecp in self._vecp_list]

            # add gaussian noise
        e = self.add_error(e)
        e2 = self.add_error(e2)

        # check vpv imaginary time
        if len(vpv_list) > 0:
            res1 = numpy.max(numpy.abs(numpy.asarray(vpv_list).real))
            res2 = numpy.max(numpy.abs(numpy.asarray(vpv_list).imag))
            print(f"significant vpc max real: {res1}, imag: {res2}")

        mp = [[vi*vj for vj in vpv_list] for vi in vpv_list]
        vp = [vi*e for vi in vpv_list]
        return cvec, mp, vp, e, e2

    def save_ansatz(self):
        with h5py.File("ansatz.h5", "w") as f:
            # initial state params
            f["/params"] = self._params_init
            # ansatz operator labels
            f["/ansatz_code"] = self._ansatz[1]
            # ngates
            f["/ngates"] = self._ngates
            # reference state
            f["/ref_state"] = self._ref_state
            

    def save_state(self, t):
        with h5py.File("state.h5", "w") as f:
            f["t"] = t
            if self._opdense:
                f["state"] = self._state
            else:
                f["state"] = self._statevec.full().reshape(-1)


def minv_error(m, minv):
    n = m.shape[0]
    res = minv.dot(m)
    res = numpy.max(numpy.abs(res - numpy.eye(n)))
    return res


def get_minv(a, delta=0):
    ap = a + delta*numpy.eye(a.shape[0])
    ainv = numpy.linalg.pinv(ap)
    return ainv


def fun_cost(params, ansatz, ref_state, h):
    state = get_ansatz_state(params, ansatz, ref_state)
    res = h.matrix_element(state, state)
    return res.real


def fun_jac(params, ansatz, ref_state, h):
    # - d <var|h|var> / d theta
    np = len(ansatz)
    vec = get_ansatz_state(params, ansatz, ref_state)
    opdense = isinstance(vec, numpy.ndarray)
    # <vec|h
    if opdense:
        h_vec = h.dot(vec)
    else:
        h_vec = h*vec

    jac = []
    state_i = ref_state
    if opdense:
        for i in range(np):
            op = ansatz[i]
            state_i = op.expm(-0.5j*params[i]).dot(state_i)
            state = op.dot(state_i)
            for theta, op in zip(params[i+1:], ansatz[i+1:]):
                state = op.expm(-0.5j*theta).dot(state)
            zes = numpy.vdot(h_vec, state)
            jac.append(zes.imag)
    else:
        for i in range(np):
            opth = -0.5j*params[i]*ansatz[i]
            state_i = opth.expm()*state_i
            state = op*state_i
            for theta, op in zip(params[i+1:], ansatz[i+1:]):
                opth = -0.5j*theta*op
                state = opth.expm()*state
            zes = h_vec.overlap(state)
            jac.append(zes.imag)
    res = numpy.array(jac)
    return res


def get_ansatz_state(params, ansatz, ref_state):
    state = ref_state
    opdense = isinstance(state, numpy.ndarray)
    for theta, op in zip(params, ansatz):
        if opdense:
            opth_expm = op.expm(-0.5j*theta)
            state = opth_expm.dot(state)
        else:
            opth = -0.5j*theta*op
            state = opth.expm()*state
    return state


##@timeit
def generate_op_pools(nq,
        order,
        pool,
        ):
    '''
    operator pool construction.
    todo: group operators into groups, and promote unitaries in commuting groups to be appended first.
    '''
    from model import get_sxyz_ops
    sops_list = get_sxyz_ops(nq)

    # operator and label
    op_pools = []
    op_pool = [[], []]
    sind_temp = numpy.zeros(order*2, dtype=numpy.int)
    if pool is None or pool[0] == '-':
        # complete pool for adapt-vqe and avqds
        for iorder in range(order):
            for ii in itertools.combinations(range(nq), iorder+1):
                for ss in itertools.product(range(3), repeat=iorder+1):
                    op = 1
                    sind = sind_temp.copy()
                    for n, i, s in zip(itertools.count(), ii, ss):
                        op *= sops_list[s][i]
                        sind[2*n:2*n+2] = [i, s+1]
                    op_pool[0].append(op)
                    op_pool[1].append(sind)
        op_pools.append(op_pool)
        op_pool = [[], []]

    # optional pool for avqds
    if pool in ["XYZ", "-XYZ"]:
        # open boundary condition
        # z operators
        for i in range(nq):
            sind = sind_temp.copy()
            sind[:2] = [i, 3]
            op_pool[0].append(sops_list[2][i])
            op_pool[1].append(sind)
        # xx and yy
        for i in range(nq-1):
            op_pool[0].append(sops_list[0][i]*sops_list[0][i+1])
            op_pool[1].append([i, 1, i+1, 1])
            op_pool[0].append(sops_list[1][i]*sops_list[1][i+1])
            op_pool[1].append([i, 2, i+1, 2])
        op_pools.append(op_pool)
    elif pool in ["Ising", "-Ising", "ZZXZ", "-ZZXZ"]:
        # periodic boundary condition
        # x operators
        for i in range(nq):
            sind = sind_temp.copy()
            sind[:2] = [i, 1]
            op_pool[0].append(sops_list[0][i])
            op_pool[1].append(sind)
        # zz
        for i in range(-1, nq-1):
            op_pool[0].append(sops_list[2][i]*sops_list[2][i+1])
            op_pool[1].append([i, 3, i+1, 3])
        op_pools.append(op_pool)

    if pool in ["ZZXZ", "-ZZXZ"]:
         # z operators
        for i in range(nq):
            sind = sind_temp.copy()
            sind[:2] = [i, 2]
            op_pool[0].append(sops_list[2][i])
            op_pool[1].append(sind)
            
    if pool in ["Heis", "-Heis"]:
        #xx, yy, and zz
        for i in range(nq-1):
            op_pool[0].append(sops_list[0][i]*sops_list[0][i+1])
            op_pool[1].append([i, 1, i+1, 1])
            op_pool[0].append(sops_list[1][i]*sops_list[1][i+1])
            op_pool[1].append([i, 2, i+1, 2])
            op_pool[0].append(sops_list[2][i]*sops_list[2][i+1])
            op_pool[1].append([i, 3, i+1, 3])  
        op_pools.append(op_pool)
        
    print(f"op pool size: {[len(op_pool[0]) for op_pool in op_pools]}")

    return op_pools





def set_par_state(args):
    ans, i = args
    vec_i = ans._ref_state
    if ans._opdense:
        th, op = ans._params[i], ans._ansatz[0][i]
        # factor of 0.5 difference from the main text.
        vec_i = op.expm(-0.5j*th).dot(vec_i)
        vec = -0.5j*op.dot(vec_i)
        for th, op in zip(ans._params[i+1:], ans._ansatz[0][i+1:]):
            vec = op.expm(-0.5j*th).dot(vec)
    else:
        th, op = ans._params[i], ans._ansatz[0][i]
        opth = -0.5j*th*op
        vec_i = opth.expm()*vec_i
        vec = -0.5j*op*vec_i
        for th, op in zip(ans._params[i+1:], ans._ansatz[0][i+1:]):
            opth = -0.5j*th*op
            vec = opth.expm()*vec
    return vec

