# author: Yongxin Yao (yxphysice@gmail.com)
import numpy
from qutip import qeye, sigmax, sigmay, sigmaz, tensor, Qobj


class spinModel:
    '''
    Base class. defines some common functions for various spin models.
    '''
    def __init__(self, nsite):
        '''
        Take number of sites, and define the single site spin operators.
        '''
        self._nsite = nsite
        self.set_sops()

    def set_sops(self):
        '''
        set up site-wise sx, sy, sz operators.
        '''
        self._sx_list, self._sy_list, self._sz_list = get_sxyz_ops(self._nsite)

    def get_loweste_states(self, t):
        '''
        get the lowest three eigenvalues and eigenstates.
        '''
        # set the current Hamiltonian.
        self.set_h(t)
        # get the lowest three eigenvalues and eigenstates.
        w, v = self._h.eigenstates(eigvals=3)
        return w, v

    def get_h_expval(self, t, vec):
        '''
        get Hamiltonian expectation value.
        '''
        # set the current Hamiltonian.
        self.set_h(t)
        # convert the state vector vec to Q object if it is an array.
        if isinstance(vec, numpy.ndarray):
            vec = Qobj(vec)
        # return the real expectation value.
        return self._h.matrix_element(vec, vec).real

    def get_h(self, t):
        '''
        get Hamiltonian.
        '''
        # set the current Hamiltonian.
        self.set_h(t)
        # return the current Hamiltonian.
        return self._h


class xyzmodel(spinModel):
    '''
    The Lieb-Schultz-Mattis XYZ model.
    '''
    def __init__(self,
            nsite=2,  # number of sites
            T=0.5,    # Linear ramp rate parameter (t/T)
            gi=1,     # Linear ramp protocol parameter
            gf=-1,    # gi + (gf - gi)*t/T
            hz=-0.7,  # transverse field
            J=1,      # ferro magnetic coupling. energy unit.
            ):
        # set up single site spin operators.
        super().__init__(nsite)
        # assign various inner-class properties.
        self._T = T
        self._gi = gi
        self._gf = gf
        self._hz = hz
        self._J = J
        self._t = None
        self._corr_ops = None  # spin-correlation operators
        # set up Hamiltonian terms
        self.set_terms()

    def set_terms(self):
        '''
        set up Hamiltonian terms in open boundary condition.
        '''
        self._xx, self._yy, self._z = 0, 0, self._sz_list[-1]
        for i in range(self._nsite-1):
            self._xx += self._sx_list[i]*self._sx_list[i+1]
            self._yy += self._sy_list[i]*self._sy_list[i+1]
            self._z += self._sz_list[i]

    @property
    def corr_ops(self):
        '''
        get spin correlation operators.
        '''
        if self._corr_ops is None:
            # set up spin correlation operators.
            self.setup_corr_ops()
        return self._corr_ops

    def setup_corr_ops(self):
        # set up head-nearest neighbour, head-tail spin correlation operators.
        self._corr_ops = [
                (self._sx_list[0]*self._sx_list[1]),
                (self._sx_list[0]*self._sx_list[-1]),
                (self._sy_list[0]*self._sy_list[1]),
                (self._sy_list[0]*self._sy_list[-1]),
                ]

    def set_h(self, t):
        '''
        set the current Hamiltonian.
        '''
        if self._t is None or abs(self._t-t) > 1e-7:
            # initial self._t is None. Keep the initial Hamiltonian up to 1e-7.
            if t > self._T:
                # take care of post-linear ramp
                t = self._T
            # change the Hamiltonian according to linear ramp.
            gamma = self._gi + (self._gf - self._gi)*t/self._T
            jx, jy = -self._J*(1 + gamma), -self._J*(1 - gamma)
            h = jx*self._xx + jy*self._yy + self._hz*self._z
            # update time and Hamiltonian
            self._t = t
            self._h = h

    def h_terms(self, t):
        '''
        list of Hamiltonian terms to be run through.
        '''
        if t > self._T:
            t = self._T
        gamma = self._gi + (self._gf - self._gi)*t/self._T
        jx, jy = -self._J*(1 + gamma), -self._J*(1 - gamma)
        hz = self._hz
        for i in range(self._nsite):
            yield hz*self._sz_list[i]
        for i in range(self._nsite-1):
            yield jx*self._sx_list[i]*self._sx_list[i+1]
        for i in range(self._nsite-1):
            yield jy*self._sy_list[i]*self._sy_list[i+1]


class isingmodel(spinModel):
    def __init__(self,
            nsite=2,  # number of sites
            T=1.6,    # total simulation time
            hx=-2.0,  # transverse field
            J=1,      # nearest-neighbor ZZ coupling: -J Z[i]*Z[i+1]. Sets units of energy.
            T1=0.5,   # Linear ramp rate parameter (t/T)
            mode=0,   # 0: sudden quench; others: linear ramp
            ):
        # set up single site spin operators.
        super().__init__(nsite)
        # assign various inner-class properties.
        self._hx = hx
        self._J = J
        self._T = T
        self._T1 = T1
        self._t = None
        self._mode = mode
        # set up Hamiltonian terms
        self.set_terms()

    def set_terms(self):
        '''
        set up Hamiltonian terms in open boundary condition.
        '''
        self._x, self._zz = 0, 0
        # periodic boundary condition
        for i in range(-1, self._nsite-1):
            self._x += self._sx_list[i]
            self._zz += self._sz_list[i]*self._sz_list[i+1]

    def set_h(self, t):
        '''
        set the current Hamiltonian.
        '''
        if self._t is None or abs(self._t-t) > 1e-7:
            # initial self._t is None. Keep the initial Hamiltonian up to 1e-7.
            if self._mode == 0:
                # sudden quench
                if t < 1.e-6:
                    # initial Hamiltonian
                    if self._t is None or self._t > 0:
                        self._h = -self._zz
                        self._t = -1
                else:
                    # quench Hamiltonian, only need to be set once.
                    if self._t < 0:
                        self._h = -self._zz + self._hx*self._x
                        self._t = 1
            else:
                if t < self._T1:
                    # linear ramp of transverse field
                    hx = t/self._T1*self._hx
                else:
                    # fixed for post-linear ramp
                    hx = self._hx
                # update time and Hamiltonian
                self._h = -self._zz + hx*self._x
                self._t = t

    def h_terms(self, t):
        '''
        list of Hamiltonian terms to be run through.
        '''
        if self._mode == 0:
            hx = self._hx
        else:
            if t < self._T1:
                hx = t/self._T1*self._hx
            else:
                hx = self._hx
        # zz terms
        for i in range(-1, self._nsite-1):
            yield -self._sz_list[i]*self._sz_list[i+1]
        # x terms
        for i in range(-1, self._nsite-1):
            yield hx*self._sx_list[i]


class zzxzmodel(spinModel):
    '''mixed field Ising model. only consider sudden quench protocol is coded.
    '''
    def __init__(self,
            nsite=2,   # number of sites
            T=6,       # total simulation time
            Jzz_init = None, # initial (pre-quench) value of J (assuming PBC), Jzz > 0 is FM.
            hx_init = None, # initial (pre-quench) value of hx
            hz_init = None, # initial (pre-quench) value of hz
            hs_init= None,  # initial (pre-quench) value of hs
            Jzz = 1.0, # Hamiltonian is -Jzz*Z_i*Z_{i+1} -> Jzz > 0 is FM
            hx=-2.0,   # x-field
            hz=0.0,    # z-field, break integrability
            hs = 0.0,  # staggered magnetic field along z
            ):
        # set up single site spin operators.
        super().__init__(nsite)
        if (Jzz_init is None):
            Jzz_init = numpy.ones(nsite)
        if (hx_init is None):
            hx_init = numpy.zeros(nsite)
        if (hz_init is None):
            hz_init = numpy.zeros(nsite)
        if (hs_init is None):
            hs_init = numpy.zeros(nsite)
        # assign various inner-class properties.
        self._T = T
        self._Jzz_init = Jzz_init
        self._hx_init = hx_init
        self._hz_init = hz_init
        self._hs_init = hs_init
        self._Jzz = Jzz
        self._hx = hx
        self._hz = hz
        self._hs = hs
        self._h = None
        self._t = None
        # set up Hamiltonian terms for post-quench Hamiltonian H_dyn
        self.set_terms()

    def set_terms(self):
        '''
        set up Hamiltonian terms in open boundary condition.
        '''
        self._x, self._zz, self._z, self._zs = 0, 0, 0, 0
        # periodic boundary condition
        for i in range(0, self._nsite):
            self._x += self._sx_list[i]
            self._zz += self._sz_list[i]*self._sz_list[(i+1)%self._nsite]
            self._z += self._sz_list[i]
            self._zs += numpy.sign((-1.)**(i+1))*self._sz_list[i]

    def set_h(self, t):
        '''
        set the current Hamiltonian.
        '''
        if self._h is None and t < 1.e-6:
            # initial Hamiltonian
            # self._h = -self._zz
            # allow for site dependent values of J, hx, hz in initial Hamiltonian
            self._h = 0
            for i in range(0, self._nsite):
                self._h += self._hx_init[i]*self._sx_list[i]
                self._h += -self._Jzz_init[i]*self._sz_list[i] * self._sz_list[(i + 1) % self._nsite]
                self._h += self._hz_init[i]*self._sz_list[i]
                self._h += numpy.sign((-1.)**(i+1))*self._hs_init[i]*self._sz_list[i]
            self._t = -1
        elif self._t < 0 and t > 1.e-6:
            # only need to be assigned once for sudden quench
            self._h = -self._Jzz*self._zz + self._hx*self._x + self._hz*self._z + self._hs*self._zs
            self._t = 1
        #else:
         #   raise Exception("Time step too small. Choose timestep > 1.e-6.")

    def h_terms(self, t):
        '''
        list of Hamiltonian terms to be run through.
        '''
        # zz terms
        for i in range(0, self._nsite):
            yield -self._Jzz*self._sz_list[i]*self._sz_list[(i+1) % self._nsite]
        # x terms
        if abs(self._hx) > 1.e-12:
            for i in range(0, self._nsite):
                yield self._hx*self._sx_list[i]
        # z term
        if abs(self._hz) > 1.e-12:
            for i in range(0, self._nsite):
                yield self._hz*self._sz_list[i]
        # staggered z term
        if abs(self._hs) > 1.e-12:
            for i in range(0, self._nsite):
                yield self._hs * self._sz_list[i]*numpy.sign((-1.)**(i+1))

    def get_staggered_magnetization(self, vec):
        '''
        return the staggered magnetization in state vec
        '''
        mag_s = 0
        for i in range(0, self._nsite):
            mag_s += self._sz_list[i]*numpy.sign((-1.)**(i+1))
        mag_s /= self._nsite
        #print(f"mag_s = {mag_s}")
        if isinstance(vec, numpy.ndarray):
            vec = Qobj(vec)
        return mag_s.matrix_element(vec,vec).real

    def get_sz(self, vec):
        '''
        return the expectation value of Z_site
        '''
        if isinstance(vec, numpy.ndarray):
            vec = Qobj(vec)
        sz = numpy.zeros(self._nsite)
        for site in range(0, self._nsite):
            sz[site] = self._sz_list[site].matrix_element(vec,vec).real
        return sz

    # return the real expectation value.
    #return self._h.matrix_element(vec, vec).real
    
    
    
    
    
    
    
    
    
    
    
    

class heis(spinModel):
    '''mixed field Ising model. only consider sudden quench protocol is coded.
    '''
    def __init__(self,
            nsite=2,   # number of sites
            T=6,       # total simulation time
            Jzz_init = None, # initial (pre-quench) value of J (assuming PBC), Jzz > 0 is FM.
            Jxx_init = None, # initial (pre-quench) value of Jxx
            Jyy_init = None, # initial (pre-quench) value of Jyy
            hs_init= None,  # initial (pre-quench) value of hs
            Jzz = 1.0, # Hamiltonian is -Jzz*Z_i*Z_{i+1} -> Jzz > 0 is FM
            Jxx=-2.0,   # x-field
            Jyy=0.0,    # z-field, break integrability
            hs = 0.0,  # staggered magnetic field along z
            ):
        # set up single site spin operators.
        super().__init__(nsite)
        if (Jzz_init is None):
            Jzz_init = numpy.ones(nsite)
        if (Jxx_init is None):
            Jxx_init = numpy.zeros(nsite)
        if (Jyy_init is None):
            Jyy_init = numpy.zeros(nsite)
        if (hs_init is None):
            hs_init = numpy.zeros(nsite)
        # assign various inner-class properties.
        self._T = T
        self._Jzz_init = Jzz_init
        self._Jxx_init = Jxx_init
        self._Jyy_init = Jyy_init
        self._hs_init = hs_init
        self._Jzz = Jzz
        self._Jxx = Jxx
        self._Jyy = Jyy
        self._hs = hs
        self._h = None
        self._t = None
        # set up Hamiltonian terms for post-quench Hamiltonian H_dyn
        self.set_terms()

    def set_terms(self):
        '''
        set up Hamiltonian terms in open boundary condition.
        '''
        self._xx, self._zz, self._yy = 0, 0, 0
        # open boundary condition
        for i in range(0, self._nsite-1):
            self._xx += self._sx_list[i]*self._sx_list[(i+1)%self._nsite]
            self._zz += self._sz_list[i]*self._sz_list[(i+1)%self._nsite]
            self._yy += self._sy_list[i]*self._sy_list[(i+1)%self._nsite]
  

    def set_h(self, t):
        '''
        set the current Hamiltonian.
        '''
        if self._h is None and t < 1.e-6:
            # initial Hamiltonian
            # self._h = -self._zz
            # allow for site dependent values of J, Jxx, Jyy in initial Hamiltonian
            self._h = 0
            for i in range(0, self._nsite-1):
                self._h += self._Jxx_init[i]*self._sx_list[i] * self._sx_list[(i + 1) % self._nsite]
                self._h += self._Jyy_init[i]*self._sy_list[i] * self._sy_list[(i + 1) % self._nsite]
                self._h += self._Jzz_init[i]*self._sz_list[i] * self._sz_list[(i + 1) % self._nsite]

            self._t = -1
        elif self._t < 0 and t > 1.e-6:
            # only need to be assigned once for sudden quench
            self._h = self._Jzz*self._zz + self._Jxx*self._xx + self._Jyy*self._yy
            self._t = 1
        #else:
         #   raise Exception("Time step too small. Choose timestep > 1.e-6.")

    def h_terms(self, t):
        '''
        list of Hamiltonian terms to be run through.
        '''
        # zz terms
        for i in range(0, self._nsite-1):
            yield self._Jzz*self._sz_list[i]*self._sz_list[(i+1) % self._nsite]
        # xx terms
        if abs(self._Jxx) > 1.e-12:
            for i in range(0, self._nsite-1):
                yield self._Jxx*self._sx_list[i]*self._sx_list[(i+1) % self._nsite]
        # yy term
        if abs(self._Jyy) > 1.e-12:
            for i in range(0, self._nsite-1):
                yield self._Jyy*self._sy_list[i]*self._sy_list[(i+1) % self._nsite]


    def get_staggered_magnetization(self, vec):
        '''
        return the staggered magnetization in state vec
        '''
        mag_s = 0
        for i in range(0, self._nsite):
            mag_s += self._sz_list[i]*numpy.sign((-1.)**(i+1))
        mag_s /= self._nsite
        #print(f"mag_s = {mag_s}")
        if isinstance(vec, numpy.ndarray):
            vec = Qobj(vec)
        return mag_s.matrix_element(vec,vec).real

    def get_sz(self, vec):
        '''
        return the expectation value of Z_site
        '''
        if isinstance(vec, numpy.ndarray):
            vec = Qobj(vec)
        sz = numpy.zeros(self._nsite)
        for site in range(0, self._nsite):
            sz[site] = self._sz_list[site].matrix_element(vec,vec).real
        return sz

    # return the real expectation value.
    #return self._h.matrix_element(vec, vec).real
    
    
    

    
    
    
    
    
    

def get_sxyz_ops(nsite):
    '''
    set up site-wise sx, sy, sz operators.
    '''
    si = qeye(2)
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    sx_list = []
    sy_list = []
    sz_list = []

    op_list = [si for i in range(nsite)]
    for i in range(nsite):
        op_list[i] = sx
        sx_list.append(tensor(op_list))
        op_list[i] = sy
        sy_list.append(tensor(op_list))
        op_list[i] = sz
        sz_list.append(tensor(op_list))
        # reset
        op_list[i] = si
    return [sx_list, sy_list, sz_list]
