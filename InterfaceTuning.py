__author__ = "Matteo Lulli"
__copyright__ = "Copyright (c) 2025 Matteo Lulli (lullimat/idea.deploy), matteo.lulli@gmail.com"
__credits__ = ["Matteo Lulli"]
__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
__version__ = "0.1"
__maintainer__ = "Matteo Lulli"
__email__ = "matteo.lulli@gmail.com"
__status__ = "Development"

import sys
sys.path.append("../../")

import numpy as np
import sympy as sp

from idpy.IdpyStencils.IdpyStencils import IDStencils
from idpy.LBM.MultiPhase import ShanChenMultiPhase
from idpy.LBM.MultiPhaseLoopChecks import CheckUConvergenceSCMP, CheckCenterOfMassDeltaPConvergence
from idpy.IdpyCode import IdpyMemory

from scipy.interpolate import UnivariateSpline

def CheckMaxUNAN(lbm):
    if np.isnan(lbm.sims_vars['max_u'][-1]):
        return True

def GetFlatAdsorbance(n_line, z_range, grains = 2 ** 9):
    z_fine = np.linspace(z_range[0], z_range[-1], grains)
    _n_g, _n_l = np.amin(n_line), np.amax(n_line)
    n_line_spl = UnivariateSpline(z_range, n_line, k = 5, s = 0)

    def _adsorbance(x):
        _ads_integ = n_line_spl(z_fine) - (_n_l - (_n_l - _n_g) * np.heaviside(z_fine - x, 1))
        _ads_integ_spl = UnivariateSpline(z_fine, _ads_integ, k = 5, s = 0)
        return _ads_integ_spl.integral(z_fine[0], z_fine[-1])

    return lambda _z: _adsorbance(_z)

def GetFlatEquimolarSurface(n_line, z_range, grains = 2 ** 9):
    z_fine = np.linspace(z_range[0], z_range[-1], grains)
    adsorbance_f = GetFlatAdsorbance(n_line, z_range, grains)
    fine_adsorbance = np.array([adsorbance_f(_) for _ in z_fine])

    fine_adsorbance_spl = \
        UnivariateSpline(z_fine, fine_adsorbance, k = 3, s = 0)
    _z_e = fine_adsorbance_spl.roots()[0]
    return _z_e

def GetSigma0(n_line, psi_sym, n_sym, fine2_exp=13):
    psi_f = sp.lambdify([n_sym], psi_sym)
    psi_line = psi_f(n_line)
    x_range = np.arange(len(n_line))
    psi_spl = UnivariateSpline(x_range, psi_line, s=0, k=5)

    x_range_fine = np.linspace(len(n_line) // 2, len(n_line), 2 ** fine2_exp)
    dx_psi_2 = psi_spl.derivative()(x_range_fine) ** 2
    dx = x_range_fine[1] - x_range_fine[0]
    Sigma_0 = np.sum(dx_psi_2) * dx
    return x_range_fine, dx_psi_2, Sigma_0

def GetSigma1(n_line, psi_sym, n_sym, fine2_exp=13):
    psi_f = sp.lambdify([n_sym], psi_sym)
    psi_line = psi_f(n_line)
    x_range = np.arange(len(n_line))
    psi_spl = UnivariateSpline(x_range, psi_line, s=0, k=5)

    x_range_fine = np.linspace(len(n_line) // 2, len(n_line), 2 ** fine2_exp)
    d2x_psi_2 = psi_spl.derivative(2)(x_range_fine) ** 2
    dx = x_range_fine[1] - x_range_fine[0]
    Sigma_1 = np.sum(d2x_psi_2) * dx
    return x_range_fine, d2x_psi_2, Sigma_1

def GetSigma2(n_line, psi_sym, n_sym, fine2_exp=13):
    psi_f = sp.lambdify([n_sym], psi_sym)
    psi_line = psi_f(n_line)
    x_range = np.arange(len(n_line))
    psi_spl = UnivariateSpline(x_range, psi_line, s=0, k=5)

    x_range_fine = np.linspace(len(n_line) // 2, len(n_line), 2 ** fine2_exp)
    d3x_psi_2 = psi_spl.derivative(3)(x_range_fine) ** 2
    dx = x_range_fine[1] - x_range_fine[0]
    Sigma_2 = np.sum(d3x_psi_2) * dx
    return x_range_fine, d3x_psi_2, Sigma_2

def GetSigma3(n_line, psi_sym, n_sym, fine2_exp=13):
    psi_f = sp.lambdify([n_sym], psi_sym)
    psi_line = psi_f(n_line)
    x_range = np.arange(len(n_line))
    psi_spl = UnivariateSpline(x_range, psi_line, s=0, k=5)

    x_range_fine = np.linspace(len(n_line) // 2, len(n_line), 2 ** fine2_exp)
    d4x_psi_2 = psi_spl.derivative(4)(x_range_fine) ** 2
    dx = x_range_fine[1] - x_range_fine[0]
    Sigma_2 = np.sum(d4x_psi_2) * dx
    return x_range_fine, d4x_psi_2, Sigma_2

def GetSigma4(n_line, psi_sym, n_sym, fine2_exp=13):
    psi_f = sp.lambdify([n_sym], psi_sym)
    psi_line = psi_f(n_line)
    x_range = np.arange(len(n_line))
    psi_spl = UnivariateSpline(x_range, psi_line, s=0, k=5)

    x_range_fine = np.linspace(len(n_line) // 2, len(n_line), 2 ** fine2_exp)
    d4x_psi_2 = psi_spl.derivative(5)(x_range_fine) ** 2
    dx = x_range_fine[1] - x_range_fine[0]
    Sigma_2 = np.sum(d4x_psi_2) * dx
    return x_range_fine, d4x_psi_2, Sigma_2

def GetDelta0(n_line, psi_sym, n_sym, fine2_exp=13):
    psi_f = sp.lambdify([n_sym], psi_sym)
    psi_line = psi_f(n_line)
    x_range = np.arange(len(n_line))
    psi_spl = UnivariateSpline(x_range, psi_line, s=0)

    x_range_fine = np.linspace(len(n_line) // 2, len(n_line), 2 ** fine2_exp)
    dx_psi_2 = psi_spl.derivative()(x_range_fine) ** 2
    dx = x_range_fine[1] - x_range_fine[0]
    Delta_0 = np.sum(dx_psi_2 * (x_range_fine - len(n_line) // 2)) * dx
    return x_range_fine, dx_psi_2, Delta_0

def GetDelta1(n_line, psi_sym, n_sym, fine2_exp=13):
    psi_f = sp.lambdify([n_sym], psi_sym)
    psi_line = psi_f(n_line)
    x_range = np.arange(len(n_line))
    psi_spl = UnivariateSpline(x_range, psi_line, s=0)

    x_range_fine = np.linspace(len(n_line) // 2, len(n_line), 2 ** fine2_exp)
    d2x_psi_2 = psi_spl.derivative(2)(x_range_fine) ** 2
    dx = x_range_fine[1] - x_range_fine[0]
    Delta_1 = np.sum(d2x_psi_2 * (x_range_fine - len(n_line) // 2)) * dx
    return x_range_fine, d2x_psi_2, Delta_1

def CheckNucleation(lbm):
    first_flag = False
    if 're_indicator' not in lbm.sims_vars:
        lbm.sims_vars['re_indicator'] = []
        lbm.sims_vars['n_min'] = []
        lbm.sims_vars['n_max'] = []
        first_flag = True
    
    _n_min, _n_max = \
        IdpyMemory.Min(lbm.sims_idpy_memory['n']), IdpyMemory.Max(lbm.sims_idpy_memory['n'])

    lbm.sims_vars['re_indicator'] += [_n_max - _n_min]
    lbm.sims_vars['n_min'] += [_n_min]
    lbm.sims_vars['n_max'] += [_n_max]

    if _n_max - _n_min > 1 * (lbm.sims_vars['n_l'] - lbm.sims_vars['n_g']) / 3.:
        for _ in ['re_indicator', 'n_min', 'n_max']:
            lbm.sims_vars[_] = np.array(lbm.sims_vars[_])
        return True
    else:
        return False

from functools import reduce

class EquimolarRadius:
    def __init__(self, mass = None, n_in_n_out = None, dim_sizes = None):
        if mass is None:
            raise Exception("Paramtere mass must not be None")
        if n_in_n_out is None:
            raise Exception("Parameter n_in_n_out must not be None")
        if dim_sizes is None:
            raise Exception("Parameter dim_sizes must not be None")

        self.mass, self.n_in_n_out, self.dim_sizes = mass, n_in_n_out, dim_sizes
        self.V = reduce(lambda x, y: x * y, self.dim_sizes)

    def GetEquimolarRadius3D(self):
        _r_swap = \
            ((3 / (4 * np.pi)) * (self.mass - self.n_in_n_out[1] * self.V)
             / (self.n_in_n_out[0] - self.n_in_n_out[1]))
        return {'Re': _r_swap ** (1 / 3)}
    
    def GetEquimolarRadius2D(self):
        _r_swap = \
            ((1 / (np.pi)) * (self.mass - self.n_in_n_out[1] * self.V)
             / (self.n_in_n_out[0] - self.n_in_n_out[1]))
        return {'Re': _r_swap ** (1 / 2)}
    
class SurfaceOfTension:
    def __init__(self, n_field = None, f_stencil = None, psi_sym = None, G = None):
        if n_field is None:
            raise Exception("Parameter n_field must not be None")
        if f_stencil is None:
            raise Exception("Parameter f_stencil must not be None")
        if psi_sym is None:
            raise Exception("Parameter psi_sym must not be None")
        if G is None:
            raise Exception("Parameter G must not be None")

        self.n_field, self.f_stencil, self.psi_sym, self.G = \
            n_field, f_stencil, psi_sym, G
        self.dim = len(self.n_field.shape)
        self.dim_center = np.array(list(map(lambda x: x//2, self.n_field.shape)))
        self.dim_sizes = self.n_field.shape
        self.psi_f = sp.lambdify(n, self.psi_sym)

        '''
        Preparing common variables
        '''
        _LPT_class = LatticePressureTensor(self.n_field, self.f_stencil, self.psi_sym, self.G)
        self.LPT = _LPT_class.GetLPT()

        self.r_range = np.arange(self.dim_sizes[2] - self.dim_center[2])

        self.radial_n = self.LPT[0, self.dim_center[0], self.dim_center[1], self.dim_center[2]:]
        self.radial_t = self.LPT[3, self.dim_center[0], self.dim_center[1], self.dim_center[2]:]
        self.radial_profile = self.n_field[self.dim_center[0],
                                           self.dim_center[1], self.dim_center[2]:]
        
    def GetSurfaceTension(self, grains_fine = 2 ** 10, cutoff = 2 ** 7):
        self.r_fine = np.linspace(self.r_range[0], self.r_range[-1], grains_fine)

        self.radial_t_spl = \
            interpolate.UnivariateSpline(self.r_range, self.radial_t, k = 5, s = 0)
        self.radial_n_spl = \
            interpolate.UnivariateSpline(self.r_range, self.radial_n, k = 5, s = 0)
        
        '''
        Rowlinson: 4.217
        '''
        def st(R):
            _p_jump = \
                (self.radial_n[0] - (self.radial_n[0] - self.radial_n[-1]) *
                 np.heaviside(self.r_fine - R, 1))

            _swap_spl = \
                interpolate.UnivariateSpline(self.r_fine, 
                                             (self.r_fine ** 2) *
                                             (_p_jump - self.radial_t_spl(self.r_fine)),
                                             k = 5, s = 0)

            return _swap_spl.integral(self.r_fine[0], self.r_fine[-1]) / (R ** 2)

        _swap_st = np.array([st(rr) for rr in self.r_fine[1:]])
        _swap_st_spl = interpolate.UnivariateSpline(self.r_fine[1:], _swap_st, k = 5, s = 0)
        _swap_rs = optimize.newton(_swap_st_spl.derivative(), x0 = 0)
        _swap_smin = _swap_st_spl(_swap_rs)
        
        return {'sigma_4.217': _swap_smin, 'Rs_4.217': _swap_rs,
                'st_spl_4.217': _swap_st_spl, 'r_fine_4.217': self.r_fine[1:]}

class InterfaceTuningSims:    
    def __init__(self, lang, device, cl_kind, dim_sizes = None, fp32_flag = False):
        if dim_sizes is not None:
            self.dim_sizes = dim_sizes
        else:
            self.dim_sizes = (128, 32)

        self.lang, self.device, self.cl_kind = lang, device, cl_kind
        self.sim = None
        self.fp32_flag = fp32_flag

    def RunFlat(self, f_stencil, n_l, n_g, G, psi_sym, psi_code, dump_name, direction=0):
        sims_params_dict = {
            'dim_sizes': self.dim_sizes, 
            'xi_stencil': IDStencils['LBM']['XI_D2Q9'], 
            'f_stencil': f_stencil,
            'psi_code': psi_code, 
            'psi_sym': psi_sym, 
            'e2_val': 1, 
            'SC_G': G, 
            'tau': 1,
            'lang': self.lang, 
            'device': self.device, 
            'cl_kind': self.cl_kind, 
            'optimizer_flag': True, 
            'fp32_flag': self.fp32_flag
        }

        self.sim = ShanChenMultiPhase(**sims_params_dict)
        if self.fp32_flag:
            self.sim.custom_types = SwitchToFP32(self.sim.custom_types)

        self.sim.InitFlatInterface(n_g=n_g, n_l=n_l, width=self.dim_sizes[0]/2, direction=direction)

        self.sim.MainLoop(
            time_steps = range(0, 2 ** 22 + 1, 2 ** 14), 
            convergence_functions = [CheckUConvergenceSCMP]
        )

        print("Dumping in", dump_name)
        self.sim.sims_dump_idpy_memory += ['n']
        self.sim.DumpSnapshot(file_name = dump_name,
                              custom_types = self.sim.custom_types)
        self.sim.End()

    def GetDensityStrip(self, direction = 0, delta_val = 1):
        _n_swap = self.sim.GetDensityField()
        _dim_center = self.sim.sims_vars['dim_center']
        '''
        I will need to get a strip that is as thick as the largest forcing vector(y) (x2)
        '''
        
        if len(self.sim.params_dict['dim_sizes']) == 2:
            _n_swap = _n_swap[_dim_center[1] - delta_val:_dim_center[1] + delta_val + 1,:]
            
        if len(self.sim.params_dict['dim_sizes']) == 3:
            _n_swap = _n_swap[_dim_center[2] - delta_val:_dim_center[2] + delta_val + 1,
                              _dim_center[1] - delta_val:_dim_center[1] + delta_val + 1,:]            
        return _n_swap        

    def Run2D(self, f_stencil, n_l, n_g, G, psi_sym, psi_code, dump_name, kind, delta_strip, discard_empty=False, get_n_field=False):
        sims_params_dict = {
            'dim_sizes': self.dim_sizes, 
            'xi_stencil': IDStencils['LBM']['XI_D2Q9'], 
            'f_stencil': f_stencil,
            'psi_code': psi_code, 
            'psi_sym': psi_sym, 
            'e2_val': 1, 
            'SC_G': G, 
            'tau': 1,
            'lang': self.lang, 
            'device': self.device, 
            'cl_kind': self.cl_kind, 
            'optimizer_flag': True, 
            'fp32_flag': self.fp32_flag
        }

        self.sim = ShanChenMultiPhase(**sims_params_dict)
        self.sim.sims_vars['empty'] = False

        if kind == 'droplet':
            self.sim.InitRadialInterface(n_g=n_g, n_l=n_l, R=self.dim_sizes[0]/4, full_flag=True)
        if kind == 'bubble':
            self.sim.InitRadialInterface(n_g=n_g, n_l=n_l, R=self.dim_sizes[0]/4, full_flag=False)

        self.sim.MainLoop(
            time_steps = range(0, 2 ** 22 + 1, 2 ** 14), 
            convergence_functions = [CheckUConvergenceSCMP, CheckCenterOfMassDeltaPConvergence, CheckMaxUNAN]
        )

        status_flag = None
        if np.isnan(self.sim.sims_vars['max_u'][-1]):
            print("The", kind, "is not stable (nan)! Dumping Empty simulation")

            self.sim.sims_dump_idpy_memory_flag = False
            self.sim.sims_vars['empty'] = 'nan'
            self.sim.DumpSnapshot(file_name = dump_name,
                                  custom_types = self.sim.custom_types)
            status_flag = "unstable"        

        elif abs(self.sim.sims_vars['delta_p'][-1]) < 1e-9:
            print("The", kind, "has burst! Dumping Empty simulation")
            '''
            Writing empty simulation file
            '''
            self.sim.sims_dump_idpy_memory_flag = False
            self.sim.sims_vars['empty'] = 'burst'
            self.sim.DumpSnapshot(file_name = dump_name,
                                  custom_types = self.sim.custom_types)
            status_flag = "burst"
                    
        elif not self.sim.sims_vars['is_centered_seq'][-1] and not discard_empty:
            print("The", kind, "is not centered! Dumping Empty simulation")
            '''
            Writing empty simulation file
            '''
            self.sim.sims_dump_idpy_memory_flag = False
            self.sim.sims_vars['empty'] = 'center'
            self.sim.DumpSnapshot(file_name = dump_name,
                                     custom_types = self.sim.custom_types)
            status_flag = "non-centered"

        else:
            self.sim.sims_vars['n_in_n_out'] = np.array(self.sim.sims_vars['n_in_n_out'])
            print("Dumping in", dump_name)
            self.sim.sims_dump_idpy_memory_flag = False
            self.sim.sims_vars['n_strip'] = np.array(self.GetDensityStrip(delta_val=delta_strip))
            self.sim.DumpSnapshot(file_name = dump_name, custom_types = self.sim.custom_types)
            status_flag = "stable"
            
        n_field = None
        if get_n_field:
            n_field = self.sim.GetDensityField()

        self.sim.End()

        if get_n_field:
            return n_field
        
        return status_flag

    def Run3D(self, f_stencil, n_l, n_g, G, psi_sym, psi_code, dump_name, kind, delta_strip, get_n_field=False):
        sims_params_dict = {
            'dim_sizes': self.dim_sizes, 
            'xi_stencil': IDStencils['LBM']['XI_D3Q19'], 
            'f_stencil': f_stencil,
            'psi_code': psi_code, 
            'psi_sym': psi_sym, 
            'e2_val': 1, 
            'SC_G': G, 
            'tau': 1,
            'lang': self.lang, 
            'device': self.device, 
            'cl_kind': self.cl_kind, 
            'optimizer_flag': True, 
            'fp32_flag': self.fp32_flag
        }

        self.sim = ShanChenMultiPhase(**sims_params_dict)
        self.sim.sims_vars['empty'] = False

        if kind == 'droplet':
            self.sim.InitRadialInterface(n_g=n_g, n_l=n_l, R=self.dim_sizes[0]/4, full_flag=True)
        if kind == 'bubble':
            self.sim.InitRadialInterface(n_g=n_g, n_l=n_l, R=self.dim_sizes[0]/4, full_flag=False)

        self.sim.MainLoop(
            time_steps = range(0, 2 ** 22 + 1, 2 ** 11), 
            convergence_functions = [CheckUConvergenceSCMP, CheckCenterOfMassDeltaPConvergence, CheckMaxUNAN]
        )

        if np.isnan(self.sim.sims_vars['max_u'][-1]):
            print("The", kind, "is not stable (nan)! Dumping Empty simulation")

            self.sim.sims_dump_idpy_memory_flag = False
            self.sim.sims_vars['empty'] = 'nan'
            self.sim.DumpSnapshot(file_name = dump_name,
                                  custom_types = self.sim.custom_types)

        elif abs(self.sim.sims_vars['delta_p'][-1]) < 1e-9:
            print("The", kind, "has bursted! Dumping Empty simulation")
            '''
            Writing empty simulation file
            '''
            self.sim.sims_dump_idpy_memory_flag = False
            self.sim.sims_vars['empty'] = 'burst'
            self.sim.DumpSnapshot(file_name = dump_name,
                                  custom_types = self.sim.custom_types)
                    
        elif not self.sim.sims_vars['is_centered_seq'][-1]:
            print("The", kind, "is not centered! Dumping Empty simulation")
            '''
            Writing empty simulation file
            '''
            self.sim.sims_dump_idpy_memory_flag = False
            self.sim.sims_vars['empty'] = 'center'
            self.sim.DumpSnapshot(file_name = dump_name,
                                     custom_types = self.sim.custom_types)
        else:
            self.sim.sims_vars['n_in_n_out'] = np.array(self.sim.sims_vars['n_in_n_out'])

            print("Dumping in", dump_name)
            self.sim.sims_dump_idpy_memory_flag = False
            self.sim.sims_vars['n_strip'] = np.array(self.GetDensityStrip(delta_val=delta_strip))
            self.sim.DumpSnapshot(file_name = dump_name,
                                custom_types = self.sim.custom_types)

        n_field = None
        if get_n_field:
            n_field = self.sim.GetDensityField()

        self.sim.End()

        if get_n_field:
            return n_field

    def RunNucleation(self, f_stencil, n_init, n_g, n_l, G, psi_sym, kBT, psi_code, reps, dump_name, discard_empty=False, 
                      get_n_field=False, print_flag=False, seed=1):
        sims_params_dict = {
            'dim_sizes': self.dim_sizes, 
            'xi_stencil': IDStencils['LBM']['XI_D2Q9'], 
            'f_stencil': f_stencil,
            'psi_code': psi_code, 
            'psi_sym': psi_sym, 
            'e2_val': 1, 
            'SC_G': G, 
            'tau': 1,
            'lang': self.lang, 
            'device': self.device, 
            'cl_kind': self.cl_kind, 
            'optimizer_flag': True,
            'fluctuations': 'Gross2011', 
            'prng_distribution': 'gaussian', 
            'indep_gaussian': False, 
            'prng_kind': 'MMIX',
            'prng_init_from': 'numpy', 
            'init_seed': seed, 
            'fp32_flag': self.fp32_flag
        }

        self.sim = ShanChenMultiPhase(**sims_params_dict)
        self.sim.sims_vars['empty'] = False

        nucleation_times = []
        flag_fields = []
        bub_threshold = (n_l + n_g) / 2

        for rep in range(reps):
            if 're_indicator' in self.sim.sims_vars:
                del self.sim.sims_vars['re_indicator']

            self.sim.InitFlatInterface(n_g=n_init, n_l=n_init, width=self.dim_sizes[0]/2, direction=0)
            self.sim.sims_vars['n_g'], self.sim.sims_vars['n_l'] = n_g, n_l

            self.sim.MainLoopGross2011SRT(
                time_steps = range(0, 2 ** 22 + 1, 2 ** 4), 
                convergence_functions = [CheckNucleation], 
                profiling = False, kBT = kBT, n0 = sp.Symbol('ln_0'), print_flag=print_flag
            )

            nucleation_times += [self.sim.sims_vars['time_steps'][-1]]
            print("rep:", rep, "time step:", self.sim.sims_vars['time_steps'][-1])
            flag_field = np.array(self.sim.GetDensityField() < bub_threshold, dtype = np.byte)
            flag_fields += [flag_field]

        # self.sim.MainLoopGross2011SRT(
        #     time_steps = range(0, 2 ** 7 + 1, 2 ** 4), 
        #     convergence_functions = [CheckUConvergenceSCMP], 
        #     profiling = False, kBT = kBT, n0 = sp.Symbol('ln_0'), print_flag=print_flag
        # )
            
        n_field = None
        if get_n_field:
            n_field = self.sim.GetDensityField()

        self.sim.End()

        if get_n_field:
            return n_field
        else:
            return {'nuc_times': np.array(nucleation_times), 
                    'ffield': np.array(flag_fields)}