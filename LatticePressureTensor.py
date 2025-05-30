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

import numpy as np
import sympy as sp

from idpy.Utils.IdpySymbolic import TaylorTuples

"""
One-dimensional projections
"""

def _p_t_e4(_n, _G, _w_list, psi_sym, n_sym):
    '''
    Computing psi(x), psi(x + 1), psi(x - 1)
    '''
    _psi_f = sp.lambdify(n_sym, psi_sym)
    _psi_swap = _psi_f(_n)
    _psi_swap_m1 = np.append(_psi_swap[-1], _psi_swap[:-1])
    _psi_swap_p1 = np.append(_psi_swap[1:], _psi_swap[0])
    '''
    Computing tangential component
    '''
    _p_t = _n/3 + \
        _G * (_w_list[1]) * _psi_swap * (_psi_swap_p1 + _psi_swap_m1) + \
        _G * _w_list[0] * (_psi_swap ** 2)

    return {'p_t': _p_t, 'psi': _psi_swap, 'psi_p1': _psi_swap_p1, 'psi_m1': _psi_swap_m1}

def _p_t_e6(_n, _G, _w_list, psi_sym, n_sym):
    '''
    Computing psi(x + 2), psi(x - 2)
    '''
    _swap_e4 = _p_t_e4(_n, _G, _w_list, psi_sym, n_sym)
    _psi_swap_m2 = np.append(_swap_e4['psi_m1'][-1], _swap_e4['psi_m1'][:-1])
    _psi_swap_p2 = np.append(_swap_e4['psi_p1'][1:], _swap_e4['psi_p1'][0])    
    '''
    Computing normal component
    '''
    _swap_e4['p_t'] += 4 * _G * _w_list[2] * (_swap_e4['psi'] ** 2)
    
    return {**_swap_e4, 'psi_p2': _psi_swap_p2, 'psi_m2': _psi_swap_m2}

def _p_t_e8(_n, _G, _w_list, psi_sym, n_sym):
    '''
    Computing psi(x), psi(x + 1), psi(x - 1)
    '''
    _swap_e6 = _p_t_e6(_n, _G, _w_list, psi_sym, n_sym)
    '''
    Computing normal component
    '''
    _swap_e6['p_t'] += \
        4 * _G * _w_list[3] * _swap_e6['psi'] * (_swap_e6['psi_p1'] + _swap_e6['psi_m1']) + \
        _G * (_w_list[3] / 2 + 2 * _w_list[4]) * _swap_e6['psi'] * (_swap_e6['psi_p2'] + _swap_e6['psi_m2']) + \
        _G * (_w_list[3] + 4 * _w_list[4]) * _swap_e6['psi_p1'] * _swap_e6['psi_m1']
    
    return {**_swap_e6}

def _p_t_e10T(_n, _G, _w_list, psi_sym, n_sym):
    '''
    Computing psi(x), psi(x + 1), psi(x - 1)
    '''
    _swap_e8 = _p_t_e8(_n, _G, _w_list, psi_sym, n_sym)
    _psi_swap_m3 = np.append(_swap_e8['psi_m2'][-1], _swap_e8['psi_m2'][:-1])
    _psi_swap_p3 = np.append(_swap_e8['psi_p2'][1:], _swap_e8['psi_p2'][0])    
    
    '''
    Computing normal component
    '''
    _swap_e8['p_t'] += \
        _G * (3 * _w_list[6]) * _swap_e8['psi'] * (_psi_swap_p3 + _psi_swap_m3) + \
        _G * (9 * _w_list[5]) * (_swap_e8['psi'] ** 2) + \
        _G * (6 * _w_list[6]) * (_swap_e8['psi_m2'] * _swap_e8['psi_p1'] + _swap_e8['psi_p2'] * _swap_e8['psi_m1'])
    
    return {**_swap_e8, 'psi_p3': _psi_swap_p3, 'psi_m3': _psi_swap_m3}

def _p_n_e4(_n, _G, _w_list, psi_sym, n_sym):
    '''
    Computing psi(x), psi(x + 1), psi(x - 1)
    '''
    _psi_f = sp.lambdify(n_sym, psi_sym)
    _psi_swap = _psi_f(_n)
    _psi_swap_m1 = np.append(_psi_swap[-1], _psi_swap[:-1])
    _psi_swap_p1 = np.append(_psi_swap[1:], _psi_swap[0])
    '''
    Computing normal component
    '''
    _p_n = _n/3 + \
        _G * (_w_list[0] + 2 * _w_list[1]) * \
        _psi_swap * (_psi_swap_p1 + _psi_swap_m1) / 2
    
    return {'p_n': _p_n, 'psi': _psi_swap, 'psi_p1': _psi_swap_p1, 'psi_m1': _psi_swap_m1}

def _p_n_e6(_n, _G, _w_list, psi_sym, n_sym):
    '''
    Computing psi(x + 2), psi(x - 2)
    '''
    _swap_e4 = _p_n_e4(_n, _G, _w_list, psi_sym, n_sym)
    _psi_swap_m2 = np.append(_swap_e4['psi_m1'][-1], _swap_e4['psi_m1'][:-1])
    _psi_swap_p2 = np.append(_swap_e4['psi_p1'][1:], _swap_e4['psi_p1'][0])    
    '''
    Computing normal component
    '''
    _swap_e4['p_n'] += \
        _G * _w_list[2] * _swap_e4['psi'] * (_psi_swap_p2 + _psi_swap_m2) + \
        2 * _G * _w_list[2] * _swap_e4['psi_p1'] * _swap_e4['psi_m1']
    
    return {**_swap_e4, 'psi_p2': _psi_swap_p2, 'psi_m2': _psi_swap_m2}

def _p_n_e8(_n, _G, _w_list, psi_sym, n_sym):
    '''
    Computing psi(x), psi(x + 1), psi(x - 1)
    '''
    _swap_e6 = _p_n_e6(_n, _G, _w_list, psi_sym, n_sym)
    '''
    Computing normal component
    '''
    _swap_e6['p_n'] += \
        _G * _w_list[3] * _swap_e6['psi'] * (_swap_e6['psi_p1'] + _swap_e6['psi_m1']) + \
        2 * _G * (_w_list[3] + _w_list[4]) * _swap_e6['psi'] * (_swap_e6['psi_p2'] + _swap_e6['psi_m2']) + \
        4 * _G * (_w_list[3] + _w_list[4]) * _swap_e6['psi_p1'] * _swap_e6['psi_m1']
    
    return {**_swap_e6}

def _p_n_e10T(_n, _G, _w_list, psi_sym, n_sym):
    '''
    Computing psi(x), psi(x + 1), psi(x - 1)
    '''
    _swap_e8 = _p_n_e8(_n, _G, _w_list, psi_sym, n_sym)
    _psi_swap_m3 = np.append(_swap_e8['psi_m2'][-1], _swap_e8['psi_m2'][:-1])
    _psi_swap_p3 = np.append(_swap_e8['psi_p2'][1:], _swap_e8['psi_p2'][0])    
    
    '''
    Computing normal component
    '''
    _swap_e8['p_n'] += \
        _G * (3 * _w_list[5] / 2 + 3 * _w_list[6]) * _swap_e8['psi'] * (_psi_swap_p3 + _psi_swap_m3) + \
        _G * (3 * _w_list[5] + 6 * _w_list[6]) * (_swap_e8['psi_p1'] * _swap_e8['psi_m2'] + _swap_e8['psi_m1'] * _swap_e8['psi_p2'])
    
    return {**_swap_e8, 'psi_p3': _psi_swap_p3, 'psi_m3': _psi_swap_m3}

"""
Two-dimensional lattice pressure tensors
"""

def VectorRoll(field, xi):
    rolled_field = np.copy(field)
    """
    Here, I need to flio the vector in order to used the coordinate index as the axis
    Cartesian directions and np array dimensions are in flipped relative order
    """
    for axis, c in enumerate(np.flip(xi)):
        rolled_field = np.roll(rolled_field, shift=-c, axis=axis)
    return rolled_field

def LPT_12_2D(G, n_field, f_stencil, psi_sym, n_sym):
    dim_sizes = n_field.shape
    d = len(dim_sizes)

    Nc2 = d * (d + 1) // 2

    LPT = np.zeros(list(dim_sizes) + [Nc2])
    tt = TaylorTuples(tuple(range(d)), 2)

    psi_f = sp.lambdify(n_sym, psi_sym)

    psi_field = psi_f(n_field).astype(np.float64)

    Ws, Es = f_stencil['Ws'], f_stencil['Es']
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 1 or l2 == 2:
            for i, t in enumerate(tt):
                LPT[:, :, i] += float(Ws[e_i]) * VectorRoll(psi_field, e) * e[t[0]] * e[t[1]]

    for i in range(Nc2):
        LPT[:, :, i] *= psi_field * G / 2
        if tt[i][0] == tt[i][1]:
            print(tt[i])
            LPT[:, :, i] += n_field / 3

    return {'LPT': LPT, 'psi_field': psi_field}

def LPT_48_2D(G, n_field, f_stencil, psi_sym, n_sym):
    LPT_12 = LPT_12_2D(G, n_field, f_stencil, psi_sym, n_sym)
    psi_field = LPT_12['psi_field']

    dim_sizes = n_field.shape
    d = len(dim_sizes)

    Nc2 = d * (d + 1) // 2

    LPT = np.zeros(list(dim_sizes) + [Nc2])
    tt = TaylorTuples(tuple(range(d)), 2)

    Ws, Es = f_stencil['Ws'], f_stencil['Es']
    """
    First term
    """
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 4 or l2 == 8:
            for i, t in enumerate(tt):
                LPT[:, :, i] += float(Ws[e_i]) * VectorRoll(psi_field, e) * e[t[0]] * e[t[1]]

    for i in range(Nc2):
        LPT[:, :, i] *= psi_field * G / 4

    """
    Second term
    """
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 4 or l2 == 8:
            for i, t in enumerate(tt):
                LPT[:, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, np.array(e) // 2) * VectorRoll(psi_field, -np.array(e) // 2) * e[t[0]] * e[t[1]]

    return {'LPT': LPT + LPT_12['LPT'], 'psi_field': psi_field}

def LPT_5_2D(G, n_field, f_stencil, psi_sym, n_sym):
    LPT_48 = LPT_48_2D(G, n_field, f_stencil, psi_sym, n_sym)
    psi_field = LPT_48['psi_field']

    dim_sizes = n_field.shape
    d = len(dim_sizes)

    Nc2 = d * (d + 1) // 2

    LPT = np.zeros(list(dim_sizes) + [Nc2])
    tt = TaylorTuples(tuple(range(d)), 2)

    Ws, Es = f_stencil['Ws'], f_stencil['Es']
    """
    A term
    """
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 5:
            for i, t in enumerate(tt):
                LPT[:, :, i] += float(Ws[e_i]) * VectorRoll(psi_field, e) * e[t[0]] * e[t[1]]

    for i in range(Nc2):
        LPT[:, :, i] *= psi_field * G / 4

    """
    B term
    """
    be = [np.array((1, 0)), np.array((0, 1))]
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 5:
            for i, t in enumerate(tt):
                if e[0] == 2 and e[1] == 1:
                    LPT[:, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0]) * VectorRoll(psi_field, be[0] + be[1]) * e[t[0]] * e[t[1]]
                    LPT[:, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] - be[1]) * VectorRoll(psi_field, be[0]) * e[t[0]] * e[t[1]]

                if e[0] == 1 and e[1] == 2:
                    LPT[:, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] - be[1]) * VectorRoll(psi_field, be[1]) * e[t[0]] * e[t[1]]
                    LPT[:, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] + be[1]) * VectorRoll(psi_field, -be[1]) * e[t[0]] * e[t[1]]

                if e[0] == -1 and e[1] == 2:
                    LPT[:, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] + be[1]) * VectorRoll(psi_field, -be[1]) * e[t[0]] * e[t[1]]
                    LPT[:, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] - be[1]) * VectorRoll(psi_field, be[1]) * e[t[0]] * e[t[1]]

                if e[0] == -2 and e[1] == 1:
                    LPT[:, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] - be[1]) * VectorRoll(psi_field, -be[0]) * e[t[0]] * e[t[1]]
                    LPT[:, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] + be[1]) * VectorRoll(psi_field, be[0]) * e[t[0]] * e[t[1]]

    return {'LPT': LPT + LPT_48['LPT'], 'psi_field': psi_field}


def LPT_9_18_2D(G, n_field, f_stencil, psi_sym, n_sym):
    LPT_5 = LPT_5_2D(G, n_field, f_stencil, psi_sym, n_sym)
    psi_field = LPT_5['psi_field']

    dim_sizes = n_field.shape
    d = len(dim_sizes)

    Nc2 = d * (d + 1) // 2

    LPT = np.zeros(list(dim_sizes) + [Nc2])
    tt = TaylorTuples(tuple(range(d)), 2)

    Ws, Es = f_stencil['Ws'], f_stencil['Es']
    """
    First term
    """
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 9 or l2 == 18:
            for i, t in enumerate(tt):
                LPT[:, :, i] += float(Ws[e_i]) * VectorRoll(psi_field, e) * e[t[0]] * e[t[1]]

    for i in range(Nc2):
        LPT[:, :, i] *= psi_field * G / 6

    """
    Second term
    """
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 9 or l2 == 18:
            for i, t in enumerate(tt):
                LPT[:, :, i] += (G / 3) * float(Ws[e_i]) * VectorRoll(psi_field, np.array(e) // 3) * VectorRoll(psi_field, -2 * np.array(e) // 3) * e[t[0]] * e[t[1]]

    return {'LPT': LPT + LPT_5['LPT'], 'psi_field': psi_field}

def LPT_16_2D(G, n_field, f_stencil, psi_sym, n_sym):
    LPT_9_18 = LPT_9_18_2D(G, n_field, f_stencil, psi_sym, n_sym)
    psi_field = LPT_9_18['psi_field']

    dim_sizes = n_field.shape
    d = len(dim_sizes)

    Nc2 = d * (d + 1) // 2

    LPT = np.zeros(list(dim_sizes) + [Nc2])
    tt = TaylorTuples(tuple(range(d)), 2)

    Ws, Es = f_stencil['Ws'], f_stencil['Es']
    """
    First term
    """
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 16:
            for i, t in enumerate(tt):
                LPT[:, :, i] += float(Ws[e_i]) * VectorRoll(psi_field, e) * e[t[0]] * e[t[1]]

    for i in range(Nc2):
        LPT[:, :, i] *= psi_field * G / 8

    """
    Second term
    """
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 16:
            for i, t in enumerate(tt):
                LPT[:, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, np.array(e) // 4) * VectorRoll(psi_field, -3 * np.array(e) // 4) * e[t[0]] * e[t[1]]

    """
    Third term
    """
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 16:
            for i, t in enumerate(tt):
                LPT[:, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, np.array(e) // 2) * VectorRoll(psi_field, -np.array(e) // 2) * e[t[0]] * e[t[1]]

    return {'LPT': LPT + LPT_9_18['LPT'], 'psi_field': psi_field}


"""
Three-dimensional lattice pressure tensors
"""

def LPT_123_3D(G, n_field, f_stencil, psi_sym, n_sym):
    dim_sizes = n_field.shape
    d = len(dim_sizes)

    Nc2 = d * (d + 1) // 2

    LPT = np.zeros(list(dim_sizes) + [Nc2])
    tt = TaylorTuples(tuple(range(d)), 2)

    psi_f = sp.lambdify(n_sym, psi_sym)

    psi_field = psi_f(n_field).astype(np.float64)

    Ws, Es = f_stencil['Ws'], f_stencil['XIs']
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 1 or l2 == 2 or l2 == 3:
            for i, t in enumerate(tt):
                LPT[:, :, :, i] += float(Ws[e_i]) * VectorRoll(psi_field, e) * e[t[0]] * e[t[1]]

    for i in range(Nc2):
        LPT[:, :, :, i] *= psi_field * G / 2
        if tt[i][0] == tt[i][1]:
            print(tt[i])
            LPT[:, :, :, i] += n_field / 3

    return {'LPT': LPT, 'psi_field': psi_field}

def LPT_48_3D(G, n_field, f_stencil, psi_sym, n_sym):
    LPT_123 = LPT_123_3D(G, n_field, f_stencil, psi_sym, n_sym)
    psi_field = LPT_123['psi_field']

    dim_sizes = n_field.shape
    d = len(dim_sizes)

    Nc2 = d * (d + 1) // 2

    LPT = np.zeros(list(dim_sizes) + [Nc2])
    tt = TaylorTuples(tuple(range(d)), 2)

    Ws, Es = f_stencil['Ws'], f_stencil['XIs']
    """
    First term
    """
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 4 or l2 == 8:
            for i, t in enumerate(tt):
                LPT[:, :, :, i] += float(Ws[e_i]) * VectorRoll(psi_field, e) * e[t[0]] * e[t[1]]

    for i in range(Nc2):
        LPT[:, :, :, i] *= psi_field * G / 4

    """
    Second term
    """
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 4 or l2 == 8:
            for i, t in enumerate(tt):
                LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, np.array(e) // 2) * VectorRoll(psi_field, -np.array(e) // 2) * e[t[0]] * e[t[1]]

    return {'LPT': LPT + LPT_123['LPT'], 'psi_field': psi_field}

def LPT_5_3D(G, n_field, f_stencil, psi_sym, n_sym):
    LPT_48 = LPT_48_3D(G, n_field, f_stencil, psi_sym, n_sym)
    psi_field = LPT_48['psi_field']

    dim_sizes = n_field.shape
    d = len(dim_sizes)

    Nc2 = d * (d + 1) // 2

    LPT = np.zeros(list(dim_sizes) + [Nc2])
    tt = TaylorTuples(tuple(range(d)), 2)

    Ws, Es = f_stencil['Ws'], f_stencil['XIs']
    """
    A term
    """
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 5:
            for i, t in enumerate(tt):
                LPT[:, :, :, i] += float(Ws[e_i]) * VectorRoll(psi_field, e) * e[t[0]] * e[t[1]]

    for i in range(Nc2):
        LPT[:, :, :, i] *= psi_field * G / 4

    """
    B term
    """
    be = [np.array((1, 0, 0)), np.array((0, 1, 0)), np.array((0, 0, 1))]
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 5:
            for i, t in enumerate(tt):
                ################### XY

                if e[0] == 2 and e[1] == 1 and e[2] == 0:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0]) * VectorRoll(psi_field, be[0] + be[1]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] - be[1]) * VectorRoll(psi_field, be[0]) * e[t[0]] * e[t[1]]

                if e[0] == 1 and e[1] == 2 and e[2] == 0:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] - be[1]) * VectorRoll(psi_field, be[1]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] + be[1]) * VectorRoll(psi_field, -be[1]) * e[t[0]] * e[t[1]]

                if e[0] == -1 and e[1] == 2 and e[2] == 0:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] + be[1]) * VectorRoll(psi_field, -be[1]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] - be[1]) * VectorRoll(psi_field, be[1]) * e[t[0]] * e[t[1]]

                if e[0] == -2 and e[1] == 1 and e[2] == 0:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] - be[1]) * VectorRoll(psi_field, -be[0]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] + be[1]) * VectorRoll(psi_field, be[0]) * e[t[0]] * e[t[1]]

                #################### YZ

                if e[0] == 0 and e[1] == 2 and e[2] == 1:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[1]) * VectorRoll(psi_field, be[1] + be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[1] - be[2]) * VectorRoll(psi_field, be[1]) * e[t[0]] * e[t[1]]

                if e[0] == 0 and e[1] == 1 and e[2] == 2:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[1] - be[2]) * VectorRoll(psi_field, be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[1] + be[2]) * VectorRoll(psi_field, -be[2]) * e[t[0]] * e[t[1]]

                if e[0] == 0 and e[1] == -1 and e[2] == 2:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[1] + be[2]) * VectorRoll(psi_field, -be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[1] - be[2]) * VectorRoll(psi_field, be[2]) * e[t[0]] * e[t[1]]

                if e[0] == 0 and e[1] == -2 and e[2] == 1:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[1] - be[2]) * VectorRoll(psi_field, -be[1]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[1] + be[2]) * VectorRoll(psi_field, be[1]) * e[t[0]] * e[t[1]]

                ##################### ZX

                if e[0] == 1 and e[1] == 0 and e[2] == 2:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[2]) * VectorRoll(psi_field, be[2] + be[0]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[2] - be[0]) * VectorRoll(psi_field, be[2]) * e[t[0]] * e[t[1]]

                if e[0] == 2 and e[1] == 0 and e[2] == 1:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[2] - be[0]) * VectorRoll(psi_field, be[0]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[2] + be[0]) * VectorRoll(psi_field, -be[0]) * e[t[0]] * e[t[1]]

                if e[0] == 2 and e[1] == 0 and e[2] == -1:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[2] + be[0]) * VectorRoll(psi_field, -be[0]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[2] - be[0]) * VectorRoll(psi_field, be[0]) * e[t[0]] * e[t[1]]

                if e[0] == 1 and e[1] == 0 and e[2] == -2:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[2] - be[0]) * VectorRoll(psi_field, -be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[2] + be[0]) * VectorRoll(psi_field, be[2]) * e[t[0]] * e[t[1]]

    return {'LPT': LPT + LPT_48['LPT'], 'psi_field': psi_field}

def LPT_6_3D(G, n_field, f_stencil, psi_sym, n_sym):
    LPT_5 = LPT_5_3D(G, n_field, f_stencil, psi_sym, n_sym)
    psi_field = LPT_5['psi_field']

    dim_sizes = n_field.shape
    d = len(dim_sizes)

    Nc2 = d * (d + 1) // 2

    LPT = np.zeros(list(dim_sizes) + [Nc2])
    tt = TaylorTuples(tuple(range(d)), 2)

    Ws, Es = f_stencil['Ws'], f_stencil['XIs']
    """
    A term
    """
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 6:
            for i, t in enumerate(tt):
                LPT[:, :, :, i] += float(Ws[e_i]) * VectorRoll(psi_field, e) * e[t[0]] * e[t[1]]

    for i in range(Nc2):
        LPT[:, :, :, i] *= psi_field * G / 4

    """
    B term
    """
    be = [np.array((1, 0, 0)), np.array((0, 1, 0)), np.array((0, 0, 1))]
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 6:
            for i, t in enumerate(tt):
                ################### XY Sign/Permutations of (2, 1, 1)

                if e[0] == 2 and e[1] == 1 and e[2] == 1:
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0]) * VectorRoll(psi_field, be[0] + be[1] + be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[0]) * VectorRoll(psi_field, -be[0] - be[1] - be[2]) * e[t[0]] * e[t[1]]

                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] - be[2]) * VectorRoll(psi_field, be[0] + be[1]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] + be[2]) * VectorRoll(psi_field, -be[0] - be[1]) * e[t[0]] * e[t[1]]

                if e[0] == 1 and e[1] == 2 and e[2] == 1:
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] - be[1]) * VectorRoll(psi_field, be[1] + be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] + be[1]) * VectorRoll(psi_field, -be[1] - be[2]) * e[t[0]] * e[t[1]]

                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] - be[1] - be[2]) * VectorRoll(psi_field, be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] + be[1] + be[2]) * VectorRoll(psi_field, -be[2]) * e[t[0]] * e[t[1]]

                if e[0] == -1 and e[1] == 2 and e[2] == 1:
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[1]) * VectorRoll(psi_field, -be[0] + be[1] + be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[1]) * VectorRoll(psi_field, be[0] - be[1] - be[2]) * e[t[0]] * e[t[1]]

                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[1] - be[2]) * VectorRoll(psi_field, -be[0] + be[1]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[1] + be[2]) * VectorRoll(psi_field, be[0] - be[1]) * e[t[0]] * e[t[1]]

                if e[0] == -2 and e[1] == 1 and e[2] == 1:
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] - be[1]) * VectorRoll(psi_field, -be[0] + be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] + be[1]) * VectorRoll(psi_field, be[0] - be[2]) * e[t[0]] * e[t[1]]

                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] - be[1] - be[2]) * VectorRoll(psi_field, -be[0]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] + be[1] + be[2]) * VectorRoll(psi_field, be[0]) * e[t[0]] * e[t[1]]

                ################### XY Sign/Permutations of (2, 1, -1)

                if e[0] == 2 and e[1] == 1 and e[2] == -1:
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] - be[1] + be[2]) * VectorRoll(psi_field, be[0]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] + be[1] - be[2]) * VectorRoll(psi_field, -be[0]) * e[t[0]] * e[t[1]]

                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] - be[1]) * VectorRoll(psi_field, be[0] - be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] + be[1]) * VectorRoll(psi_field, -be[0] + be[2]) * e[t[0]] * e[t[1]]

                if e[0] == 1 and e[1] == 2 and e[2] == -1:
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] - be[1] + be[2]) * VectorRoll(psi_field, be[1]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] + be[1] - be[2]) * VectorRoll(psi_field, -be[1]) * e[t[0]] * e[t[1]]

                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] - be[1]) * VectorRoll(psi_field, be[1] - be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] + be[1]) * VectorRoll(psi_field, -be[1] + be[2]) * e[t[0]] * e[t[1]]

                if e[0] == -1 and e[1] == 2 and e[2] == -1:
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[1] + be[2]) * VectorRoll(psi_field, -be[0] + be[1]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[1] - be[2]) * VectorRoll(psi_field, be[0] - be[1]) * e[t[0]] * e[t[1]]

                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[1]) * VectorRoll(psi_field, -be[0] + be[1] - be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[1]) * VectorRoll(psi_field, be[0] - be[1] + be[2]) * e[t[0]] * e[t[1]]

                if e[0] == -2 and e[1] == 1 and e[2] == -1:
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] - be[1] + be[2]) * VectorRoll(psi_field, -be[0]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] + be[1] - be[2]) * VectorRoll(psi_field, be[0]) * e[t[0]] * e[t[1]]

                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] - be[1]) * VectorRoll(psi_field, -be[0] - be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] + be[1]) * VectorRoll(psi_field, be[0] + be[2]) * e[t[0]] * e[t[1]]

                ################### XZ Sign/Permutations of (+-1, 1, +-2)

                if e[0] == 1 and e[1] == 1 and e[2] == 2:
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] - be[1] - be[2]) * VectorRoll(psi_field, be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] + be[1] + be[2]) * VectorRoll(psi_field, -be[2]) * e[t[0]] * e[t[1]]

                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[1] - be[2]) * VectorRoll(psi_field, be[0] + be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[1] + be[2]) * VectorRoll(psi_field, -be[0] - be[2]) * e[t[0]] * e[t[1]]

                if e[0] == -1 and e[1] == 1 and e[2] == 2:
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[1] - be[2]) * VectorRoll(psi_field, -be[0] + be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[1] + be[2]) * VectorRoll(psi_field, be[0] - be[2]) * e[t[0]] * e[t[1]]

                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] - be[1] - be[2]) * VectorRoll(psi_field, be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] + be[1] + be[2]) * VectorRoll(psi_field, -be[2]) * e[t[0]] * e[t[1]]

                if e[0] == 1 and e[1] == 1 and e[2] == -2:
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] - be[1] + be[2]) * VectorRoll(psi_field, -be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] + be[1] - be[2]) * VectorRoll(psi_field, be[2]) * e[t[0]] * e[t[1]]

                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[1] + be[2]) * VectorRoll(psi_field, be[0] - be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[1] - be[2]) * VectorRoll(psi_field, -be[0] + be[2]) * e[t[0]] * e[t[1]]

                if e[0] == -1 and e[1] == 1 and e[2] == -2:
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[1] + be[2]) * VectorRoll(psi_field, -be[0] - be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[1] - be[2]) * VectorRoll(psi_field, be[0] + be[2]) * e[t[0]] * e[t[1]]

                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] - be[1] + be[2]) * VectorRoll(psi_field, -be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] + be[1] - be[2]) * VectorRoll(psi_field, be[2]) * e[t[0]] * e[t[1]]

    return {'LPT': LPT + LPT_5['LPT'], 'psi_field': psi_field}


def LPT_9_221_3D(G, n_field, f_stencil, psi_sym, n_sym):
    LPT_6 = LPT_6_3D(G, n_field, f_stencil, psi_sym, n_sym)
    psi_field = LPT_6['psi_field']

    dim_sizes = n_field.shape
    d = len(dim_sizes)

    Nc2 = d * (d + 1) // 2

    LPT = np.zeros(list(dim_sizes) + [Nc2])
    tt = TaylorTuples(tuple(range(d)), 2)

    Ws, Es = f_stencil['Ws'], f_stencil['XIs']
    """
    A term
    """
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 9 and e[0] != 0 and e[1] != 0 and e[2] != 0:
            for i, t in enumerate(tt):
                LPT[:, :, :, i] += float(Ws[e_i]) * VectorRoll(psi_field, e) * e[t[0]] * e[t[1]]

    for i in range(Nc2):
        LPT[:, :, :, i] *= psi_field * G / 4

    """
    B term
    """
    be = [np.array((1, 0, 0)), np.array((0, 1, 0)), np.array((0, 0, 1))]
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 9 and e[0] != 0 and e[1] != 0 and e[2] != 0:
            for i, t in enumerate(tt):
                ## First four
                if e[0] == 2 and e[1] == 2 and e[2] == 1:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] - be[1] - be[2]) * VectorRoll(psi_field, be[0] + be[1]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] + be[1] + be[2]) * VectorRoll(psi_field, -be[0] - be[1]) * e[t[0]] * e[t[1]]

                if e[0] == -2 and e[1] == 2 and e[2] == 1:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] - be[1] - be[2]) * VectorRoll(psi_field, -be[0] + be[1]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] + be[1] + be[2]) * VectorRoll(psi_field, be[0] - be[1]) * e[t[0]] * e[t[1]]

                if e[0] == 2 and e[1] == 2 and e[2] == -1:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] - be[1]) * VectorRoll(psi_field, be[0] + be[1] - be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] + be[1]) * VectorRoll(psi_field, -be[0] - be[1] + be[2]) * e[t[0]] * e[t[1]]

                if e[0] == -2 and e[1] == 2 and e[2] == -1:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] - be[1]) * VectorRoll(psi_field, -be[0] + be[1] - be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] + be[1]) * VectorRoll(psi_field, be[0] - be[1] + be[2]) * e[t[0]] * e[t[1]]

                ## Second four Z > 0
                if e[0] == 2 and e[1] == 1 and e[2] == 2:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] - be[1] - be[2]) * VectorRoll(psi_field, be[0] + be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] + be[1] + be[2]) * VectorRoll(psi_field, -be[0] - be[2]) * e[t[0]] * e[t[1]]

                if e[0] == 1 and e[1] == 2 and e[2] == 2:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] - be[1] - be[2]) * VectorRoll(psi_field, be[1] + be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] + be[1] + be[2]) * VectorRoll(psi_field, -be[1] - be[2]) * e[t[0]] * e[t[1]]

                if e[0] == -1 and e[1] == 2 and e[2] == 2:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[1] - be[2]) * VectorRoll(psi_field, -be[0] + be[1] + be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[1] + be[2]) * VectorRoll(psi_field, be[0] - be[1] - be[2]) * e[t[0]] * e[t[1]]

                if e[0] == -2 and e[1] == 1 and e[2] == 2:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] - be[1] - be[2]) * VectorRoll(psi_field, -be[0] + be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] + be[1] + be[2]) * VectorRoll(psi_field, be[0] - be[2]) * e[t[0]] * e[t[1]]

                ## Second four Z < 0
                if e[0] == 2 and e[1] == 1 and e[2] == -2:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] - be[1] + be[2]) * VectorRoll(psi_field, be[0] - be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] + be[1] - be[2]) * VectorRoll(psi_field, -be[0] + be[2]) * e[t[0]] * e[t[1]]

                if e[0] == 1 and e[1] == 2 and e[2] == -2:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] - be[1] + be[2]) * VectorRoll(psi_field, be[1] - be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] + be[1] - be[2]) * VectorRoll(psi_field, -be[1] + be[2]) * e[t[0]] * e[t[1]]

                if e[0] == -1 and e[1] == 2 and e[2] == -2:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[1] + be[2]) * VectorRoll(psi_field, -be[0] + be[1] - be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[1] - be[2]) * VectorRoll(psi_field, be[0] - be[1] + be[2]) * e[t[0]] * e[t[1]]

                if e[0] == -2 and e[1] == 1 and e[2] == -2:
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, be[0] - be[1] + be[2]) * VectorRoll(psi_field, -be[0] - be[2]) * e[t[0]] * e[t[1]]
                    LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, -be[0] + be[1] - be[2]) * VectorRoll(psi_field, be[0] + be[2]) * e[t[0]] * e[t[1]]

    return {'LPT': LPT + LPT_6['LPT'], 'psi_field': psi_field}



def LPT_9_18_3D(G, n_field, f_stencil, psi_sym, n_sym):
    LPT_9_221 = LPT_9_221_3D(G, n_field, f_stencil, psi_sym, n_sym)
    psi_field = LPT_9_221['psi_field']

    dim_sizes = n_field.shape
    d = len(dim_sizes)

    Nc2 = d * (d + 1) // 2

    LPT = np.zeros(list(dim_sizes) + [Nc2])
    tt = TaylorTuples(tuple(range(d)), 2)

    Ws, Es = f_stencil['Ws'], f_stencil['XIs']
    """
    First term
    """
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if (l2 == 9 or l2 == 18) and (e[0] == 0 or e[1] == 0 or e[2] == 0):
            for i, t in enumerate(tt):
                LPT[:, :, :, i] += float(Ws[e_i]) * VectorRoll(psi_field, e) * e[t[0]] * e[t[1]]

    for i in range(Nc2):
        LPT[:, :, :, i] *= psi_field * G / 6

    """
    Second term
    """
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if (l2 == 9 or l2 == 18) and (e[0] == 0 or e[1] == 0 or e[2] == 0):
            for i, t in enumerate(tt):
                LPT[:, :, :, i] += (G / 3) * float(Ws[e_i]) * VectorRoll(psi_field, np.array(e) // 3) * VectorRoll(psi_field, -2 * np.array(e) // 3) * e[t[0]] * e[t[1]]

    return {'LPT': LPT + LPT_9_221['LPT'], 'psi_field': psi_field}

def LPT_16_3D(G, n_field, f_stencil, psi_sym, n_sym):
    LPT_9_18 = LPT_9_18_3D(G, n_field, f_stencil, psi_sym, n_sym)
    psi_field = LPT_9_18['psi_field']

    dim_sizes = n_field.shape
    d = len(dim_sizes)

    Nc2 = d * (d + 1) // 2

    LPT = np.zeros(list(dim_sizes) + [Nc2])
    tt = TaylorTuples(tuple(range(d)), 2)

    Ws, Es = f_stencil['Ws'], f_stencil['XIs']
    """
    First term
    """
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 16:
            for i, t in enumerate(tt):
                LPT[:, :, :, i] += float(Ws[e_i]) * VectorRoll(psi_field, e) * e[t[0]] * e[t[1]]

    for i in range(Nc2):
        LPT[:, :, :, i] *= psi_field * G / 8

    """
    Second term
    """
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 16:
            for i, t in enumerate(tt):
                LPT[:, :, :, i] += (G / 4) * float(Ws[e_i]) * VectorRoll(psi_field, np.array(e) // 4) * VectorRoll(psi_field, -3 * np.array(e) // 4) * e[t[0]] * e[t[1]]

    """
    Third term
    """
    for e_i, e in enumerate(Es):
        l2 = np.sum(np.array(e) ** 2)
        if l2 == 16:
            for i, t in enumerate(tt):
                LPT[:, :, :, i] += (G / 8) * float(Ws[e_i]) * VectorRoll(psi_field, np.array(e) // 2) * VectorRoll(psi_field, -np.array(e) // 2) * e[t[0]] * e[t[1]]

    return {'LPT': LPT + LPT_9_18['LPT'], 'psi_field': psi_field}
