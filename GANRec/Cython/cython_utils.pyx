"""
@author Ervin Dervishaj
"""

import cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def get_non_interactions(int[:] all_users, URM_train):
    not_selected = {}
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t max_users = all_users.shape[0]
    cdef int u
    for i in range(max_users):
        u = all_users[i]
        not_selected[u] = np.nonzero(URM_train[u].toarray()[0] == 0)[0]
    return not_selected

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
def compute_masks(int[:] all_users, dict not_selected, str scheme, double zr_ratio, double zp_ratio):
    zr_sample_indices = {}
    pm_sample_indices = {}
    cdef Py_ssize_t i = 0
    cdef int u
    cdef Py_ssize_t max_users = all_users.shape[0]
    for i in range(max_users):
        u = all_users[i]
        if scheme == 'ZP' or scheme == 'ZR':
            selected = np.random.choice(not_selected[u], size=int(len(not_selected[u]) * zr_ratio),
                                        replace=False)
            zr_sample_indices[u] = selected

        if scheme == 'ZP' or scheme == 'PM':
            selected = np.random.choice(not_selected[u], size=int(len(not_selected[u]) * zr_ratio),
                                        replace=False)
            pm_sample_indices[u] = selected

    return zr_sample_indices, pm_sample_indices
