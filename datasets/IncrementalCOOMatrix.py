#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
@author: Ervin Dervishaj
@email: vindervishaj@gmail.com
'''

import array
import numpy as np
import scipy.sparse as sps
from operator import itemgetter

class IncrementalCOOMatrix(object):
    '''
    Implementation from
    https://maciejkula.github.io/2015/02/22/incremental-construction-of-sparse-matrices/
    '''

    def __init__(self, data_dtype=np.float32, idx_dtype=np.uint32):
        '''
        Constructor
        :param data_dtype: datatype for the data of the matrix
        :param idx_dtype: datatype for the indices of elements of matrix
        :param shape: shape of the matrix
        '''

        if data_dtype is np.int32:
            self.d_type_flag = 'i'
        elif data_dtype is np.int64:
            self.d_type_flag = 'l'
        elif data_dtype is np.float32:
            self.d_type_flag = 'f'
        elif data_dtype is np.float64:
            self.d_type_flag = 'd'
        else:
            raise Exception('Dtype for data not supported.')

        if idx_dtype is np.uint32:
            self.i_type_flag = 'I'
        elif data_dtype is np.uint64:
            self.i_type_flag = 'L'
        else:
            raise Exception('Dtype for indices not supported.')

        self.idx_dtype = idx_dtype
        self.data_dtype = data_dtype

        self.rows = array.array(self.i_type_flag)
        self.cols = array.array(self.i_type_flag)
        self.data = array.array(self.d_type_flag)


    def append(self, i, j, v):
        '''
        Appends data to the matrix: A[rows(i), cols(j)] = data(v)
        :param i: row index
        :param j: col index
        :param v: data
        '''

        self.rows.append(i)
        self.cols.append(j)
        self.data.append(v)

    def extend(self, i, j, v):
        '''
        Appends data to the matrix: A[rows(i), cols(j)] = data(v)
        :param i: row index
        :param j: col index
        :param v: data
        '''

        self.rows.extend(i)
        self.cols.extend(j)
        self.data.extend(v)


    def tocoo(self):
        '''
        Replaces indices in rows and cols arrays with indices of the matrix
        and then creates the COO matrix
        :return: constructed COO matrix
        '''

        # Change datatype of indices if bigger than default
        if self.shape[0] > np.iinfo(np.uint32).max or self.shape[1] > np.iinfo(np.uint32).max:
            self.idx_dtype = np.uint64

        coo_rows = array.array(self.i_type_flag)
        coo_cols = array.array(self.i_type_flag)
        coo_data = array.array(self.d_type_flag)

        coo_rows.extend([self.row_to_user[key] for key in self.rows])
        coo_cols.extend([self.col_to_item[key] for key in self.cols])
        coo_data.extend(self.data)

        rows = np.frombuffer(coo_rows, dtype=self.idx_dtype)
        cols = np.frombuffer(coo_cols, dtype=self.idx_dtype)
        data = np.frombuffer(coo_data, dtype=self.data_dtype)

        self.coo_matrix = sps.coo_matrix((data, (rows, cols)), shape=self.shape, dtype=self.data_dtype)

        # Delete arrays to save space
        self.rows, self.cols, self.data, self.row_to_user, self.col_to_item = (None,) * 5

        return self.coo_matrix


    def set_shape(self, shape):
        self.shape = shape

    def set_mappers(self, row_to_user, col_to_item):
        self.row_to_user = row_to_user
        self.col_to_item = col_to_item