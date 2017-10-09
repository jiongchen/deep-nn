#!/usr/bin/python

from __future__ import print_function
import hashlib
import numpy as np


def read_dense_matrix(path):
    """
    int64_t row, col, 8 bytes
    double  data, 8 bytes
    """
    try:
        f = open(path, "r")
        row = np.fromfile(f, np.dtype(np.int64), count=1)
        col = np.fromfile(f, np.dtype(np.int64), count=1)
        mat = np.fromfile(f, np.dtype(np.float64), count=row*col)
    except IOError:
        print('[WARN] can not open %s' % path)
    finally:
        if f is not None:
            f.close()
    return row[0], col[0], np.reshape(mat, (row[0], col[0]), 'F')

def write_dense_matrix(path, mat):
    """
    mat is an np array
    """
    row, col = mat.shape
    flat_mat = mat.flatten('F')
    try:
        f = open(path, "w")
        np.array([row], dtype=np.int64).tofile(f)
        np.array([col], dtype=np.int64).tofile(f)
        np.array(flat_mat, dtype=np.float64).tofile(f)
    except IOError:
        print('[WARN] can not open %s' % path)
    finally:
        if f is not None:
            f.close()
            

if __name__ == '__main__':
    def func_to_regress(arr):
        return arr[0]*arr[1]+np.sin(arr[2])*np.exp(arr[3])+arr[4]**4

    def gen_data_sample(sample_num):
        x_data = np.random.rand(5, sample_num)
        write_dense_matrix('./train.dat', x_data)

        y = np.apply_along_axis(func_to_regress, 0, x_data)
        y_data = y.reshape((1, y.size))
        write_dense_matrix('./label.dat', y_data)

    gen_data_sample(sample_num=10000)
    
    _, _, mat = read_dense_matrix('./train.dat')
    write_dense_matrix('./dump.dat', mat)
    
    orig_hash = hashlib.md5(open('./train.dat','rb').read()).hexdigest()
    dump_hash = hashlib.md5(open('./dump.dat', 'rb').read()).hexdigest()

    print(orig_hash, dump_hash)
    
