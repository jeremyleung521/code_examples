#!/usr/bin/env python
import numpy

def construct_auxdata1(n_iter, iter_group):
    dim1 = iter_group['auxdata/phi']
    dim2 = iter_group['auxdata/psi']
    dataset = numpy.dstack((dim1, dim2))
    #dataset = numpy.expand_dims(dim1,axis=2)
    return dataset
