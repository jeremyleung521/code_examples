#!/usr/bin/env python
#
# drms.py
#
# A suite of functions to confirm/verify mean structure hypothesis using distance RMS metric (different from normal RMSD).
#
# Includes code that:
#     - calculate the mean distance matrix (i.e. a distance matrix with each entry already weighted-averaged from individual "trial" dist matrices)
#     - calculate the distance RMS given two (sets of) matrices
#     - some verification function that makes sure the dist matrix is a valid one
#
#
#

import numpy
import os
from tqdm.auto import tqdm, trange
import pickle
import gc
import argparse

## Variables
variants = ['wsh2045', 'wsh2029','wsh2036','wsh2044','wsh2054'] # list of all the variants to loop over, also used to access those files.
ref_var = [ 'dist_matrix', 'dist_matrix_avg', 'dist_matrix_avg_helix' ] # list of file names for reference structure

rfc_path = "/home/jml230/BdpA Publishable/rfc" # should be pointed to your current directory.
mean_matrix_template = '/home/jml230/BdpA Publishable/rfc/{var}_eng/mean_matrix.npy' # points to the mean_matrix file, assumed saved as a numpy array file.
dist_matrix_template = '/home/jml230/BdpA Publishable/avg_struct/{var}/{ref}.dat' # points to where the reference files, assumed saved as a txt file.

weights_template = '/home/jml230/bdpa_{var}_p2_haMSM/restart0_scikit_nowat/run1/{var}_final_reweighted_weights.dat' # points to where the final weights are, a text file
matrix_template = '/home/jml230/BdpA Publishable/rfc/{var}/dist_matrix_{file_num}.npy' # points to where the dist matrix arrays are. Assume numpy array files.
eligible_states_template = '/home/jml230/BdpA Publishable/new_dist_with_state_def/{var}/eligible_states_eng_all.dat' # points to where the eligible states list are.

single_matrix_template = '/home/jml230/BdpA Publishable/rfc/{var}_eng/dist_matrix_eng_all.npy' # points to where the dist matrix arrays are. Assume numpy array files. Single versions, so files are not per-run.
single_weights_template = '/home/jml230/BdpA Publishable/rfc/{var}_eng/{var}_eligible_states_eng_all.dat' # points to where the final weights are, a text file

### Old/tester variables
#variants = ['wsh2045']
#ref_path = "/home/jml230/BdpA Publishable/new_folded_struct"
#matrix_template = "/home/jml230/BdpA Publishable/rfc/{var}/dist_matrix_{file}.npy"


parser = argparse.ArgumentParser(description='Process size_var')
parser.add_argument('--size', metavar='N', nargs='?', type=int, action='store', const=8, default=8)
parser.add_argument('--subset', action='store_true', help='calculate submatrix of (default) size 8') 
parser.add_argument('--sigma', default=6, type=int, help="exponent for the logistic contact")
parser.add_argument('--radius', default=5.0, type=float, help="midpoint distance for the logistic contact")
parser.add_argument('--eligible_states', '-eg', action='store_true', help="use a list of eligible states instead of every single segment.")
parser.add_argument('--single', '-s', action='store_true', help="distance matrices are saved in a single file instead of per-run.")
args = parser.parse_args()
size_var = args.size
subset_var = args.subset
sigma_var = args.sigma
radius_var = args.radius


## Functions

def compare_drms(subset=False, size=size_var):
    """
    The catch-all function to calculate drms of a set of distance matrices and then weight-average it.
    Final output will be a pickle object that it of the following format (called drms.pickle):
        A list of dictionaries, each one corresponding to a reference structure. There will be n-elements in the list, corresponding to the number of frames.
        There will also be additional datasets (2x the normal number) if subset is set to True, where those datasets will include the sub-matrices. Those will be in the order of (i,i), (i+1,i+1), ..., (N-1,N-1), (N,N).
    
    Just pickle.load() it after it's done. Will likely be huge.

    """
    for variant in tqdm(variants, desc='variant'): # Loop through variants
        final_list = []
        weights = numpy.loadtxt(weights_template.format(var=variant), skiprows=1)
        matrices, num_traj = read_individual_matrices(variant)
        for matrix in tqdm(matrices, desc=f'{variant} traj'): # Go through each trajectory
            ref_dict = {}
            final_list.append(ref_dict)
            for ref_type in ref_var:
                ref_matrix = numpy.loadtxt(dist_matrix_template.format(var=variant, ref=ref_type))
                if subset:
		    # Grab the submatrices and do checks
                    submatrices = get_submatrices(matrix, size)
                    ref_submatrices = get_submatrices(ref_matrix, size)
                    num_matrices = len(submatrices) # Figure out the number submatrices

                    assert len(submatrices) == len(ref_submatrices), 'Wasn\'t able to generate the same number of submatrices for the reference and mean matrices.'

                    # Calculating the submatrices stuff and output to dedicated file
                    submatrix_list = []
                    for idx in range(0, num_matrices):
                        val_drms = drms(submatrices[idx], ref_submatrices[idx])
                        submatrix_list.append(val_drms)
                    
                    # Saving the array
                    ref_dict[f'{ref_type}_submatrices'] = numpy.asarray(submatrix_list)

                # Do the normal thing of computing the whole matrix drms
                val_drms = drms(matrix, ref_matrix)
                ref_dict[ref_type]=val_drms

        with open(f'{rfc_path}/{variant}/drms_{size}.pickle', 'wb') as po:
            pickle.dump(final_list, po)


def compare_mean_drms(subset=False, size=size_var):
    """
    The catch-all function to compare a single mean distance matrix with a set of reference distance matrices. 
    """
    for variant in variants: # Loop through variants
        #TODO current tied to having one mean_matrix and multiple ref_var. Can be generalized to more...
        mean_matrix = numpy.load(mean_matrix_template.format(var=variant)) # Load mean matrix 
        with open(f'{rfc_path}/{variant}/dist_output.dat', 'w') as fo:
            fo.write(f"{'ref_type':30} {'dRMS':>18}\n")
            for ref_type in ref_var:
                ref_matrix = numpy.loadtxt(dist_matrix_template.format(var=variant, ref=ref_type))
                if subset:
                    # Grab the submatrices and do checks
                    submatrices = get_submatrices(mean_matrix, size)
                    ref_submatrices = get_submatrices(ref_matrix, size)
                    num_matrices = len(submatrices) # Figure out the number submatrices

                    assert len(submatrices) == len(ref_submatrices), 'Wasn\'t able to generate the same number of submatrices for the reference and mean matrices.'
                    # Calculating the submatrices stuff and output to dedicated file
                    with open(f'{rfc_path}/{variant}/{ref_type}_output.dat', 'w') as subfo:
                        subfo.write(f"{'index':>8}\t{'dRMS':>18}\n")
                        for idx in range(0, num_matrices):
                            val_drms = drms(submatrices[idx], ref_submatrices[idx])
                            subfo.write(f'{idx:8}\t{val_drms:>18}\n')
                
                # Do the normal thing of computing the whole matrix drms
                val_drms = drms(mean_matrix, ref_matrix)
                fo.write(f'{ref_type:30} {val_drms:>18}\n')


def compare_transformed_drms(sigma=sigma_var, radius=radius_var, subset=False, size=size_var):
    """
    The catch-all function to calculate a transformed drms of a set of distance matrices and then weight-average it.

    Instead of the dRMS, the "distance" is transformed to CC = 1- [1/(1+(r/radius)^sigma)] instead.
    This theoretically will limit the extra effects of longer distances distances dominating (who has extra room for fluctuations)

    Sigma is defaulted to 6. Larger values is supposed to make the 1->0 transition sharper.
    Radius is defaulted to 5 Angstroms. This is the "inflection point" for the sigmoid shape. 5 Angstroms is close enough to the 4.5 Ang we use for contacts I think.

    Shamefully stolen from PyLOOS and Dr. Alan Grossfield. This is because I already had distances calculated and his code doesn't take
    into account of weights.

    Final output will be a pickle object that it of the following format (called logistic_drms.pickle):
        A list of dictionaries, each one corresponding to a reference structure. There will be n-elements in the list, corresponding to the number of frames.
        There will also be additional datasets (2x the normal number) if subset is set to True, where those datasets will include the sub-matrices. Those will be in the order of (i,i), (i+1,i+1), ..., (N-1,N-1), (N,N).
    
    Just pickle.load() it after it's done. Will likely be huge if doing subsets.

    """
    for variant in tqdm(variants, desc='variant'): # Loop through variants
        final_list = []
        if args.single: # If the trajectories are already saved as 1 .npy
            weights = numpy.loadtxt(single_weights_template.format(var=variant), skiprows=1, usecols=3)
            matrices, num_traj = read_matrices(variant)
        else: # If trajectories are still saved as "per-run"
            weights = numpy.loadtxt(weights_template.format(var=variant), skiprows=1)
            matrices, num_traj = read_individual_matrices(variant)

        for matrix in tqdm(matrices, desc=f'{variant} traj'): # Go through each trajectory
            ref_dict = {}
            final_list.append(ref_dict)
            for ref_type in ref_var:
                ref_matrix = numpy.loadtxt(dist_matrix_template.format(var=variant, ref=ref_type))

                # Do the logistic transformation
                ref_matrix = logistic_transform(ref_matrix, sigma_var, radius_var)
                matrix = logistic_transform(matrix, sigma_var, radius_var)

                if subset:
                    # Grab the submatrices and do checks
                    submatrices = get_submatrices(matrix, size)
                    ref_submatrices = get_submatrices(ref_matrix, size)
                    num_matrices = len(submatrices) # Figure out the number submatrices

                    assert len(submatrices) == len(ref_submatrices), 'Wasn\'t able to generate the same number of submatrices for the reference and mean matrices.'

                    # Calculating the submatrices stuff and output to dedicated file
                    submatrix_list = []
                    for idx in range(0, num_matrices):
                        val_drms = drms(submatrices[idx], ref_submatrices[idx])
                        submatrix_list.append(val_drms)
                    
                    # Saving the array
                    ref_dict[f'{ref_type}_submatrices'] = numpy.asarray(submatrix_list)

                # Do the normal thing of computing the whole matrix drms
                val_drms = drms(matrix, ref_matrix)
                ref_dict[ref_type]=val_drms

        with open(f'{rfc_path}/{variant}/logistic_drms_{radius_var}.pickle', 'wb') as po:
            pickle.dump(final_list, po)


def compare_transformed_mean_drms(sigma=sigma_var, radius=radius_var, subset=False, size=size_var):
    """
    The catch-all function to compare a single mean distance matrix with a set of reference distance matrices. All matrices are logistically transformed before it is compared. 
    """
    for variant in variants: # Loop through variants
        #TODO current tied to having one mean_matrix and multiple ref_var. Can be generalized to more...
        mean_matrix = numpy.load(mean_matrix_template.format(var=variant)) # Load mean matrix 
        mean_matrix = logistic_transform(mean_matrix, sigma_var, radius_var) # Logistic Transform the mean matrix

        with open(f'{rfc_path}/{variant}/dist_transformed_{radius_var}_output.dat', 'w') as fo:
            fo.write(f"{'ref_type':30} {'dRMS':>18}\n")
            for ref_type in ref_var:
                ref_matrix = numpy.loadtxt(dist_matrix_template.format(var=variant, ref=ref_type))

                # Do the logistic transformation
                ref_matrix = logistic_transform(ref_matrix, sigma_var, radius_var)

                if subset:
                    # Grab the submatrices and do checks
                    submatrices = get_submatrices(mean_matrix, size)
                    ref_submatrices = get_submatrices(ref_matrix, size)
                    num_matrices = len(submatrices) # Figure out the number submatrices

                    assert len(submatrices) == len(ref_submatrices), 'Wasn\'t able to generate the same number of submatrices for the reference and mean matrices.'
                    # Calculating the submatrices stuff and output to dedicated file
                    with open(f'{rfc_path}/{variant}/{ref_type}_transformed_{radius_var}_output.dat', 'w') as subfo:
                        subfo.write(f"{'index':>8}\t{'dRMS':>18}\n")
                        for idx in range(0, num_matrices):
                            val_drms = drms(submatrices[idx], ref_submatrices[idx])
                            subfo.write(f'{idx:8}\t{val_drms:>18}\n')
                
                # Do the normal thing of computing the whole matrix drms
                val_drms = drms(mean_matrix, ref_matrix)
                fo.write(f'{ref_type:30} {val_drms:>18}\n')


def calc_avg_length():
    """
    The catch-all function to extract average length of a set of n residue from the distance matrices and then weight-average it.
    
    Saves as a numpy array and returns nothing.

    """
    for variant in tqdm(variants, desc='variant'): # Loop through variants
        weights = numpy.loadtxt(weights_template.format(var=variant), skiprows=1)
        matrices, num_traj = read_individual_matrices(variant)
        matrix_dim = len(matrices[0])

        for size_idx in range(2, matrix_dim + 1): # Loop through the different sizes of submatrices; note: 1-based
            num_matrices = matrix_dim - size_idx + 1 # Has to be 1-based
            submatrix_list = numpy.zeros(num_matrices) # List of weighted avg distance for each atom pair

            for traj_idx, matrix in enumerate(tqdm(matrices, desc=f'{variant} traj')): # Go through each trajectory
                # Grab the submatrices and do checks
                submatrices = get_submatrices(matrix, size_idx)
                assert num_matrices == len(submatrices), f'Wrong indexing; {len(submatrices)} != expected length of {num_matrices}'
    
                # Calculating the submatrices stuff and weight-average them
                submatrix_list = numpy.zeros(num_matrices)
                index_list = numpy.full(num_matrices, size_idx)
                for idx in range(0, num_matrices):
                    submatrix_list[idx] += weights[traj_idx] * matrix[0,-1]
                        
            
            # Appending to the final numpy array
            if not final_list:
                final_list = numpy.dstack((index_list, submatrix_list))
            else: 
                final_list = numpy.append(final_list, numpy.dstack((index_list, submatrix_list), axis=1))

        with open(f'{rfc_path}/{variant}/avg_dist.npy', 'wb') as no:
            numpy.save(no, final_list)
    

def logistic_transform(a_matrix, sigma=6, radius=5.0):
    """
    Code to calculate the logistic transform. Shamefully stolen from PyLOOS and Dr. Grossfield. 

    Instead of the dRMS, each "distance" is transformed CC = 1- [1/(1+(r/radius)^sigma)] instead. The latter term is referred to C(r)
    This theoretically will limit the extra effects of longer distances distances dominating (who has extra room for fluctuations)

    Sigma is defaulted to 6. Larger values is supposed to make the 1->0 transition sharper.
    Radius is defaulted to 5 Angstroms. This is the "inflection point" for the sigmoid shape. 5 Angstroms is close enough to the 4.5 Ang we use for contacts I think.

    returns trans_matrix, which is the transformed matrix.

    """
    check_dist_matrix(a_matrix, desc='logistic transform input')

    a_shape = len(a_matrix)
    
    trans_matrix = a_matrix # Duplicate...

    for row in range(0, a_shape): # Loop through all rows
        for column in range(row, a_shape): # Loop through all columns
            term = (a_matrix[row, column] / radius )**sigma
            if row == column:
                trans_matrix[row, column] = 1 + term
            else:
                trans_matrix[row, column] = 1 + term
                trans_matrix[column, row] = 1 + term

    # Do the 1 / [term] thing for the C(r) term...
    trans_matrix = 1 / trans_matrix

    # Do the final 1 - C(r) thing...
    trans_matrix = 1 - trans_matrix

    return trans_matrix


def drms(a_matrix, b_matrix):
    """
    Code to calculate dRMS:
    
    Applies the following equation given distance matrices a_matrix and b_matrix and returns a float:

    drms(a_matrix, b_matrix) = sqrt( 1/(n * (n-1)) * sum of i[ sum of j[ (a_matrix(ij) - b_matrix(ij))^2 ] ] )

    
    """
    
    check_dist_matrix(a_matrix, desc='First input')
    check_dist_matrix(b_matrix, desc='Second input')

    a_shape = len(a_matrix)
    b_shape = len(b_matrix)
 
    assert a_shape == b_shape, "Matrices given are not of the same shape."
   
    sum_term = 0 # Value of the double sum
    diff = 0 # Value of the difference

    #TODO Tied to 2 dimensions now. should expand to a new algorithm using nditer
    for row in range(0, a_shape): # Loop through all rows
        for column in range(row+1, a_shape): # Loop through columns that are > than row, this is to prevent double counting and speed up stuff.
            diff = a_matrix[row, column] - b_matrix[row, column]
            sum_term += 2 * (diff)**2

    denominator = 1 / (a_shape * (a_shape-1))
    
    return numpy.sqrt(denominator * sum_term)


def check_dist_matrix(matrix, desc='Matrix'):
    # Check if input is a list or an array-type.
    if not isinstance(matrix, (list, numpy.ndarray)):
        raise TypeError(f'{desc} is not a list or an array.')
    
    # Check dimension of matrix
    dimension = numpy.shape(matrix)
    for idx in range(1, len(dimension)):
        if dimension[idx-1] != dimension[idx]:
           raise TypeError(f'{desc} is not a square matrix.')
    
    # Check to see if symmetrical
    assert numpy.allclose(matrix, matrix.T), f'{desc} is not symmetrical. Probably not a valid distance matrix.'

    # Check if diagonal is zero
    for x in range(0,len(dimension)):
        assert matrix[x,x] == 0, f'Diagonal element ({x},{x}) is not zero in the mean matrix.'


def get_submatrices(a_matrix, size=8):
    """
    Function that picks out submatrices

    Say given the size 8 (default) then it'll loop through the square matrix and output all the possible contiguous submatrices into 
    one big array. Does the crazy checks first (check_dist_matrix()).

    returns a list with all the arrays.

    """
    check_dist_matrix(a_matrix, desc='Matrix')
    
    size = int(size) # Make sure the size is an integer
    
    matrix_size = len(a_matrix) # Grab the length of the input matrix

    submatrices = [] # Empty submatrix

    for idx in range(0, matrix_size - size):
        #TODO Currently tied to 2 dimensions. should expand to a new algorithm using nditer.
        temp_matrix = a_matrix[idx:idx+size, idx:idx+size]
        try:
            submatrices.append(temp_matrix)
        except:
            submatrices = [temp_matrix]

    assert len(submatrices) == matrix_size - size, "Number of submatrices generated \({len(submatrices)}\) is not consistent with what's expected ({matrix_size - size})"
    return submatrices


def read_individual_matrices(variant):
    """
    Function that reads the individual distance matrices (from the .npy files) and concatenates them into one big list.
    
    returns: the list of individual distance matrices and the number of distance matrices in the former.

    """
    num_traj = 0 # Counter to check we have the correct final matrix, that is the sum of shape[0] == the shape[0] of 'matrices'
    matrices = numpy.asarray([])

    for file_idx in trange(1,6, desc=f'{variant} reading file:'): # Loop through the 5 files
        temp = numpy.load(matrix_template.format(var=variant, file_num=file_idx))
        num_traj += temp.shape[0]
        try:
            matrices = numpy.append(matrices, temp, axis=0)
        except ValueError:
            matrices = temp

    assert matrices.shape[0] == num_traj, 'The combined numpy array matrix is not created correctly.'
    
    return matrices, num_traj

def read_matrices(variant):
    """
    Function that reads the distance matrices that are already concatenated into single .npy.
    
    returns: the list of individual distance matrices and the number of distance matrices in the former.

    """

    matrices = numpy.load(single_matrix_template.format(var=variant))
    num_traj = len(matrices)
    
    return matrices, num_traj


def load_weights(variant):
    """
    Function that reads the weights and then explicitly normalize them if necessary.

    """
    if args.single:
        weights = numpy.loadtxt(single_weights_template.format(var=variant), skiprows=1, usecols=3)
    else:
        weights = numpy.loadtxt(weights_template.format(var=variant), skiprows=1)
    if not numpy.isclose(sum(weights), 1):
        print('Sum doesn\'t add up to 1, explicitly normalizing it now.')
        weights = weights / sum(weights)

    return weights


def calc_mean_matrix(use_weights=False):
    """
    A catch-all function that reads individual distance matrix and then weights them if weights=True.

    """
    for variant in tqdm(variants, desc='variants'): # Loop through variants
        # Create dummy matrix and all that, assumed to be 58 x 58, here.
        #TODO generalize this. 
        mean_matrix = numpy.zeros((58,58)) # Blank mean matrix
        
        if args.single: # If the trajectories are already saved as 1 .npy
            matrices, num_traj = read_matrices(variant)
        else: # If trajectories are still saved as "per-run"
            matrices, num_traj = read_individual_matrices(variant)
        
        # If we're using weights
        if use_weights:
            weights = load_weights(variant)
        else:
            weights = numpy.ones(matrices.shape[0])
            weights = weights / sum(weights) 
        
        assert matrices.shape[0] == weights.shape[0], f'Number of trajectories are not consistent. You have {matrices.shape[0]} segments but {weights.shape[0]} number of weights in the file.'
   
        for traj_idx in trange(0, num_traj, desc=f'{variant} traj number'):
            for row in range(0,58): # Loop through all rows
                for column in range(row+1,58): # Loop through columns that are > than row, this is to prevent double counting and speed up stuff.
                    val = matrices[traj_idx, row, column]
                    mean_matrix[row, column] += weights[traj_idx] * val
                    mean_matrix[column, row] += weights[traj_idx] * val
    
        # Final check to see if it's a valid matrix
        check_dist_matrix(mean_matrix)
        
        numpy.save(f'{rfc_path}/{variant}_eng/mean_matrix.npy', mean_matrix)


if __name__ == "__main__":
    #compare_drms(subset=subset_var, size=size_var)
    # Runs through the whole suite of generating a "mean" distance matrix then compare it to a list of refs for the overall dRMS.
    #calc_mean_matrix(use_weights=True)
    #compare_mean_drms(subset=subset_var, size=size_var)
    compare_transformed_drms(sigma=sigma_var, radius=radius_var, subset=subset_var, size=size_var)
    compare_transformed_mean_drms(sigma=sigma_var, radius=radius_var, subset=subset_var, size=size_var)
    #calc_mean_matrix(use_weights=True)
    pass
