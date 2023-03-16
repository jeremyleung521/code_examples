# get_coord.py
#
# A script to output coordinates as auxdata in WESTPA on the fly.
# Coordinates should be outputted into 'iterations/iter_{i:08d}/auxdata/coord'
# for the haMSM or other uses, under the shape (n_segs, n_frames, n_atoms, 3).
#
# Make sure you declare the dataset in west.cfg and specify `loader: npy_loader`
# under executable/dataset. You will have to run this script in runseg.sh and copy
# the output file using `cp coord.npy $WEST_COORD_RETURN`.


import mdtraj
import numpy
import os.path

# Topology and other paths
topology_path = os.path.expandvars('$WEST_SIM_ROOT/common_files/diala_nowat_ff14_hmr.prmtop')
traj_path = os.path.expandvars('$WEST_CURRENT_SEG_DATA_REF/seg.nc')
parent_path = os.path.expandvars('$WEST_CURRENT_SEG_DATA_REF/parent.ncrst')
ref_path = os.path.expandvars('$WEST_SIM_ROOT/common_files/diala_nowat_eq2.pdb')
atom_slice = numpy.asarray(range(0, 22)) # Atoms to load, see MDTraj documentation for format
ref_slice = atom_slice # Atoms to "superpose"/align with

# Loading the trajectories...
parent_traj = mdtraj.load(parent_path, top=topology_path, atom_indices=atom_slice)
seg_traj = mdtraj.load(traj_path, top=topology_path, atom_indices=atom_slice)
ref_file = mdtraj.load(ref_path, top=topology_path, atom_indices=atom_slice)

# Start loading and stuff
full_traj = parent_traj.join(seg_traj)
full_traj = full_traj.superpose(ref_file, atom_indices=ref_slice)
all_coords = full_traj._xyz

# Alternate loading just in case things are weird, like traj and parent don't share the same topology
parent_traj = parent_traj.superpose(ref_file, atom_indices=ref_slice)
seg_traj = seg_traj.superpose(ref_file, atom_indices=ref_slice)
all_coords = numpy.squeeze(numpy.concatenate((parent_traj._xyz,seg_traj._xyz)))

#print(all_coords.shape)
numpy.save('coord.npy',all_coords)

