import h5py
import numpy as np
import os


# ============== #
# HDF5 UTILITIES #
# ============== #

# Load snapshot metadata from HDF5 file header.
def load_snapshot_metadata(filepath):
    with h5py.File(filepath, 'r') as f:
        header = f['Header']

        metadata = {
            'redshift'      : float(header.attrs['Redshift']),
            'boxsize'       : float(header.attrs['BoxSize']),  # ckpc/h
            'hubble'        : float(header.attrs['HubbleParam']),
            'omega_matter'  : float(header.attrs['Omega0']),
            'omega_lambda'  : float(header.attrs['OmegaLambda']),
            'num_gas'       : int(header.attrs['NumPart_ThisFile'][0]),
            'num_dm'        : int(header.attrs['NumPart_ThisFile'][1]),
            'num_stars'     : int(header.attrs['NumPart_ThisFile'][4]),
            'time'          : float(header.attrs['Time']),
        }

        # Derived quantities
        metadata['boxsize_mpc'] = metadata['boxsize'] / metadata['hubble'] / 1000  # Mpc/h comoving
        metadata['boxsize_proper'] = metadata['boxsize'] / metadata['hubble'] / (1 + metadata['redshift']) / 1000  # Mpc proper

    return metadata


# Load gas particle properties from snapshot.
def load_gas_properties(filepath, fields=None, stride=1, max_particles=None):
    with h5py.File(filepath, 'r') as f:
        if 'PartType0' not in f:
            raise ValueError("No gas particles (PartType0) in snapshot")
        
        gas = f['PartType0']
        
        # If fields is None, include all
        if fields is None:
            fields = list(gas.keys())
        
        # Determine slice
        if max_particles is not None:
            end = min(max_particles * stride, len(gas['Coordinates']))
            slice_obj = slice(0, end, stride)
        else:
            slice_obj = slice(None, None, stride)
        
        # Load data
        data = {}
        for field in fields:
            if field in gas:
                data[field] = gas[field][slice_obj]
            else:
                print(f"Warning: Field '{field}' not found in snapshot")
        
        # Add metadata
        data['n_particles'] = len(data[list(data.keys())[0]])
        
    return data


# Print the structure of an HDF5 snapshot file.
def explore_hdf5_structure(filepath):
    structure = {'groups': [], 'header': {}}
    
    with h5py.File(filepath, 'r') as f:
        # Top-level groups
        structure['groups'] = list(f.keys())
        
        print(f"\n{'='*70}")
        print(f"HDF5 Structure: {os.path.basename(filepath)}")
        print(f"{'='*70}")
        print(f"\nTop-level groups: {structure['groups']}")
        
        # Header info
        if 'Header' in f:
            header = f['Header']
            print(f"\nHeader attributes:")
            for key in header.attrs.keys():
                structure['header'][key] = header.attrs[key]
        
        # Particle type info
        for ptype in ['PartType0', 'PartType1', 'PartType4', 'PartType5']:
            if ptype in f:
                datasets = list(f[ptype].keys())
                structure[ptype] = datasets
                
                ptype_name = {
                    'PartType0': 'GAS',
                    'PartType1': 'DARK MATTER',
                    'PartType4': 'STARS',
                    'PartType5': 'BLACK HOLES'
                }.get(ptype, ptype)
                
                print(f"\n{ptype} ({ptype_name}):")
                for ds in datasets:
                    shape = f[ptype][ds].shape
                    dtype = f[ptype][ds].dtype
                    print(f"  {ds:30s} shape={str(shape):20s} dtype={dtype}")
    
    return structure
