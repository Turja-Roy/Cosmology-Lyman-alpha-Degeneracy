import numpy as np


def apply_fake_spectra_bugfixes():
    # Fixes:
    # 1. uint32 overflow in get_npart calculation
    # 2. float32/float64 type mismatches in C extension
    try:
        from fake_spectra import abstractsnapshot
        from fake_spectra import spectra
        from fake_spectra._spectra_priv import _Particle_Interpolate as _PI_original
        
        # FIX 1: uint32 overflow
        def get_npart_fixed(self):
            """Get the total number of particles (fixed for uint32 overflow)."""
            npart_total = self.get_header_attr("NumPart_Total").astype(np.int64)
            npart_high = self.get_header_attr("NumPart_Total_HighWord").astype(np.int64)
            return npart_total + (2**32) * npart_high
        
        abstractsnapshot.AbstractSnapshotFactory.get_npart = get_npart_fixed
        abstractsnapshot.HDF5Snapshot.get_npart = get_npart_fixed
        abstractsnapshot.BigFileSnapshot.get_npart = get_npart_fixed
        
        # FIX 2: float32/float64 type casting
        def _do_interpolation_work_fixed(self, pos, vel, elem_den, temp, hh, amumass, line, get_tau):
            """Run the interpolation, casting arrays to whatever width the C build wants."""
            if self.turn_off_selfshield:
                gamma_X = 0
            else:
                gamma_X = line.gamma_X
            
            # Ensure all scalar parameters are float32
            box = np.float32(self.box)
            velfac = np.float32(self.velfac)
            atime = np.float32(self.atime)
            lambda_X = np.float32(line.lambda_X * 1e-8)
            gamma_X_f32 = np.float32(gamma_X)
            fosc_X = np.float32(line.fosc_X)
            amumass_f32 = np.float32(amumass)
            tautail = np.float32(self.tautail)
            
            # Which float width the C extension accepts depends on how it was built:
            # py_module.cpp checks pos/vel/dens/temp/h against one type and raises
            # TypeError on the other. Current builds want float64; older ones wanted
            # float32, so try the wide one and fall back rather than pinning a version.
            axis = np.asarray(self.axis, dtype=np.int32)
            cofm = np.asarray(self.cofm, dtype=np.float64)  # cofm is always float64

            def _call(dtype):
                return _PI_original(get_tau*1, self.nbins, self.kernel_int, box, velfac, atime,
                                    lambda_X, gamma_X_f32, fosc_X, amumass_f32, tautail,
                                    np.asarray(pos, dtype=dtype), np.asarray(vel, dtype=dtype),
                                    np.asarray(elem_den, dtype=dtype),
                                    np.asarray(temp, dtype=dtype),
                                    np.asarray(hh, dtype=dtype), axis, cofm)

            try:
                return _call(np.float64)
            except TypeError:
                return _call(np.float32)
        
        spectra.Spectra._do_interpolation_work = _do_interpolation_work_fixed
        
        print("Applied fake_spectra bugfixes for Python 3.13 compatibility")
        return True
        
    except ImportError:
        print("Warning: fake_spectra not installed - skipping bugfixes")
        return False


def compute_temp_density_chunked(spec, elem, ion, chunk_size=None, verbose=True):
    """
    Compute temperature and density-weighted density in chunks to reduce memory usage.
    Each sightline is computed independently, so chunking produces identical results
    to computing all at once, but with much lower peak memory usage.
    """
    import gc
    
    # Get total number of sightlines
    n_sightlines = spec.cofm.shape[0]
    
    # Auto-detect chunk size if not specified
    if chunk_size is None:
        if n_sightlines < 1000:
            chunk_size = n_sightlines  # No chunking for small datasets
        elif n_sightlines < 5000:
            chunk_size = 1000
        else:
            chunk_size = 2000
    
    n_chunks = (n_sightlines + chunk_size - 1) // chunk_size
    
    if verbose:
        if n_chunks == 1:
            print(f"Computing temperature and density for {n_sightlines} sightlines...")
        else:
            print(f"\nComputing temperature and density in {n_chunks} chunks of ~{chunk_size} sightlines...")
            print(f"  Total sightlines: {n_sightlines}")
            print(f"  Chunk size: {chunk_size}")
    
    # Save original sightline configuration
    original_cofm = spec.cofm
    original_axis = spec.axis
    original_numlos = spec.NumLos
    
    # Save original colden (needed for density normalization)
    original_colden = spec.colden.get((elem, ion))
    if original_colden is None:
        return False, f"Column density not found for {elem} {ion}. Must compute tau/colden first."
    
    temp_chunks = []
    dens_chunks = []
    
    try:
        for i in range(n_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, n_sightlines)
            n_in_chunk = end_idx - start_idx
            
            if verbose and n_chunks > 1:
                print(f"\n  Chunk {i+1}/{n_chunks}: sightlines {start_idx}-{end_idx-1} ({n_in_chunk} sightlines)")
            
            # Garbage collection before each chunk
            gc.collect()
            
            # Temporarily modify spec to use subset of sightlines
            spec.cofm = original_cofm[start_idx:end_idx]
            spec.axis = original_axis[start_idx:end_idx]
            spec.NumLos = n_in_chunk
            spec.colden[(elem, ion)] = original_colden[start_idx:end_idx]
            
            # Compute temperature for this chunk
            if verbose:
                prefix = "    " if n_chunks > 1 else ""
                print(f"{prefix}Computing temperature...", end=' ', flush=True)
            try:
                temp_chunk = spec._get_mass_weight_quantity(spec._temp_single_file, elem, ion)
                temp_chunks.append(temp_chunk)
                if verbose:
                    print(f"OK (shape: {temp_chunk.shape})")
            except Exception as e:
                return False, f"Temperature computation failed on chunk {i+1}: {e}"
            
            # Clear memory between temp and density
            del temp_chunk
            gc.collect()
            
            # Compute density for this chunk
            if verbose:
                prefix = "    " if n_chunks > 1 else ""
                print(f"{prefix}Computing density-weighted density...", end=' ', flush=True)
            try:
                dens_chunk = spec._get_mass_weight_quantity(spec._densweightdens, elem, ion)
                dens_chunks.append(dens_chunk)
                if verbose:
                    print(f"OK (shape: {dens_chunk.shape})")
            except Exception as e:
                return False, f"Density computation failed on chunk {i+1}: {e}"
            
            # Clear memory after chunk
            del dens_chunk
            gc.collect()
        
        # Restore original configuration
        spec.cofm = original_cofm
        spec.axis = original_axis
        spec.NumLos = original_numlos
        spec.colden[(elem, ion)] = original_colden
        
        # Concatenate all chunks
        if verbose and n_chunks > 1:
            print(f"\n  Concatenating {n_chunks} chunks...")
        
        temp_full = np.concatenate(temp_chunks, axis=0)
        dens_full = np.concatenate(dens_chunks, axis=0)
        
        if verbose and n_chunks > 1:
            print(f"    Temperature shape: {temp_full.shape}")
            print(f"    Density shape: {dens_full.shape}")
        
        # Store in spec object
        spec.temp[(elem, ion)] = temp_full
        spec.dens_weight_dens[(elem, ion)] = dens_full
        
        # Final cleanup
        del temp_chunks, dens_chunks, temp_full, dens_full
        gc.collect()
        
        if verbose and n_chunks > 1:
            print("  ✓ Temperature and density computation complete")
        
        return True, None
        
    except Exception as e:
        # Always restore original configuration on error
        spec.cofm = original_cofm
        spec.axis = original_axis
        spec.NumLos = original_numlos
        spec.colden[(elem, ion)] = original_colden
        
        return False, f"Unexpected error: {e}"
