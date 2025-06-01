import os
import numpy as np
from pathlib import Path
import re
from fractions import Fraction
import logging
import concurrent.futures
from tqdm import tqdm

# Global parameters
DEBUG = False

debugfile = open("debug.log", "w")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def debug_print(*args, **kwargs):
    """Print debug information only if DEBUG is True"""
    if DEBUG:
        # with open(debugfile, "a") as f:
        # debugfile.write(f"{args} {kwargs}\n")
        print(*args, **kwargs, file=debugfile)

def get_element_symbol(label):
    """Extract element symbol from atom label (e.g., 'O1' -> 'O')"""
    # Match the first letter and any following letters (for elements like Fe, Co, etc.)
    match = re.match(r'([A-Za-z]+)', label)
    if match:
        return match.group(1)
    return label

def parse_symop(symop_str):
    """Parse symmetry operation string into rotation matrix, translation vector and spin sign"""
    # Split into rotation and translation parts
    parts = symop_str.split(',')
    rot_matrix = np.zeros((3, 3))
    trans_vector = np.zeros(3)
    spin = 1  # Default spin sign
    
    # Process each part (x,y,z components)
    for i, part in enumerate(parts[:3]):  # Only process first 3 parts
        part = part.strip()
        
        # First handle rotation matrix
        for j, var in enumerate(['x', 'y', 'z']):
            if var in part:
                # Check if there's a minus sign before the variable
                if part.split(var)[0].strip().endswith('-'):
                    rot_matrix[i, j] = -1.0
                else:
                    rot_matrix[i, j] = 1.0
        
        # Then handle translation
        if '+' in part:
            trans = part.split('+')[1].strip()
            if '/' in trans:
                num, den = trans.split('/')
                trans_vector[i] = float(Fraction(int(num), int(den)))
        elif '-' in part and not part.startswith('-'):
            trans = part.split('-')[1].strip()
            if '/' in trans:
                num, den = trans.split('/')
                trans_vector[i] = -float(Fraction(int(num), int(den)))
    
    # Get spin sign from the last part
    if len(parts) > 3:
        spin_part = parts[3].strip()
        if spin_part == '-1':
            spin = -1
    
    return rot_matrix, trans_vector, spin

def apply_symmetry_operations(atoms, positions, occupancies, moments, symops, is_op_on_spin=False):
    """Apply symmetry operations to atoms and their moments"""
    new_atoms = []
    new_positions = []
    new_occupancies = []
    new_moments = []
    
    # Convert positions to numpy array for easier manipulation
    positions = np.array(positions)
    moments = np.array([m[1] for m in moments])  # Extract moment vectors
    
    # Process each atom
    for i, (atom, pos, occ, moment) in enumerate(zip(atoms, positions, occupancies, moments)):
        debug_print("*"*50)
        debug_print(f"Processing atom {i}: {atom} at {pos} with moment {moment} and occupancy {occ}")
        # Add original atom
        new_atoms.append(atom)
        new_positions.append(pos)
        new_occupancies.append(occ)
        new_moments.append((atom, moment))
        
        # Apply each symmetry operation
        for symop in symops:
            try:
                rot_matrix, trans_vector, spin = parse_symop(symop)
                debug_print(f"\n>>>>>>>>> Applying symmetry operation: {symop}")
                debug_print(f"Rotation matrix:\n {rot_matrix}")
                debug_print(f"Translation vector: {trans_vector}")
                debug_print(f"det of rotation matrix: {np.linalg.det(rot_matrix)}")
                debug_print(f"Spin: {spin}")
                
                # Apply rotation to position
                new_pos = np.dot(rot_matrix, pos) + trans_vector
                # Bring position back to unit cell with higher precision
                new_pos = np.mod(new_pos, 1.0)
                # Round to 6 decimal places to avoid numerical errors
                new_pos = np.round(new_pos, decimals=6)
                
                # Apply rotation to moment (as an axial vector)
                if is_op_on_spin:
                    det = np.linalg.det(rot_matrix)
                    new_moment = np.dot(rot_matrix, moment) * det * spin
                else:
                    new_moment = moment * spin
                # Round moment to 6 decimal places to avoid numerical errors
                new_moment = np.round(new_moment, decimals=6)
                
                debug_print(f"pos: {pos} --> {new_pos}")
                debug_print(f"moment: {moment} --> {new_moment}")
                
                # Check if this position already exists
                position_exists = False
                for j, existing_pos in enumerate(new_positions):
                    # Use exact equality for positions after rounding
                    if np.array_equal(new_pos, existing_pos):
                        position_exists = True
                        break
                
                if not position_exists:
                    # If no duplicate position found, add the new atom
                    new_atoms.append(atom)
                    new_positions.append(new_pos)
                    new_occupancies.append(occ)
                    new_moments.append((atom, new_moment))
                    debug_print(f"Added new atom at {new_pos} with moment {new_moment} and occupancy {occ}")
            except Exception as e:
                logging.warning(f"Failed to apply symmetry operation {symop}: {str(e)}")
                continue
    
    # Print final moments for debugging
    debug_print("\nFinal moments:")
    for i, (atom, moment) in enumerate(new_moments):
        debug_print(f"Atom {i}: {atom} at {new_positions[i]} with moment {moment} and occupancy {new_occupancies[i]}")
    
    return new_atoms, new_positions, new_occupancies, new_moments

def apply_centering_operations(atoms, positions, occupancies, moments, centering_ops):
    """Apply magnetic centering operations to atoms and their moments"""
    new_atoms = []
    new_positions = []
    new_occupancies = []
    new_moments = []
    
    # Convert positions to numpy array for easier manipulation
    positions = np.array(positions)
    moments = np.array([m[1] for m in moments])  # Extract moment vectors
    
    x=0
    # Process each atom
    for i, (atom, pos, occ, moment) in enumerate(zip(atoms, positions, occupancies, moments)):
        # Add original atom
        new_atoms.append(atom)
        new_positions.append(pos)
        new_occupancies.append(occ)
        new_moments.append((atom, moment))
        
        # Apply each centering operation
        for centering_op in centering_ops:
            try:
                rot_matrix, trans_vector, spin = parse_symop(centering_op)
                if x==0:
                    debug_print(f"Applying centering operation: {centering_op}")
                    debug_print(f"Rotation matrix: {rot_matrix}")
                    debug_print(f"Translation vector: {trans_vector}")
                    debug_print(f"Spin: {spin}")
                
                # Apply translation to position
                new_pos = (pos + trans_vector) % 1.0
                
                # Apply spin sign to moment
                new_moment = moment * spin
                
                # Bring position back to unit cell
                new_pos = new_pos % 1.0
                
                # Check if this position is unique
                is_unique = True
                for existing_pos in new_positions:
                    if np.allclose(new_pos, existing_pos, atol=1e-4):
                        is_unique = False
                        break
                
                if is_unique:
                    new_atoms.append(atom)
                    new_positions.append(new_pos)
                    new_occupancies.append(occ)
                    new_moments.append((atom, new_moment))
            except Exception as e:
                logging.warning(f"Failed to apply centering operation {centering_op}: {str(e)}")
                continue
        x+=1
    
    return new_atoms, new_positions, new_occupancies, new_moments

def remove_uncertainty(value_str):
    """Remove parentheses and trailing dots, but keep the values inside parentheses"""
    # Remove all parentheses
    value = value_str.replace('(', '').replace(')', '')
    # Remove trailing dot if it exists
    if value.endswith('.'):
        value = value[:-1]
    if value.endswith(','):
        value = value[:-1]
    return value

def remove_close_atoms(atoms, positions, occupancies, moments, lattice, min_distance=0.1):
    """Remove atoms that are too close to each other"""
    # Convert positions to numpy array for easier manipulation
    positions = np.array(positions)
    moments = np.array([m[1] for m in moments])
    
    # Convert fractional coordinates to Cartesian coordinates
    cart_positions = np.dot(positions, lattice)
    
    # Calculate all pairwise distances
    n_atoms = len(atoms)
    to_remove = set()
    
    for i in range(n_atoms):
        if i in to_remove:
            continue
        for j in range(i + 1, n_atoms):
            if j in to_remove:
                continue
                
            # Calculate distance between atoms i and j
            diff = cart_positions[i] - cart_positions[j]
            
            # Consider periodic boundary conditions
            # Convert back to fractional coordinates
            inv_lattice = np.linalg.inv(lattice)
            frac_diff = np.dot(diff, inv_lattice)
            # Round to nearest integer
            frac_diff = frac_diff - np.round(frac_diff)
            # Convert back to Cartesian coordinates
            diff = np.dot(frac_diff, lattice)
            
            distance = np.sqrt(np.sum(diff * diff))
            
            if distance < min_distance:
                # Check total occupancy and atom/moment differences
                total_occ = occupancies[i] + occupancies[j]
                atoms_different = atoms[i] != atoms[j]
                moments_different = not np.allclose(moments[i], moments[j], atol=0.01)
                
                if total_occ > 1.1 and (atoms_different or moments_different):
                    logging.warning(f"Found atoms too close to each other with total occupancy > 1.1 and different atoms/moments:\n"
                                  f"Atom {i} ({atoms[i]}) at {positions[i]} with moment {moments[i]} and occupancy {occupancies[i]}\n"
                                  f"Atom {j} ({atoms[j]}) at {positions[j]} with moment {moments[j]} and occupancy {occupancies[j]}\n"
                                  f"Total occupancy: {total_occ}")
                to_remove.add(j)
                debug_print(f">>> Removing atom {j} because it is too close to atom {i}")
                debug_print(f"    positions: {positions[i]}, {positions[j]}")
                debug_print(f"    moments: {moments[i]}, {moments[j]}")
                debug_print(f"    occupancies: {occupancies[i]}, {occupancies[j]}")
    
    # Create new lists excluding removed atoms
    new_atoms = [atom for i, atom in enumerate(atoms) if i not in to_remove]
    new_positions = [pos for i, pos in enumerate(positions) if i not in to_remove]
    new_occupancies = [occ for i, occ in enumerate(occupancies) if i not in to_remove]
    new_moments = [(atom, moment) for i, (atom, moment) in enumerate(zip(atoms, moments)) if i not in to_remove]
    
    if to_remove:
        logging.info(f"Removed {len(to_remove)} atoms that were too close to others")
    
    return new_atoms, new_positions, new_occupancies, new_moments

def read_mcif(file_path, is_op_on_spin=False):
    """Read mcif file and extract relevant information"""
    try:
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'gbk', 'cp1252']
        content = None
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            # If all encodings fail, read as binary and decode with errors='ignore'
            with open(file_path, 'rb') as f:
                content = f.read().decode('utf-8', errors='ignore')
        
        # Initialize data dictionary
        data = {
            'lattice': np.zeros((3, 3)),
            'atoms': [],
            'positions': [],
            'occupancies': [],  # Add occupancies list
            'magnetic_moments': [],  # Will store (label, moment) pairs
            'symops': [],  # Will store symmetry operations
            'centering_ops': []  # Will store magnetic centering operations
        }
        
        # Parse lattice parameters
        lines = content.split('\n')
        alpha = beta = gamma = 90.0  # Default values
        
        for i, line in enumerate(lines):
            if '_cell_length_a' in line:
                data['lattice'][0, 0] = float(remove_uncertainty(line.split()[1]))
            elif '_cell_length_b' in line:
                data['lattice'][1, 1] = float(remove_uncertainty(line.split()[1]))
            elif '_cell_length_c' in line:
                data['lattice'][2, 2] = float(remove_uncertainty(line.split()[1]))
            elif '_cell_angle_alpha' in line:
                alpha = float(remove_uncertainty(line.split()[1])) * np.pi / 180
            elif '_cell_angle_beta' in line:
                beta = float(remove_uncertainty(line.split()[1])) * np.pi / 180
            elif '_cell_angle_gamma' in line:
                gamma = float(remove_uncertainty(line.split()[1])) * np.pi / 180
        
        # Convert angles to radians and calculate lattice vectors
        if not (alpha == beta == gamma == np.pi/2):
            # Non-orthogonal lattice
            a = data['lattice'][0, 0]
            b = data['lattice'][1, 1]
            c = data['lattice'][2, 2]
            
            # Calculate lattice vectors
            data['lattice'][0, 0] = a
            data['lattice'][0, 1] = b * np.cos(gamma)
            data['lattice'][0, 2] = c * np.cos(beta)
            data['lattice'][1, 1] = b * np.sin(gamma)
            data['lattice'][1, 2] = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
            data['lattice'][2, 2] = c * np.sqrt(1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 + 
                                               2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        
        # Parse atomic positions, occupancies, magnetic moments, and symmetry operations
        in_atom_site = False
        in_moment = False
        in_symop = False
        in_centering = False
        has_occupancy = False  # Flag to check if occupancy is present
        
        # First check if occupancy is present
        for line in lines:
            if '_atom_site_occupancy' in line:
                has_occupancy = True
                break
        
        # First read all atoms and positions
        for line in lines:
            if 'loop_' in line:
                in_atom_site = False
                in_moment = False
                in_symop = False
                in_centering = False
            elif '_atom_site_label' in line:
                in_atom_site = True
                continue
            elif '_atom_site_moment.label' in line:
                in_moment = True
                continue
            elif '_space_group_symop_magn_operation.xyz' in line:
                in_symop = True
                continue
            elif '_space_group_symop_magn_centering.xyz' in line:
                in_centering = True
                continue
            
            if in_atom_site and line.strip() and not line.startswith('_'):
                parts = line.split()
                if len(parts) >= 5:  # At least 5 parts for basic atom info
                    data['atoms'].append(parts[0])
                    data['positions'].append([float(remove_uncertainty(x)) for x in parts[2:5]])
                    # Set occupancy to 1.0 if not present in the file
                    if has_occupancy and len(parts) >= 6:
                        data['occupancies'].append(float(remove_uncertainty(parts[5])))
                    else:
                        data['occupancies'].append(1.0)
                    # Initialize magnetic moment as zero
                    data['magnetic_moments'].append((parts[0], [0.0, 0.0, 0.0]))
            
            if in_moment and line.strip() and not line.startswith('_'):
                parts = line.split()
                if len(parts) >= 4:
                    label = parts[0]
                    moment = [float(remove_uncertainty(x)) for x in parts[1:4]]
                    # Update magnetic moment for this atom
                    for i, (atom_label, _) in enumerate(data['magnetic_moments']):
                        if atom_label == label:
                            data['magnetic_moments'][i] = (label, moment)
                            break
            
            if in_symop and line.strip() and not line.startswith('_'):
                parts = line.split()
                if len(parts) >= 2:
                    data['symops'].append(parts[1])
            
            if in_centering and line.strip() and not line.startswith('_'):
                parts = line.split()
                if len(parts) >= 2:
                    data['centering_ops'].append(parts[1])
        
        # Apply symmetry operations
        if data['symops']:
            data['atoms'], data['positions'], data['occupancies'], data['magnetic_moments'] = apply_symmetry_operations(
                data['atoms'], data['positions'], data['occupancies'], data['magnetic_moments'], data['symops'], is_op_on_spin=is_op_on_spin
            )
        
        # Apply magnetic centering operations
        if data['centering_ops']:
            data['atoms'], data['positions'], data['occupancies'], data['magnetic_moments'] = apply_centering_operations(
                data['atoms'], data['positions'], data['occupancies'], data['magnetic_moments'], data['centering_ops']
            )
        
        # Remove atoms that are too close to each other
        data['atoms'], data['positions'], data['occupancies'], data['magnetic_moments'] = remove_close_atoms(
            data['atoms'], data['positions'], data['occupancies'], data['magnetic_moments'], data['lattice']
        )
        
        # Verify that we have data
        if not data['atoms']:
            raise ValueError(f"No atoms found in {file_path}")
        if not data['positions']:
            raise ValueError(f"No positions found in {file_path}")
        
        return data
        
    except Exception as e:
        logging.error(f"Error reading {file_path}: {str(e)}")
        raise

def write_poscar(data, output_path):
    """Write POSCAR file with magnetic moments in the first line"""
    try:
        # Create magnetic moment string
        # Use list instead of dict to preserve order and duplicates
        magmom_list = []
        
        debug_print("\nWriting POSCAR with moments:")
        for i, atom in enumerate(data['atoms']):
            # Get moment directly from the moments list
            moment = data['magnetic_moments'][i][1]
            # Keep the original moment values with their signs
            moment_str = f"{moment[0]:+.2f} {moment[1]:+.2f} {moment[2]:+.2f}"
            magmom_list.append(moment_str)
            debug_print(f"Atom {atom}: {moment_str} with occupancy {data['occupancies'][i]}")
        
        # Count atoms of each type (using element symbols)
        atom_counts = {}
        for atom in data['atoms']:
            element = get_element_symbol(atom)
            atom_counts[element] = atom_counts.get(element, 0) + 1
        
        # Get unique elements in order of first appearance
        unique_elements = []
        seen_elements = set()
        for atom in data['atoms']:
            element = get_element_symbol(atom)
            if element not in seen_elements:
                unique_elements.append(element)
                seen_elements.add(element)
        
        # Write POSCAR
        with open(output_path, 'w') as f:
            # First line: magnetic moments
            f.write('MAGMOM = ' + ' '.join(magmom_list) + '\n')
            # Scale factor
            f.write('1.0\n')
            # Lattice vectors
            for vec in data['lattice']:
                f.write(f'{vec[0]:20.16f} {vec[1]:20.16f} {vec[2]:20.16f}\n')
            # Atom types (in order of first appearance)
            f.write(' '.join(unique_elements) + '\n')
            # Atom counts
            f.write(' '.join(str(atom_counts[e]) for e in unique_elements) + '\n')
            # Direct coordinates
            f.write('Direct\n')
            
            # Write positions in the original order
            for pos in data['positions']:
                f.write(f'{pos[0]:20.16f} {pos[1]:20.16f} {pos[2]:20.16f}\n')
    except Exception as e:
        logging.error(f"Error writing POSCAR: {str(e)}")
        raise

def process_mcif_file(mcif_path, output_dir, is_op_on_spin=False):
    """Process a single mcif file and generate POSCAR file"""
    try:
        logging.info(f"Processing {mcif_path}...")
        
        # Extract index from filename (e.g., "material_0.1.mcif" -> "0.1")
        index = Path(mcif_path).stem.split('_')[1]
        
        # Read mcif file
        data = read_mcif(mcif_path, is_op_on_spin=is_op_on_spin)
        
        # Write POSCAR file with magnetic moments
        output_path = os.path.join(output_dir, f"POSCAR_{index}.vasp")
        write_poscar(data, output_path)
        
        logging.info(f"Successfully generated {output_path}")
        return True
    except Exception as e:
        logging.error(f"Error processing {mcif_path}: {str(e)}")
        return False

def main(mcif_dir, output_dir, is_op_on_spin=False):
    # Example usage
    # mcif_dir = 'commensurate'  # Directory containing mcif files
    # output_dir = 'poscar_op_on_spin' if is_op_on_spin else 'poscar_op_not_on_spin'  # Base directory for VASP input files
    stopping_index = 20000
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of all mcif files
    mcif_files = list(Path(mcif_dir).glob('*.mcif'))
    if stopping_index:
        mcif_files = mcif_files[:stopping_index]
    
    # Process files in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Create a list of futures
        futures = [executor.submit(process_mcif_file, str(mcif_file), output_dir, is_op_on_spin) 
                  for mcif_file in mcif_files]
        
        # Use tqdm to show progress
        for future in tqdm(concurrent.futures.as_completed(futures), 
                         total=len(futures), 
                         desc="Processing mcif files"):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Task failed: {str(e)}")

def test_conversion(entry="0.747", is_op_on_spin=False):
    """Test the conversion algorithm with material_{entry}.mcif"""
    test_file = f'commensurate/material_{entry}.mcif'
    output_dir = 'test_output'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Process the test file
        process_mcif_file(test_file, output_dir, is_op_on_spin=is_op_on_spin)
        
        # Read the generated POSCAR file
        output_path = os.path.join(output_dir, f'POSCAR_{entry}.vasp')
        with open(output_path, 'r') as f:
            content = f.read()
            debug_print("\nGenerated POSCAR content:")
            debug_print(content)
            
    except Exception as e:
        logging.error(f"Test failed: {str(e)}")

if __name__ == "__main__":
    main(mcif_dir='commensurate', output_dir='poscar_op_on_spin', is_op_on_spin=True) 
    debugfile.close()