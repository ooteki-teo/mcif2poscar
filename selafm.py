import os
import shutil
import numpy as np

def is_antiferromagnetic(magmom_line):
    # Split the MAGMOM line into individual values
    magmoms = [float(x) for x in magmom_line.strip().split('=')[1].strip().split()]
    
    # Group into (mx, my, mz) triplets
    triplets = np.array(magmoms).reshape(-1, 3)
    
    # Calculate sum of magnetic moments
    total_magmom = np.sum(triplets, axis=0)
    # print(total_magmom)
    
    # Check if sum is close to zero (using small tolerance for floating point comparison)
    return np.allclose(total_magmom, [0, 0, 0], atol=0.1), total_magmom

def process_poscar_files(poscar_dir, output_dir):
    # Create AFMposcar directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open a file to write results
    with open('magnetic_analysis.txt', 'w') as result_file:
        result_file.write("Filename\tTotal Magnetic Moment (mx, my, mz)\tIs AFM\n")
        result_file.write("-" * 80 + "\n")
    
        # Process all POSCAR files in poscar2 directory
        for filename in os.listdir(poscar_dir):
            if filename.startswith('POSCAR_') and filename.endswith('.vasp'):
                filepath = os.path.join(poscar_dir, filename)
                
                # Read the first line of the POSCAR file
                with open(filepath, 'r') as f:
                    first_line = f.readline().strip()
                
                # Check if it's antiferromagnetic
                is_afm, total_magmom = is_antiferromagnetic(first_line)
                
                # Write results to file
                result_file.write(f"{filename}\t{total_magmom}\t{'Yes' if is_afm else 'No'}\n")
                
                # If antiferromagnetic, copy to AFMposcar directory
                if is_afm:
                    shutil.copy2(filepath, os.path.join(output_dir, filename))
                    print(f"Copied {filename} to AFMposcar directory")

if __name__ == "__main__":
    process_poscar_files(poscar_dir='poscar_op_on_spin', output_dir='AFMposcar')
