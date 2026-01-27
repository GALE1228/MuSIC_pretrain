import subprocess
import tempfile
import argparse
import os
import shutil
from tqdm import tqdm
import threading
import time

import glob


def monitor_folding_progress(output_file, total_sequences, pbar):
    """
    Monitor RNAfold folding progress, no longer relies on .ps files, but counts the number of '>' headers in output_file.

    Parameters:
        output_file (str): The path to the RNAfold result file.
        total_sequences (int): The total number of input RNA sequences.
        pbar (tqdm): Progress bar object.
    """
    folded_rna_count = 0
    while folded_rna_count < total_sequences:
        # Count the number of ">" in the output_file
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                folded_rna_count = sum(1 for line in f if line.startswith(">"))

        pbar.update(folded_rna_count - pbar.n)  # Update the progress bar
        time.sleep(1)  # Check every 1 second


def run_rnafold(data_path, data_type, rbp_name, tt):
    """
    Handle RNAfold computation, similar to the RNAfold call in a Bash script, outputs the RNAfold result text.

    Parameters:
        data_path (str): The path where data is stored.
        data_type (str): The type of data (e.g., training data, test data).
        rbp_name (str): The RBP name.
        tt (str): The input file name (without the extension).
    """
    # RNAfold command path
    RNAfold = "RNAfold"

    # Input file path
    input_file_path = os.path.join(data_path, rbp_name, f"{data_type}_data", f"{tt}.fa")

    # Color codes
    GREEN = "\033[32m"
    RESET = "\033[0m"
    print(f"Start using RNAfold folding {GREEN}{input_file_path}{RESET}")

    # Create target directory
    target_directory = os.path.join(data_path, "RNAfold_results", rbp_name)
    os.makedirs(target_directory, exist_ok=True)

    # Count the number of sequences in the input file
    with open(input_file_path, "r") as infile:
        total_sequences = sum(1 for line in infile if line.startswith(">"))

    # Result file path
    output_file = os.path.join(target_directory, f"{tt}_fold.result")

    print(f"Total sequences: {total_sequences}")
    print(f"Folding sequences...")

    # Use tqdm to display the progress bar
    with tqdm(total=total_sequences, desc="Folding RNA", unit="seq") as pbar:
        # Start a monitoring thread to track the number of ">" lines in output_file
        monitor_thread = threading.Thread(target=monitor_folding_progress, args=(output_file, total_sequences, pbar))
        monitor_thread.daemon = True
        monitor_thread.start()

        # Define a function to periodically delete .ps files
        def delete_ps_files_periodically():
            while True:
                for file_path in glob.iglob(os.path.join(target_directory, "*.ps")):
                    try:
                        os.remove(file_path)
                    except Exception:
                        # Ignore errors during file deletion
                        pass
                # Execute every 2 seconds
                time.sleep(2)

        # Start a thread to periodically delete .ps files
        ps_deletion_thread = threading.Thread(target=delete_ps_files_periodically)
        ps_deletion_thread.daemon = True
        ps_deletion_thread.start()

        # Execute RNAfold computation
        with open(input_file_path, "r") as infile, open(output_file, "w") as outfile:
            subprocess.run(
                [RNAfold, "-p", "--noPS"],
                stdin=infile,
                stdout=outfile,
                cwd=target_directory
            )

        monitor_thread.join()  # Wait for the monitoring thread to finish
        # Wait a little longer to ensure the deletion thread completes the final deletion
        time.sleep(2)

    # Output statistics
    print(f"Folding complete: {total_sequences}/{total_sequences} sequences folded.")
    print(f"Folding complete. Result saved to: {output_file}")

    return output_file

def run_infer_rnafold(fasta_filepath):
    """
    Handle RNAfold computation, similar to the RNAfold call in a Bash script, outputs the RNAfold result text.

    Parameters:
        fasta_filepath: The path to the fasta file.
    """
    RNAfold = "RNAfold"
    GREEN = "\033[32m"
    RESET = "\033[0m"

    print(f"Start using RNAfold folding {GREEN}{fasta_filepath}{RESET}")

    target_directory = os.path.dirname(fasta_filepath)
    os.makedirs(target_directory, exist_ok=True)

    # Generate the RNAfold result file path
    output_file = os.path.splitext(fasta_filepath)[0] + "_fold.result"

    # Count the number of sequences in the input file
    with open(fasta_filepath, "r") as infile:
        total_sequences = sum(1 for line in infile if line.startswith(">"))

    print(f"Total sequences to fold: {total_sequences}")

    # Use tqdm to display the progress bar
    with tqdm(total=total_sequences, desc="Folding RNA", unit="seq") as pbar:
        # Start a monitoring thread to track the number of ">" lines in output_file
        monitor_thread = threading.Thread(target=monitor_folding_progress, args=(output_file, total_sequences, pbar))
        monitor_thread.daemon = True
        monitor_thread.start()

        # Start a new thread to periodically delete .ps files
        def delete_ps_files_periodically():
            while True:
                for file_path in glob.iglob(os.path.join(target_directory, "*.ps")):
                    try:
                        os.remove(file_path)
                    except Exception:
                        # Ignore errors during file deletion
                        pass
                # Execute every 2 seconds
                time.sleep(2)

        ps_deletion_thread = threading.Thread(target=delete_ps_files_periodically)
        ps_deletion_thread.daemon = True
        ps_deletion_thread.start()

        # Execute RNAfold computation
        with open(fasta_filepath, "r") as infile, open(output_file, "w") as outfile:
            subprocess.run(
                [RNAfold, "-p", "--noPS"],
                stdin=infile,
                stdout=outfile,
                cwd=target_directory
            )

        monitor_thread.join()  # Wait for the monitoring thread to finish
        # Although it's a daemon thread, we let the main thread wait a bit for the deletion thread to execute the final deletion
        time.sleep(2)

    print(f"Folding complete: {total_sequences}/{total_sequences} sequences folded.")
    print(f"Folding complete. Result saved to: {output_file}")

    return output_file

    
def annotate_str(mfe_structure_line):
    """
    Annotate a single MFE structure line, save it to a temporary file and call a C program to process it, 
    then extract the annotation result.

    Parameters:
        mfe_structure_line (str): The MFE structure in dot-bracket format.

    Returns:
        str: The annotated string (C program output).
    """
    try:
        # Use tempfile to create temporary files
        with tempfile.NamedTemporaryFile(delete=True, mode="w") as structure_file, \
             tempfile.NamedTemporaryFile(delete=True, mode="r") as annotation_file:

            # Save the MFE structure to the temporary file
            structure_file.write(mfe_structure_line + "\n")
            structure_file.flush()  # Ensure the file is written

            # Call C program to annotate the temporary file
            c_program = "./data_gerenate/fold_str_annotion/parse_secondary_structure_v2"  # Path to C program
            subprocess.run(
                [c_program, structure_file.name, annotation_file.name],
                check=True)
            # Read results from annotation file
            annotated_structure = annotation_file.read().strip()

        return annotated_structure

    except subprocess.CalledProcessError as e:
        print(f"Error running C program: {e}")
        return None

    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

def process_rnafold_and_annotate(input_file, output_file):
    """
    Extract the 1st line (header), 2nd line (RNA sequence), and 3rd line (MFE structure) from RNAfold output file,
    annotate the structure, convert it to a 2-letter format, and save the results in TSV format.

    Parameters:
        input_file (str): The input RNAfold output file path.
        output_file (str): The output processed result file path (should end with .tsv).
    """
    # Define 7-letter to 4-letter and 4-letter to 2-letter structure conversion dictionaries
    struct_7_convert_4 = {'B': 'M', 'E': 'U', 'H': 'L', 'L': 'P', 'M': 'M', 'R': 'P', 'T': 'M'}
    struct_4_convert_2 = {'P': 'P', 'L': 'U', 'U': 'U', 'M': 'U'}

    # Check if output directory exists, if not, create it
    dirname = os.path.dirname(output_file)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Open the input file and process each RNA record block
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        lines = infile.readlines()

        # Each RNA record block consists of 6 lines
        for i in range(0, len(lines), 6):
            # Ensure there are enough lines in the current block
            if i + 2 < len(lines):
                header_line = lines[i].strip()  # 1st line
                sequence_line = lines[i + 1].strip()  # 2nd line
                mfe_structure_line = lines[i + 2].strip().split()[0]  # 3rd line, discard energy value

                # Annotate the structure (call annotation function)
                annotated_structure = annotate_str(mfe_structure_line)

                # Convert structure: 7-letter -> 4-letter -> 2-letter
                struct_4 = []
                struct_2 = []

                for i in range(len(annotated_structure)):
                    # Convert the 7-letter structure to 4-letter and then to 2-letter
                    struct_4.append(struct_7_convert_4[annotated_structure[i]])
                    struct_2.append(struct_4_convert_2[struct_7_convert_4[annotated_structure[i]]])

                # Join the results into strings
                struct_4 = ''.join(struct_4)
                struct_2 = ''.join(struct_2)

                # Write the final results: header, sequence, original structure, annotated structure
                outfile.write(f"{header_line}\t{sequence_line}\t{struct_2}\n")

    RED = "\033[31m"
    RESET = "\033[0m"

    print(f"Sequence and Structure Annotation Files {RED}{output_file}{RESET} has been saved.")
    
    os.remove(input_file)