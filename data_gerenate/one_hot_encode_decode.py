import numpy as np

"""
# Example usage
rna_sequences = ["CAGGGUCAGGAUCCGCGUGCUUGCUAGUCCACCUUACCGCCGGAUGGGCAGCCGCUGCUGAGCGCCUGCUAGGUGGUCCGACGCGCAGCUCCGAGCUGUCCUGGACUGGCGCUUUUUAGGAGCGAAGGGGAACCCCGCAGGAGACCAGGGCCCUGAACUCAGGGGCUUCGCCACUGAUUGUCCAAACGCAAUUCUUGUACGAGUCUGCGGCCAACCGAGAAUUGUGGCUGGACAUCUGUGGCUGAGCUCCGGGCGCAACAGGGGCGGGGGCCCCAGGGACAGGGCUCAGCGCGGGCGAGACCUCUCGGGGCGGCGGCUGGCUGGCCCCAGCGCGAGGAUCGCGGUCCCGGCCCGCGCGCACAGAGACGCCGGUCCCUGCCACCAGCGCCGCCAUCCGCAU",
                 "AAGCGUUCAAGCUCAACACCCACUACCUAAAAAAUCCCAAACAUAUAACUGAACUCCUCACACCCAAUUGGACCAAUCUAUCACCCUAUAGAAGAACUAAUGUUAGUAUAAGUAACAUGAAAACAUUCUCCUCCGCAUAAGCCUGCGUCAGAUUAAAACACUGAACUGACAAUUAACAGCCCAAUAUCUACAAUCAACCAACAAGUCAUUAUUACCCUCACUGUCAACCCAACACAGGCAUGCUCAUAAGGAAAGGUUAAAAAAAGUAAAAGGAACUCGGCAAAUCUUACCCCGCCUGUUUACCAAAAACAUCACCUCUAGCAUCACCAGUAUUAGAGGCACCGCCUGCCCAGUGACACAUGUUUAACGGCCGCGGUACCCUAACCGUGCAAAGGUAGCA"]
max_length = 400

# 1. Convert RNA sequences to one-hot encoding
one_hot_encoded = convert_one_hot(rna_sequences, max_length=max_length)
print("One-hot encoding result:")
print(one_hot_encoded)

# 2. Decode back to RNA sequences
decoded_sequences = [decodeRNA(seq) for seq in one_hot_encoded]
print("\nDecoded RNA sequences:")
print(decoded_sequences)

"""
# seq_4
def convert_one_hot_seq_4(sequence, max_length):
    # Assuming sequence is a single RNA sequence string
    seq = sequence.upper()  # Convert to uppercase
    seq_length = len(seq)
    
    if seq_length > max_length:
        seq = seq[:max_length]
        seq_length = max_length
    
    # Create a zero matrix with 4 rows and seq_length columns
    one_hot = np.zeros((4, seq_length))  # 4 channels, corresponding to A, C, G, U

    # Assign one-hot encoding for each nucleotide
    index = [j for j in range(seq_length) if seq[j] == 'A']
    one_hot[0, index] = 1
    index = [j for j in range(seq_length) if seq[j] == 'C']
    one_hot[1, index] = 1
    index = [j for j in range(seq_length) if seq[j] == 'G']
    one_hot[2, index] = 1
    index = [j for j in range(seq_length) if seq[j] == 'U']  # U is used instead of T in RNA
    one_hot[3, index] = 1

    # If max_length is provided, apply zero-padding
    if max_length:
        offset1 = int((max_length - seq_length) / 2)
        offset2 = max_length - seq_length - offset1
        if offset1:
            one_hot = np.hstack([np.zeros((4, offset1)), one_hot])
        if offset2:
            one_hot = np.hstack([one_hot, np.zeros((4, offset2))])

    return one_hot  # Return a single (4, max_length) matrix
# seq_4
def decode_seq_4(m):
    na = ["A", "C", "G", "U"]  # Nucleotides in RNA
    seq = ""
    for i in range(m.shape[1]):  # Traverse each column
        # Find the index of the row with value 1 (corresponding nucleotide position)
        var = np.where(m[:, i] == 1)[0]
        if len(var) == 1:  # Ensure a unique match is found
            seq = seq + na[var[0]]  # Add the corresponding nucleotide to the sequence
    return seq

"""

# Example usage
sequences = ["UUUUUPPPPPUUUUUUUUPPUPPPPPPPUUPPPPUPPPPPPUUPPUPPPPUUUUUPPPPPPPPPPUUUPPPPPUUUUUUPPPPPUUUPPPPUUUUUPPPPPPUUUPPPPUPPUUUUPPPPPPUPPPPUPPPPUPPPPPPUUUUUUUPUUUUUUUUPUUUUUUPPPPPPUPPPPPPPPPPUUPPPPPPPUUUUPPPPPPPUUPPPPPPUUUUUPPPPPPPPUUUUUPPPPPPPPPPPPPPPUPPPPPPPUPPPPUPPUUUUUUUUUUUUPPUPPPPPPPPPPPPPPUUUUUUUPPPPPPPUUUUUUUUUUUUUPPPPPPPPPPUUUUUUUUPPPPPUUUUUUUUUUPPPPPUUUUUUUPPPPPPUUUPPPPPUUUUUUUUUUPPPPPPPPPPPPPPPPPPU",
             "UUPPPPPPPPPPPPPUUUUUUUPPPPUPPPUUPPPPPPUPPPUUPPPPPPPPPPPUUUUUPPPPPPPUPPPUUUPPPUPPPPUPPPPPPPPPPPUUPPPPUPPUUPPPPPUUUPPPPPPPPPPUUUUPPPPPPPPPPPUUPPPPPPUUUUPPPPPPUUUUUUUUPPPPPPPPPPPPPPPPPPPPPPUUUPPPPPPPPPPUUUUUUPPPPPPUUUUPPPPPPUUUUUUUUUUUUUUUUUUUUUUUUUUUUUPPPPPPPPPUPPPPPPPUUUUPPPPPPPPPPPPUUUPPPPPPPPUPPPUPPPUUUPPPPUUUUUPPPPPPPPUPPPPPPPPPPUPPPUUUPPPUUPPPPPUUUUUUUUUUPPPPPUUPPPUPPPUPPPPPPPPPPUPPPPPPPUUPPPUU"
             ]
max_length = 400  # Assuming max length is 6

# 1. Convert sequences to one-hot encoding
one_hot_encoded = convert_up_one_hot(sequences, max_length=max_length)
print("One-hot encoding result:")
print(one_hot_encoded)

# 2. Decode back to original sequences
decoded_sequences = [decode_up(seq) for seq in one_hot_encoded]
print("\nDecoded sequences:")
print(decoded_sequences)

"""
# str_2
def convert_one_hot_str_2(sequence, max_length):
    # Assuming sequence is a single string
    seq = sequence.upper()  # Convert to uppercase
    seq_length = len(seq)

    if seq_length > max_length:
        seq = seq[:max_length]
        seq_length = max_length

    # Create a zero matrix with 2 rows and seq_length columns
    one_hot = np.zeros((2, seq_length))  # 2 channels, corresponding to U and P

    # Assign one-hot encoding for each character
    index_u = [j for j in range(seq_length) if seq[j] == 'U']
    one_hot[0, index_u] = 1  # 'U' corresponds to [1, 0]
    
    index_p = [j for j in range(seq_length) if seq[j] == 'P']
    one_hot[1, index_p] = 1  # 'P' corresponds to [0, 1]

    # Apply zero-padding if max_length is provided
    if max_length:
        offset1 = int((max_length - seq_length) / 2)
        offset2 = max_length - seq_length - offset1
        if offset1:
            one_hot = np.hstack([np.zeros((2, offset1)), one_hot])  # Left padding
        if offset2:
            one_hot = np.hstack([one_hot, np.zeros((2, offset2))])  # Right padding

    return one_hot  # Return a single (2, max_length) matrix
# str_2
def decode_str_2(m):
    seq = ""
    for i in range(m.shape[1]):  # Traverse each column
        # Find the channel with value 1
        if m[0, i] == 1:
            seq += 'U'
        elif m[1, i] == 1:
            seq += 'P'
    return seq


"""
# Input sequences
sequences = ["PLUMPULMMULP" , "PPP"]
max_length = 14  # Assuming max length is 6

# 1. Convert sequences to one-hot encoding
one_hot_encoded = convert_plum_one_hot(sequences, max_length=max_length)
print("One-hot encoding result:")
print(one_hot_encoded)

# 2. Decode back to original sequences
decoded_sequences = [decode_plum(seq) for seq in one_hot_encoded]
print("\nDecoded sequences:")
print(decoded_sequences)
"""
# str_4
def convert_one_hot_str_4(sequence, max_length):
    # Mapping letters to one-hot encoding
    letter_to_index = {'P': 0, 'L': 1, 'U': 2, 'M': 3}
    
    # Assuming sequence is a single string
    seq = sequence.upper()  # Convert to uppercase
    seq_length = len(seq)

    if seq_length > max_length:
        seq = seq[:max_length]
        seq_length = max_length
    
    one_hot = np.zeros((4, seq_length))  # 4 channels, corresponding to P, L, U, M
    
    # Assign one-hot encoding for each letter
    for i, letter in enumerate(seq):
        if letter in letter_to_index:
            one_hot[letter_to_index[letter], i] = 1  # Select the corresponding channel and set to 1

    # Apply zero-padding if max_length is provided
    if max_length:
        offset1 = int((max_length - seq_length) / 2)
        offset2 = max_length - seq_length - offset1
        if offset1:
            one_hot = np.hstack([np.zeros((4, offset1)), one_hot])  # Left padding
        if offset2:
            one_hot = np.hstack([one_hot, np.zeros((4, offset2))])  # Right padding

    return one_hot  # Return a single (4, max_length) matrix
# str_4
def decode_str_4(m):
    index_to_letter = {0: 'P', 1: 'L', 2: 'U', 3: 'M'}
    seq = ""
    for i in range(m.shape[1]):  # Traverse each column
        # Find the channel with value 1
        var = np.where(m[:, i] == 1)[0]
        if len(var) == 1:  # Ensure a unique match is found
            seq += index_to_letter[var[0]]  # Add the corresponding letter to the sequence
    return seq

"""
# Input sequences
sequences = ["BHTLM", "RHTLM", "BEEE", "RRR"]
max_length = 6  # Assuming max length is 6

# 1. Convert sequences to one-hot encoding
one_hot_encoded = convert_behmlrt_one_hot(sequences, max_length=max_length)
print("One-hot encoding result:")
print(one_hot_encoded)

# 2. Decode back to original sequences
decoded_sequences = [decode_behmlrt(seq) for seq in one_hot_encoded]
print("\nDecoded sequences:")
print(decoded_sequences)
"""
# str_7
def convert_one_hot_str_7(sequence, max_length):
    # Mapping letters to one-hot encoding
    letter_to_index = {'B': 0, 'E': 1, 'H': 2, 'L': 3, 'M': 4, 'R': 5, 'T': 6}
    
    # Assuming sequence is a single string
    seq = sequence.upper()  # Convert to uppercase
    seq_length = len(seq)
    if seq_length > max_length:
        seq = seq[:max_length]
        seq_length = max_length
    # Create a 7 row, seq_length column zero matrix
    one_hot = np.zeros((7, seq_length))  # 7 channels, corresponding to B, E, H, L, M, R, T
    
    # Assign one-hot encoding for each letter
    for i, letter in enumerate(seq):
        if letter in letter_to_index:
            one_hot[letter_to_index[letter], i] = 1  # Select the corresponding channel and set to 1

    # Apply zero-padding if max_length is provided
    if max_length:
        offset1 = int((max_length - seq_length) / 2)
        offset2 = max_length - seq_length - offset1
        if offset1:
            one_hot = np.hstack([np.zeros((7, offset1)), one_hot])  # Left padding
        if offset2:
            one_hot = np.hstack([one_hot, np.zeros((7, offset2))])  # Right padding

    return one_hot  # Return a single (7, max_length) matrix
# str_7
def decode_str_7(m):
    index_to_letter = {0: 'B', 1: 'E', 2: 'H', 3: 'L', 4: 'M', 5: 'R', 6: 'T'}
    seq = ""
    for i in range(m.shape[1]):  # Traverse each column
        # Find the channel with value 1
        var = np.where(m[:, i] == 1)[0]
        if len(var) == 1:  # Ensure a unique match is found
            seq += index_to_letter[var[0]]  # Add the corresponding letter to the sequence
    return seq

# seq_str_8
def convert_one_hot_seq_str_8(sequence, max_length):
    # Mapping letters to one-hot encoding
    letter_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7}
    
    # Assuming sequence is a single string
    seq = sequence.upper()  # Convert to uppercase
    seq_length = len(seq)
    if seq_length > max_length:
        seq = seq[:max_length]
        seq_length = max_length
    # Create an 8 row, seq_length column zero matrix
    one_hot = np.zeros((8, seq_length))  # 8 channels, corresponding to A, B, C, D, E, F, G, H
    
    # Assign one-hot encoding for each letter
    for i, letter in enumerate(seq):
        if letter in letter_to_index:
            one_hot[letter_to_index[letter], i] = 1  # Select the corresponding channel and set to 1

    # Apply zero-padding if max_length is provided
    if max_length:
        offset1 = int((max_length - seq_length) / 2)
        offset2 = max_length - seq_length - offset1
        if offset1:
            one_hot = np.hstack([np.zeros((8, offset1)), one_hot])  # Left padding
        if offset2:
            one_hot = np.hstack([one_hot, np.zeros((8, offset2))])  # Right padding

    return one_hot  # Return a single (8, max_length) matrix
# seq_str_8
def decode_seq_str_8(m):
    index_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H'}
    seq = ""
    for i in range(m.shape[1]):  # Traverse each column
        # Find the channel with value 1
        var = np.where(m[:, i] == 1)[0]
        if len(var) == 1:  # Ensure a unique match is found
            seq += index_to_letter[var[0]]  # Add the corresponding letter to the sequence
    return seq

"""
# Input sequences
sequences = ["ABCD", "EFGH", "IJKL", "MNOP"]
max_length = 8  # Assuming max length is 8

# 1. Convert sequences to one-hot encoding
one_hot_encoded = convert_behmlrt_16_one_hot(sequences, max_length=max_length)
print("One-hot encoding result:")
print(one_hot_encoded)

# 2. Decode back to original sequences
decoded_sequences = [decode_behmlrt_16(seq) for seq in one_hot_encoded]
print("\nDecoded sequences:")
print(decoded_sequences)
"""
# seq_str_16
def convert_one_hot_seq_str_16(sequence, max_length):
    # Mapping letters to one-hot encoding
    letter_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 
                       'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15}
    
    # Process a single sequence
    seq = sequence.upper()  # Convert to uppercase
    seq_length = len(seq)

    if seq_length > max_length:
        seq = seq[:max_length]
        seq_length = max_length
    
    one_hot = np.zeros((16, seq_length))  # Create a 16 channel matrix
    
    # Assign one-hot encoding for each letter
    for i, letter in enumerate(seq):
        if letter in letter_to_index:
            one_hot[letter_to_index[letter], i] = 1  # Select the corresponding channel and set to 1

    # Apply zero-padding if max_length is provided
    if max_length:
        offset1 = int((max_length - seq_length) / 2)
        offset2 = max_length - seq_length - offset1
        if offset1:
            one_hot = np.hstack([np.zeros((16, offset1)), one_hot])  # Left padding
        if offset2:
            one_hot = np.hstack([one_hot, np.zeros((16, offset2))])  # Right padding

    return one_hot  # Return a (16, max_length) matrix
# seq_str_16
def decode_seq_str_16(m):
    index_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
                       8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P'}
    seq = ""
    for i in range(m.shape[1]):  # Traverse each column
        # Find the channel with value 1
        var = np.argmax(m[:, i])  # Find the index of the maximum value to avoid np.where performance issue
        seq += index_to_letter[var]  # Add the corresponding letter to the sequence

    return seq


def combine_one_hot_matrix(matrices, axis=0):
    combined_matrix = np.concatenate(matrices, axis=axis)
    # print(f"Combined shape: {combined_matrix.shape}")
    return combined_matrix