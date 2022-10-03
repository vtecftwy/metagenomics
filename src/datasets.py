import tensorflow as tf



def strings_to_tensors(b):
    """Convert batch of strings into three tensors: (x_seqs, (y_labels, y_pos))"""
    
    # Split the string in three : returns a ragged tensor which needs to be converted into a normal tensor using .to_tensor()
    t = tf.strings.split(b, '\t').to_tensor(default_value = '', shape=[None, 3])

    # Split each sequence string into a list of single base strings:
    # 'TCAAAATAATCA' -> ['T','C','A','A','A','A','T','A','A','T','C','A']
    seqs = tf.strings.bytes_split(t[:, 0]).to_tensor(shape=(None, 50))


    # BHE sequences
    # Each base letter (A, C, G, T, N) is replaced by a OHE vector
    #     "A" converted into [1,0,0,0,0]
    #     "C" converted into [0,1,0,0,0]
    #     "G" converted into [0,0,1,0,0]
    #     "T" converted into [0,0,0,1,0]
    #     "N" converted into [0,0,0,0,1]
    # 
    # Technical Notes:
    # a. The batch of sequence `seqs` has a shape (batch_size, 50) after splitting each byte. 
    #    Must flatten it first, then apply the transform on each base, then reshape to original shape
    # b. We need to map each letter to one vector/tensor. 
    #    1. Cast bytes seqs into integer sequence (uint8 to work byte by byte)
    #    2. For each base letter (A, C, G, T, N) create one tensor (batch_size, 50) (seqs_A, _C, _G, _T, _N)
    #    3. Value is 1 if it is the base in the sequence, otherwise 0
    #    4. Concatenate these 5 tensors into a tensor of shape (batch_size, 50, 5)
 
    seqs_uint8 = tf.io.decode_raw(seqs, out_type=tf.uint8)
    # note: tf.io.decode_raw adds one dimension at the end in the process
    #       [b'C', b'A', b'T'] will return [[67], [65], [84]] and not [67, 65, 84]
    #       this is actually what we want to contatenate the values for each base letter

    A, C, G, T, N = 65, 67, 71, 84, 78

    seqs_A = tf.cast(seqs_uint8 == A, tf.float32)
    seqs_C = tf.cast(seqs_uint8 == C, tf.float32)
    seqs_G = tf.cast(seqs_uint8 == G, tf.float32)
    seqs_T = tf.cast(seqs_uint8 == T, tf.float32)
    seqs_N = tf.cast(seqs_uint8 == N , tf.float32)

    x_seqs = tf.concat([seqs_A, seqs_C, seqs_G, seqs_T, seqs_N], axis=2)

    # OHE labels
    n_labels = 187
    y_labels = tf.strings.to_number(t[:, 1], out_type=tf.int32)
    y_labels = tf.gather(tf.eye(n_labels), y_labels)

    # OHE positions
    n_pos = 10
    y_pos = tf.strings.to_number(t[:, 2], out_type=tf.int32)
    y_pos= tf.gather(tf.eye(n_pos), y_pos)

    return (x_seqs, (y_labels, y_pos))