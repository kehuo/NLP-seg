# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.layers import Layer, InputSpec
from keras import backend as K
from keras.layers import Layer, InputSpec
import tensorflow as tf
import numpy as np


#
# Transition Score Layer
# The output shape is (tag_set_size, tag_set_size)
#
class TS(Layer):

    def __init__(self, size, **kwargs):
        self.size = size
        super(TS, self).__init__(**kwargs)


    def build(self, input_shape):
        # transition score matrix
        # A[i,i+1]
        self.trans = self.add_weight(name='transition',
                                     shape=(self.size+1, self.size),
                                     initializer=initTS,
                                     trainable=True)
        super(TS, self).build(input_shape)  # Be sure to call this somewhere!


    # return the "global" transition scores
    # "x" is left to avoid Keras errors, it's not used
    def call(self, x):
        return self.trans

    def compute_output_shape(self, input_shape):
        return (self.size+1, self.size)


# Initializer for TS
def initTS(shape, dtype=None):
     # ts = np.random.normal(loc=0.0, scale=0.05, size=shape)
     ts = np.zeros(shape)
     return K.constant(ts, dtype=dtype)


# CRF - Given 
# P - a sequence of emission scores that corresponds to P(Yi|Xi)
# A - a matrix of transition scores that corresponds to A(Yi+1|Yi)
# Y - The correct sequence Y during the training time
# L - The actual length of the X and Y 
# We score each possible tag sequence and compute the P(Y|X) with softmax
# Score Y = sum(P(Yi|Xi) + A(Yi|Yi-1))
class CRF(Layer):

    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size
        self.seq_len = 0
        self.tag_set_size = 0
        super(CRF, self).__init__(**kwargs)

    # input_shape is an array of [P, A, Y, L]'s shape
    def build(self, input_shape):
        self.seq_len = input_shape[0][1]
        self.tag_set_size = input_shape[0][2]
        super(CRF, self).build(input_shape)  # Be sure to call this somewhere!

    # Forward propagation
    # Compute the cross-entropy loss for tag sequence Y
    # Let S[k,i] be the total score for sequences of len k that ends with tag i
    # We have S[k,i] = log_sum_exp(S[k-1,j] + A[j,i] + P[k]
    #         S[k]   = log_sum_exp(S[k,i])
    #
    # Assume T (size of the tag set) = 3, k = 5
    #        S[5,1]	 	       S[5,2]		       S[5,3]
    # S[4,1]+A[1,1]+P[5,1]	S[4,1]+A[1,2]+P[5,2]	S[4,1]+A[1,3]+P[5,3]
    # S[4,2]+A[2,1]+P[5,1]	S[4,2]+A[2,2]+P[5,2]	S[4,2]+A[2,3]+P[5,3]
    # S[4,3]+A[3,1]+P[5,1]	S[4,3]+A[3,2]+P[5,2]	S[4,3]+A[3,3]+P[5,3]
    #
    def call(self, input):
        # P's shape is (batch_size, seq_len, tag_set_size)
        # A's shape is (tag_set_size+1, tag_set_size)
        # Y's shape is (batch_size, seq_len, 1)
        # L's shape is (batch_size, )
        [P, A, Y, L] = input

        # A's score is invariant w.r.t. the batch and sequence
        # Extend the A to [batch_size, tag_size+1, tag_size] tensors
        A_mat = K.expand_dims(A, axis=0)
        A_k = K.repeat_elements(A_mat, self.batch_size, axis=0)

        # S is the emission + transition 
        # S is an array of seq_len, each of shape (batch_size, tag_set_size)
        # For each sample, S[0] = P[0,] + A[-1,]
        S = []
        S.append(P[:,0,] + A_k[:,-1,])

        # Prepare score for Y, the corresponding tag sequence for X
        # Y is an array of seq_len, each of shape (batch_size, )
        # Thus score_Y[0] = P[0,Y[0]] + A[-1,Y[0]]
        score_Y = []
        score_P0 = self.P_score(P[:,0,], Y[:,0,0])
        const_n1 = K.constant(self.tag_set_size, shape=(self.batch_size, ), dtype='int32')
        score_A0 = self.A_score(A, const_n1, Y[:,0,0])
        score_Y.append(score_P0 + score_A0)

        # For each sample, Loss is computed for all lengths.
        # Right loss is picked for each sample at the end based on its length.
        losses = []
        losses.append(K.logsumexp(S[0], axis=1) - score_Y[0])

        # Compute the S timestep-by-timestep
        # Extend the P, S to [batch_size, tag_size, tag_size] tensors
        for k in range(1, self.seq_len):
            P_mat = K.expand_dims(P[:,k,], axis=1)
            P_k = K.repeat_elements(P_mat, self.tag_set_size, axis=1)

            S_mat = K.expand_dims(S[k-1], axis=2)
            S_k_1 = K.repeat_elements(S_mat, self.tag_set_size, axis=2)

            S_k = P_k + A_k[:,:-1,] + S_k_1
            S_k_r = K.logsumexp(S_k, axis=1)
            S.append(S_k_r)

            score_P = self.P_score(P[:,k,], Y[:,k,0])
            score_A = self.A_score(A, Y[:,k-1,0], Y[:,k,0])
            score_Yk = score_Y[k-1] + score_P + score_A
            score_Y.append(score_Yk)

            # cross-entropy loss
            loss_k = K.logsumexp(S_k_r, axis=1) - score_Yk
            losses.append(loss_k)

        # Pick the right loss for each sample based on its length L
        loss_b = self.batch_loss(losses, L)
        return loss_b

    # From P_t which is a tensor of shape [batch_size, tag_set_size]
    # and index_t which is a tensor of shape [batch_size, ]
    # Return P_t[:,index_t], which is a tensor of shape [batch_size, ]
    def P_score(self, P_t, index_t):
        n = self.batch_size
        m = self.tag_set_size
        # first convert tensor P into 1-d shape
        P1d_t = K.flatten(P_t)
        # next convert index into 1-d index, beware the reshape for index_t
        i_t = K.arange(n) * m + index_t
        # finally get the tensor
        r_t = K.gather(P1d_t, i_t)
        return r_t

    # From A_t which is a matrix of shape [tag_set_size+1, tag_set_size]
    # Return the A_t[m, n], where m and n are a tensor array of indexes
    # m and n are [batch_size, ]
    def A_score(self, A_t, m, n):
        # First convert tensor A into a 1-d shape
        A1d_t = K.flatten(A_t)
        # Then index them
        r_t = K.gather(A1d_t, m * self.tag_set_size + n)
        return r_t

    # From losses which is an array[seq_len] of tensors [batch_size, ]
    # Pick the loss value for each sample at position L [batch_size, ]
    # Return the resulting loss tensor of shape [batch_size, ]
    def batch_loss(self, losses, L):
        n = self.batch_size
        m = self.seq_len
        loss_t = K.concatenate(losses)
        l1d_t = K.flatten(loss_t)
        i_t = K.arange(n) + (L-1) * n
        r_t = K.gather(l1d_t, i_t)
        return r_t

    # Return the log loss
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)


#
# Viterbi decoder layer
# The output shape is (tag_set_size, tag_set_size)
# Assume T (size of the tag set) = 3, k = 5
#        S[5,1]                    S[5,2]                  S[5,3]
# S[4,1]+A[1,1]+P[5,1]      S[4,1]+A[1,2]+P[5,2]    S[4,1]+A[1,3]+P[5,3]
# S[4,2]+A[2,1]+P[5,1]      S[4,2]+A[2,2]+P[5,2]    S[4,2]+A[2,3]+P[5,3]
# S[4,3]+A[3,1]+P[5,1]      S[4,3]+A[3,2]+P[5,2]    S[4,3]+A[3,3]+P[5,3]
#
class Viterbi():
    # "yp" is Y prior, if we have any prior knowledge about the outcome of the sequence
    # yp's shape is (batch_size, seq_len), yp[i,k] = tag_index
    def __init__(self, yp=None):
        self._yp = yp
        return

    # Dynamic programming to keep the best score for each tag for Xi
    def decode(self, P, A, L):
        # P is the emission score for each char in sentence
        # P's shape is (batch_size, seq_len, tag_set_size)
        # A is the transition score for adjacent tags
        # A's shape is (tag_set_size+1, tag_set_size)
        (batch_size, seq_len, tag_set_size) = P.shape

        # Adjust P value with prior knowledge
        # If we know a given bit should be mapped to a given tag t,
        # adjust the P value for that bit s.t. P[t] >> P[i], i <> t
        self.apply_yp(P, A)

        # A's score is invariant w.r.t. the batch and sequence
        # Extend the A to [batch_size, tag_size+1, tag_size] tensors
        A_mat = np.expand_dims(A, axis=0)
        A_k = np.repeat(A_mat, batch_size, axis=0)

        # Go through the entire sequence, compute the scores
        S = np.zeros((batch_size, seq_len, tag_set_size))
        B = np.zeros((batch_size, seq_len, tag_set_size))
        S[:,0,] = P[:,0,] + A_k[:,-1,]
        for k in range(1, seq_len):
            P_mat = np.expand_dims(P[:,k,], axis=1)
            P_k = np.repeat(P_mat, tag_set_size, axis=1)

            S_mat = np.expand_dims(S[:,k-1,], axis=-1)
            S_k_1 = np.repeat(S_mat, tag_set_size, axis=-1)

            S_k = P_k + A_k[:,:-1,] + S_k_1
            (S[:,k,], B[:,k,]) = self.max_bptr(S_k)

        # Go back to get the max sequence
        T = np.zeros((batch_size, seq_len), dtype='int')
        for i in range(batch_size):
            idx = L[i] - 1
            T[i, idx] = np.argmax(S[i,idx,], axis=-1)
            for k in range(idx-1, -1, -1):
                T[i,k] = B[i,k+1,T[i,k+1]]

        return (T,S)

    # Return the max value and back pointer for each S[k,i] from the matrix
    def max_bptr(self, S_k):
        S = np.amax(S_k, axis=1) 
        B = np.argmax(S_k, axis=1) 
        return (S, B)

    # Apply prior knowledge through modifying "P"
    def apply_yp(self, P, A):
        if self._yp is None:
            return

        (batch_size, seq_len, tag_set_size) = P.shape
        A_diff = np.amax(A) - np.amin(A)
        P_diff = np.amax(P) - np.amin(P)
        max_val = (A_diff + P_diff) * seq_len
        for i in range(0, batch_size):
            for k in range(0, seq_len):
                tagp = self._yp[i, k]
                if tagp >= 0:
                    P[i,k,tagp] += max_val


#
# Test viterbi decoder
class TestViterbi():
    def __init__(self, **kwargs):
        self._seq_len = 5
        self._tag_set_size = 2
        P = np.asarray([0.9, 0.1, 0.9, 0.1, 0.7, 0.3, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.1, 0.9, 0.8, 0.2, 0.1, 0.9, 1.0, 0.0])
        self._P = P.reshape(2,5,2)
        A = np.asarray([0.8, 0.2, 0.7, 0.3, 1.0, 0.0])
        self._A = A.reshape(3,2)
        self._L = np.asarray([5,5])
        yp = np.asarray([-1, -1, 1, 0, -1, 1, 0, -1, -1, -1])
        self._yp = yp.reshape(2,5)

    # Sequences T should be 
    # [0 0 0 1 0] [0 1 0 1 0]
    # Scores S should be 
    # [[[ 0.9  0.1] [ 3.6  2.2] [ 5.1  4.1] [ 5.9  6.3] [ 8.   6.6]]
    #  [[ 2.   0. ] [ 2.9  3.1] [ 4.6  3.6] [ 5.5  5.7] [ 7.4  6. ]]]
    def test(self):
        V = Viterbi()
        (T, S) = V.decode(self._P, self._A, self._L)
        print(T)
        print(S)
        Vp = Viterbi(self._yp)
        (Tp, Sp) = Vp.decode(self._P, self._A, self._L)
        print(Tp)
        print(Sp)


if __name__ == '__main__':
    tv = TestViterbi()
    tv.test()