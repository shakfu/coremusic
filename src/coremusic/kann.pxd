# kann.pxd - Cython declarations for the KANN neural network library
# KANN: Klib Artificial Neural Network
# https://github.com/attractivechaos/kann

from libc.stdint cimport int32_t, uint8_t, uint16_t, uint32_t, uint64_t
from libc.stdio cimport FILE

# ============================================================================
# Constants
# ============================================================================

cdef extern from "kautodiff.h":
    # Max dimensions and operators
    cdef int KAD_MAX_DIM  # 4
    cdef int KAD_MAX_OP   # 64

    # Node flags
    cdef int KAD_VAR       # 0x1 - variable
    cdef int KAD_CONST     # 0x2 - constant
    cdef int KAD_POOL      # 0x4 - pool
    cdef int KAD_SHARE_RNG # 0x10 - share RNG across time steps

    # Padding modes
    cdef int KAD_PAD_NONE  # 0 - smallest zero-padding
    cdef int KAD_PAD_SAME  # -2 - output same dim as input

    # Operation modes
    cdef int KAD_ALLOC     # 1
    cdef int KAD_FORWARD   # 2
    cdef int KAD_BACKWARD  # 3
    cdef int KAD_SYNC_DIM  # 4


cdef extern from "kann.h":
    # External flags
    cdef int KANN_F_IN     # 0x1 - input
    cdef int KANN_F_OUT    # 0x2 - output
    cdef int KANN_F_TRUTH  # 0x4 - truth output
    cdef int KANN_F_COST   # 0x8 - final cost

    # Cost function types
    cdef int KANN_C_CEB     # 1 - binary cross-entropy (sigmoid)
    cdef int KANN_C_CEM     # 2 - multi-class cross-entropy (softmax)
    cdef int KANN_C_CEB_NEG # 3 - binary cross-entropy for tanh (-1,1)
    cdef int KANN_C_MSE     # 4 - mean square error

    # RNN flags
    cdef int KANN_RNN_VAR_H0 # 0x1 - variable initial hidden states
    cdef int KANN_RNN_NORM   # 0x2 - layer normalization


# ============================================================================
# Data Structures
# ============================================================================

cdef extern from "kautodiff.h":
    # Computational graph node
    ctypedef struct kad_node_t:
        uint8_t n_d            # number of dimensions (max KAD_MAX_DIM)
        uint8_t flag           # node type flags
        uint16_t op            # operator ID
        int32_t n_child        # number of child nodes
        int32_t tmp            # temporary field
        int32_t ptr_size       # size of ptr
        int32_t d[4]           # dimensions (KAD_MAX_DIM=4)
        int32_t ext_label      # external label
        uint32_t ext_flag      # external flags
        float *x               # values
        float *g               # gradients
        void *ptr              # operator-specific parameters
        void *gtmp             # temp gradient data
        kad_node_t **child     # child nodes
        kad_node_t *pre        # previous node (for RNN)

    ctypedef kad_node_t* kad_node_p

    # Operator function type
    ctypedef int (*kad_op_f)(kad_node_t*, int)

    # Global operator arrays
    kad_op_f kad_op_list[64]   # KAD_MAX_OP
    char *kad_op_name[64]


cdef extern from "kann.h":
    # Neural network structure
    ctypedef struct kann_t:
        int n              # number of nodes
        kad_node_t **v     # list of nodes
        float *x           # collated variable values
        float *g           # collated gradients
        float *c           # collated constant values
        void *mt           # multi-threading data

    # Verbose flag
    int kann_verbose


# ============================================================================
# kautodiff.h Functions - Computational Graph API
# ============================================================================

cdef extern from "kautodiff.h":
    # Graph compilation and deletion
    kad_node_t **kad_compile_array(int *n_node, int n_roots, kad_node_t **roots)
    kad_node_t **kad_compile(int *n_node, int n_roots, ...)
    void kad_delete(int n, kad_node_t **a)

    # Evaluation
    const float *kad_eval_at(int n, kad_node_t **a, int from_node)
    void kad_eval_marked(int n, kad_node_t **a)
    int kad_sync_dim(int n, kad_node_t **v, int batch_size)

    # Gradient computation
    void kad_grad(int n, kad_node_t **a, int from_node)

    # RNN unrolling
    kad_node_t **kad_unroll(int n_v, kad_node_t **v, int *new_n, int *length)
    int kad_n_pivots(int n_v, kad_node_t **v)
    kad_node_t **kad_clone(int n, kad_node_t **v, int batch_size)

    # Node creation
    kad_node_t *kad_var(float *x, float *g, int n_d, ...)
    kad_node_t *kad_const(float *x, int n_d, ...)
    kad_node_t *kad_feed(int n_d, ...)

    # Arithmetic operators
    kad_node_t *kad_add(kad_node_t *x, kad_node_t *y)
    kad_node_t *kad_sub(kad_node_t *x, kad_node_t *y)
    kad_node_t *kad_mul(kad_node_t *x, kad_node_t *y)
    kad_node_t *kad_matmul(kad_node_t *x, kad_node_t *y)
    kad_node_t *kad_cmul(kad_node_t *x, kad_node_t *y)

    # Loss functions
    kad_node_t *kad_mse(kad_node_t *x, kad_node_t *y)
    kad_node_t *kad_ce_multi(kad_node_t *x, kad_node_t *y)
    kad_node_t *kad_ce_bin(kad_node_t *x, kad_node_t *y)
    kad_node_t *kad_ce_bin_neg(kad_node_t *x, kad_node_t *y)
    kad_node_t *kad_ce_multi_weighted(kad_node_t *pred, kad_node_t *truth, kad_node_t *weight)

    # Convolution and pooling
    kad_node_t *kad_conv2d(kad_node_t *x, kad_node_t *w, int r_stride, int c_stride, int r_pad, int c_pad)
    kad_node_t *kad_max2d(kad_node_t *x, int kernel_h, int kernel_w, int r_stride, int c_stride, int r_pad, int c_pad)
    kad_node_t *kad_conv1d(kad_node_t *x, kad_node_t *w, int stride, int pad)
    kad_node_t *kad_max1d(kad_node_t *x, int kernel_size, int stride, int pad)
    kad_node_t *kad_avg1d(kad_node_t *x, int kernel_size, int stride, int pad)

    # Regularization
    kad_node_t *kad_dropout(kad_node_t *x, kad_node_t *r)
    kad_node_t *kad_sample_normal(kad_node_t *x)

    # Unary operators
    kad_node_t *kad_square(kad_node_t *x)
    kad_node_t *kad_sigm(kad_node_t *x)
    kad_node_t *kad_tanh(kad_node_t *x)
    kad_node_t *kad_relu(kad_node_t *x)
    kad_node_t *kad_softmax(kad_node_t *x)
    kad_node_t *kad_1minus(kad_node_t *x)
    kad_node_t *kad_exp(kad_node_t *x)
    kad_node_t *kad_log(kad_node_t *x)
    kad_node_t *kad_sin(kad_node_t *x)
    kad_node_t *kad_stdnorm(kad_node_t *x)

    # Pooling operators
    kad_node_t *kad_avg(int n, kad_node_t **x)
    kad_node_t *kad_max(int n, kad_node_t **x)
    kad_node_t *kad_stack(int n, kad_node_t **x)
    kad_node_t *kad_select(int n, kad_node_t **x, int which)

    # Reduction operators
    kad_node_t *kad_reduce_sum(kad_node_t *x, int axis)
    kad_node_t *kad_reduce_mean(kad_node_t *x, int axis)

    # Shape operators
    kad_node_t *kad_slice(kad_node_t *x, int axis, int start, int end)
    kad_node_t *kad_concat(int axis, int n, ...)
    kad_node_t *kad_concat_array(int axis, int n, kad_node_t **p)
    kad_node_t *kad_reshape(kad_node_t *x, int n_d, int *d)
    kad_node_t *kad_reverse(kad_node_t *x, int axis)
    kad_node_t *kad_switch(int n, kad_node_t **p)

    # Size utilities
    int kad_size_var(int n, kad_node_t *const* v)
    int kad_size_const(int n, kad_node_t *const* v)
    int kad_len(const kad_node_t *p)

    # Graph I/O
    int kad_save(FILE *fp, int n_node, kad_node_t **node)
    kad_node_t **kad_load(FILE *fp, int *_n_node)

    # Random number generator
    void *kad_rng()
    void kad_srand(void *d, uint64_t seed)
    uint64_t kad_rand(void *d)
    double kad_drand(void *d)
    double kad_drand_normal(void *d)

    # BLAS-like operations
    void kad_saxpy(int n, float a, const float *x, float *y)

    # Debugging
    void kad_trap_fe()
    void kad_print_graph(FILE *fp, int n, kad_node_t **v)
    void kad_check_grad(int n, kad_node_t **a, int from_node)


# ============================================================================
# kann.h Functions - Neural Network API
# ============================================================================

cdef extern from "kann.h":
    # Network lifecycle
    kann_t *kann_new(kad_node_t *cost, int n_rest, ...)
    kann_t *kann_unroll(kann_t *a, ...)
    kann_t *kann_unroll_array(kann_t *a, int *length)
    kann_t *kann_clone(kann_t *a, int batch_size)
    void kann_delete(kann_t *a)
    void kann_delete_unrolled(kann_t *a)

    # Multi-threading
    void kann_mt(kann_t *ann, int n_threads, int max_batch_size)

    # Data binding and evaluation
    int kann_feed_bind(kann_t *a, uint32_t ext_flag, int32_t ext_label, float **x)
    float kann_cost(kann_t *a, int cost_label, int cal_grad)
    int kann_eval(kann_t *a, uint32_t ext_flag, int ext_label)
    int kann_eval_out(kann_t *a)
    int kann_class_error(const kann_t *ann, int *base)
    void kann_copy_mt(kann_t *a, int mbs, int i)

    # Node utilities
    int kann_find(const kann_t *a, uint32_t ext_flag, int32_t ext_label)
    int kann_feed_dim(const kann_t *a, uint32_t ext_flag, int32_t ext_label)

    # RNN control
    void kann_rnn_start(kann_t *a)
    void kann_rnn_end(kann_t *a)

    # Training/inference mode
    void kann_switch(kann_t *a, int is_train)

    # Optimization
    void kann_RMSprop(int n, float h0, const float *h, float decay, const float *g, float *t, float *r)
    void kann_shuffle(int n, int *s)
    float kann_grad_clip(float thres, int n, float *g)

    # Layer creation
    kad_node_t *kann_layer_input(int n1)
    kad_node_t *kann_layer_dense(kad_node_t *inp, int n1)
    kad_node_t *kann_layer_dropout(kad_node_t *t, float r)
    kad_node_t *kann_layer_layernorm(kad_node_t *inp)
    kad_node_t *kann_layer_rnn(kad_node_t *inp, int n1, int rnn_flag)
    kad_node_t *kann_layer_lstm(kad_node_t *inp, int n1, int rnn_flag)
    kad_node_t *kann_layer_gru(kad_node_t *inp, int n1, int rnn_flag)
    kad_node_t *kann_layer_conv2d(kad_node_t *inp, int n_flt, int k_rows, int k_cols, int stride_r, int stride_c, int pad_r, int pad_c)
    kad_node_t *kann_layer_conv1d(kad_node_t *inp, int n_flt, int k_size, int stride, int pad)
    kad_node_t *kann_layer_cost(kad_node_t *t, int n_out, int cost_type)

    # Weight/node creation
    kad_node_t *kann_new_leaf(uint8_t flag, float x0_01, int n_d, ...)
    kad_node_t *kann_new_scalar(uint8_t flag, float x)
    kad_node_t *kann_new_weight(int n_row, int n_col)
    kad_node_t *kann_new_bias(int n)
    kad_node_t *kann_new_weight_conv2d(int n_out, int n_in, int k_row, int k_col)
    kad_node_t *kann_new_weight_conv1d(int n_out, int n_in, int kernel_len)

    # Advanced layer creation (with offset and parent tracking)
    kad_node_t *kann_new_leaf2(int *offset, kad_node_p *par, uint8_t flag, float x0_01, int n_d, ...)
    kad_node_t *kann_layer_dense2(int *offset, kad_node_p *par, kad_node_t *inp, int n1)
    kad_node_t *kann_layer_dropout2(int *offset, kad_node_p *par, kad_node_t *t, float r)
    kad_node_t *kann_layer_layernorm2(int *offset, kad_node_t **par, kad_node_t *inp)
    kad_node_t *kann_layer_rnn2(int *offset, kad_node_t **par, kad_node_t *inp, kad_node_t *h0, int rnn_flag)
    kad_node_t *kann_layer_gru2(int *offset, kad_node_t **par, kad_node_t *inp, kad_node_t *h0, int rnn_flag)

    # Training (feedforward with single input/output)
    int kann_train_fnn1b(kann_t *ann, float lr, int mini_size, int max_epoch, int min_epoch, int max_drop_streak, float frac_val, int n, float **_x, float **_y)
    int kann_train_fnn1(kann_t *ann, float lr, int mini_size, int max_epoch, int max_drop_streak, float frac_val, int n, float **_x, float **_y)
    float kann_cost_fnn1(kann_t *a, int n, float **x, float **y)

    # Inference
    const float *kann_apply1_to(kann_t *a, float *x, int ext_flag, int ext_label)
    const float *kann_apply1(kann_t *a, float *x)

    # Model I/O
    void kann_save_fp(FILE *fp, kann_t *ann)
    void kann_save(const char *fn, kann_t *ann)
    kann_t *kann_load_fp(FILE *fp)
    kann_t *kann_load(const char *fn)


# ============================================================================
# kann_extra/kann_data.h - Data Loading Utilities
# ============================================================================

cdef extern from "kann_extra/kann_data.h":
    # Data structure for tabular data
    ctypedef struct kann_data_t:
        int n_row          # number of rows
        int n_col          # number of columns
        int n_grp          # number of groups
        float **x          # data matrix [n_row][n_col]
        char **rname       # row names
        char **cname       # column names
        int *grp           # group boundaries

    # Data I/O
    kann_data_t *kann_data_read(const char *fn)
    void kann_data_free(kann_data_t *d)
