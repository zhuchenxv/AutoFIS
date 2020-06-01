from __future__ import division

import os

import numpy as np
import tensorflow as tf

import __init__

dtype = tf.float32 if __init__.config['dtype'] == 'float32' else tf.float64
minval = __init__.config['minval']
maxval = __init__.config['maxval']
mean = __init__.config['mean']
stddev = __init__.config['stddev']


def get_variable(init_type='xavier', shape=None, name=None, minval=minval, maxval=maxval, mean=mean,
                 stddev=stddev, dtype=dtype, ):
    if type(init_type) is str:
        init_type = init_type.lower()
    if init_type == 'tnormal':
        return tf.Variable(tf.truncated_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype), name=name)
    elif init_type == 'uniform':
        return tf.Variable(tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype), name=name)
    elif init_type == 'normal':
        return tf.Variable(tf.random_normal(shape=shape, mean=mean, stddev=stddev, dtype=dtype), name=name)
    elif init_type == 'xavier':
        maxval = np.sqrt(6. / np.sum(shape))
        minval = -maxval
        print(name, 'initialized from:', minval, maxval)
        return tf.Variable(tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype), name=name)
    elif init_type == 'xavier_out':
        maxval = np.sqrt(3. / shape[1])
        minval = -maxval
        print(name, 'initialized from:', minval, maxval)
        return tf.Variable(tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype), name=name)
    elif init_type == 'xavier_in':
        maxval = np.sqrt(3. / shape[0])
        minval = -maxval
        print(name, 'initialized from:', minval, maxval)
        return tf.Variable(tf.random_uniform(shape=shape, minval=minval, maxval=maxval, dtype=dtype), name=name)
    elif init_type == 'zero':
        return tf.Variable(tf.zeros(shape=shape, dtype=dtype), name=name)
    elif init_type == 'one':
        return tf.Variable(tf.ones(shape=shape, dtype=dtype), name=name)
    elif init_type == 'identity' and len(shape) == 2 and shape[0] == shape[1]:
        return tf.Variable(tf.diag(tf.ones(shape=shape[0], dtype=dtype)), name=name)
    elif 'int' in init_type.__class__.__name__ or 'float' in init_type.__class__.__name__:
        return tf.Variable(tf.ones(shape=shape, dtype=dtype) * init_type, name=name)


def selu(x):
    with tf.name_scope('selu'):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def activate(weights, act_type):
    if type(act_type) is str:
        act_type = act_type.lower()
    if act_type == 'sigmoid':
        return tf.nn.sigmoid(weights)
    elif act_type == 'softmax':
        return tf.nn.softmax(weights)
    elif act_type == 'relu':
        return tf.nn.relu(weights)
    elif act_type == 'tanh':
        return tf.nn.tanh(weights)
    elif act_type == 'elu':
        return tf.nn.elu(weights)
    elif act_type == 'selu':
        return selu(weights)
    elif act_type == 'none':
        return weights
    else:
        return weights


def get_optimizer(opt_algo):
    opt_algo = opt_algo.lower()
    if opt_algo == 'adaldeta':
        return tf.train.AdadeltaOptimizer
    elif opt_algo == 'adagrad':
        return tf.train.AdagradOptimizer
    elif opt_algo == 'adam':
        return tf.train.AdamOptimizer
    elif opt_algo == 'moment':
        return tf.train.MomentumOptimizer
    elif opt_algo == 'ftrl':
        return tf.train.FtrlOptimizer
    elif opt_algo == 'gd' or opt_algo == 'sgd':
        return tf.train.GradientDescentOptimizer
    elif opt_algo == 'padagrad':
        return tf.train.ProximalAdagradOptimizer
    elif opt_algo == 'pgd':
        return tf.train.ProximalGradientDescentOptimizer
    elif opt_algo == 'rmsprop':
        return tf.train.RMSPropOptimizer
    else:
        return tf.train.GradientDescentOptimizer


def get_loss(loss_func):
    loss_func = loss_func.lower()
    if loss_func == 'weight' or loss_func == 'weighted':
        return tf.nn.weighted_cross_entropy_with_logits
    elif loss_func == 'sigmoid':
        return tf.nn.sigmoid_cross_entropy_with_logits
    elif loss_func == 'softmax':
        return tf.nn.softmax_cross_entropy_with_logits


def check(x):
    try:
        return x is not None and x is not False and float(x) > 0
    except TypeError:
        return True


def get_l2_loss(params, variables):
    _loss = None
    with tf.name_scope('l2_loss'):
        for p, v in zip(params, variables):
            print('add l2', p, v)
            if not type(p) is list:
                if check(p):
                    if type(v) is list:
                        for _v in v:
                            if _loss is None:
                                _loss = p * tf.nn.l2_loss(_v)
                            else:
                                _loss += p * tf.nn.l2_loss(_v)
                    else:
                        if _loss is None:
                            _loss = p * tf.nn.l2_loss(v)
                        else:
                            _loss += p * tf.nn.l2_loss(v)
            else:
                for _lp, _lv in zip(p, v):
                    if _loss is None:
                        _loss = _lp * tf.nn.l2_loss(_lv)
                    else:
                        _loss += _lp * tf.nn.l2_loss(_lv)
    return _loss


def normalize(norm, x, scale):
    if norm:
        return x * scale
    else:
        return x


def mul_noise(noisy, x, training=None):
    if check(noisy) and training is not None:
        with tf.name_scope('mul_noise'):
            noise = tf.truncated_normal(
                shape=tf.shape(x),
                mean=1.0, stddev=noisy)
            return tf.where(
                training,
                tf.multiply(x, noise),
                x)
    else:
        return x


def add_noise(noisy, x, training):
    if check(noisy):
        with tf.name_scope('add_noise'):
            noise = tf.truncated_normal(
                shape=tf.shape(x),
                mean=0, stddev=noisy)
            return tf.where(
                training,
                x + noise,
                x)
    else:
        return x


def create_placeholder(num_inputs, dtype=dtype, training=False):
    with tf.name_scope('input'):
        inputs = tf.placeholder(tf.int32, [None, num_inputs], name='input')
        labels = tf.placeholder(tf.float32, [None], name='label')
        if check(training):
            training = tf.placeholder(dtype=tf.bool, name='training')
    return inputs, labels, training


def split_data_mask(inputs, num_inputs, norm=False, real_inputs=None, num_cat=None):
    if not check(real_inputs):
        if check(norm):
            mask = np.sqrt(1. / num_inputs)
        else:
            mask = 1
        flag = norm
    else:
        inputs, mask = inputs[:, :real_inputs], inputs[:, real_inputs:]
        mask = tf.to_float(mask)
        if check(norm):
            mask /= np.sqrt(num_cat + 1)
            mask_cat, mask_mul = mask[:, :num_cat], mask[:, num_cat:]
            sum_mul = tf.reduce_sum(mask_mul, 1, keep_dims=True)
            sum_mul = tf.maximum(sum_mul, tf.ones_like(sum_mul))
            mask_mul /= tf.sqrt(sum_mul)
            mask = tf.concat([mask_cat, mask_mul], 1)
        flag = True
        num_inputs = real_inputs
    return inputs, mask, flag, num_inputs


def drop_out(training, keep_probs, ):
    with tf.name_scope('drop_out'):
        keep_probs = tf.where(training,
                              keep_probs,
                              np.ones_like(keep_probs),
                              name='keep_prob')
    return keep_probs

#
# def embedding_lookup(init, input_dim, factor, inputs, apply_mask=False, mask=None,
#                      use_w=True, use_v=True, use_b=True, use_third=True, use_fourth=True, fm_path=None, fm_step=None, fm_data=None):
#     if fm_path is not None and fm_step is not None:
#         print('initialized from fm', fm_path, fm_step)
#         fm_dict = load_fm(fm_path, fm_step, fm_data)
#         with tf.name_scope('embedding'):
#             xw, xv, b, xps, x_fourth = None, None, None, None, None
#             if use_w:
#                 w = tf.Variable(fm_dict['w'], name='w', dtype=dtype)
#                 xw = tf.gather(w, inputs)
#                 # xw = mul_noise(noisy, xw, training)
#                 if apply_mask:
#                     xw = xw * mask
#             if use_v:
#                 v = tf.Variable(fm_dict['v'], name='v', dtype=dtype)
#                 xv = tf.gather(v, inputs)
#                 # xv = mul_noise(noisy, xv, training)
#                 if apply_mask:
#                     xv = xv * tf.expand_dims(mask, 2)
#             if use_b:
#                 b = tf.Variable(fm_dict['b'], name='b', dtype=dtype)
#             return xw, xv, b, xps, x_fourth
#     else:
#         print('random initialize')
#         with tf.name_scope('embedding'):
#             xw, xv, b, xps, x_fourth = None, None, None, None, None
#             if use_w:
#                 # TODO embedding init
#                 # w = get_variable(init, name='w', shape=[input_dim, 1])
#                 w = get_variable(init, name='w', shape=[input_dim,])
#                 # w = get_variable(init_type='tnormal', name='w', shape=[input_dim, 1], stddev=1. / np.sqrt(num_inputs))
#                 # batch * fields * 1
#                 xw = tf.gather(w, inputs)
#                 # xw = mul_noise(noisy, xw, training)
#                 if apply_mask:
#                     xw = xw * mask
#             if use_v:
#                 v = get_variable(init_type=init, name='v', shape=[input_dim, factor])
#                 # v = get_variable(init_type='xavier_out', name='v', shape=[input_dim, factor])
#                 # maxval = np.sqrt(3. / (num_inputs * factor))
#                 # minval = -maxval
#                 # v = get_variable(init_type='uniform', name='v', shape=[input_dim, factor], minval=minval, maxval=maxval)
#                 # batch * fields * k
#                 xv = tf.gather(v, inputs)
#                 # xv = mul_noise(noisy, xv, training)
#                 if apply_mask:
#                     if type(mask) is np.float64:
#                         xv = xv * mask
#                     else:
#                         xv = xv * tf.expand_dims(mask, 2)
#             if use_b:
#                 b = get_variable('zero', name='b', shape=[1])
#
#             if use_third:
#                 ps = get_variable(init_type=init, name='xps', shape=[input_dim, factor])
#                 # batch * fields * k
#                 xps = tf.gather(ps, inputs)
#                 if apply_mask:
#                     if type(mask) is np.float64:
#                         xps = xps * mask
#                     else:
#                         xps = xps * tf.expand_dims(mask, 2)
#             if use_fourth:
#                 fourth = get_variable(init_type=init, name='x_fourth', shape=[input_dim, factor])
#                 # batch * fields * k
#                 x_fourth = tf.gather(fourth, inputs)
#                 if apply_mask:
#                     if type(mask) is np.float64:
#                         x_fourth = x_fourth * mask
#                     else:
#                         x_fourth = x_fourth * tf.expand_dims(mask, 2)
#
#             return xw, xv, b, xps, x_fourth


def embedding_lookup(init, input_dim, factor, inputs, apply_mask=False, mask=None,
                     use_w=True, use_v=True, use_b=True, fm_path=None, fm_step=None,  third_order=False,order=None,
                     embedsize=None):
    xw, xv, b, xps = None, None, None, None
    if fm_path is not None and fm_step is not None:
        fm_dict = load_fm(fm_path, fm_step)
        with tf.name_scope('embedding'):
            if use_w:
                w = tf.Variable(fm_dict['w'], name='w', dtype=dtype)
                xw = tf.gather(w, inputs)
                if apply_mask:
                    xw = xw * mask
            if use_v:
                v = tf.Variable(fm_dict['v'], name='v', dtype=dtype)
                xv = tf.gather(v, inputs)
                if apply_mask:
                    xv = xv * tf.expand_dims(mask, 2)
            if use_b:
                b = tf.Variable(fm_dict['b'], name='b', dtype=dtype)
            #TODO: deal with xps
    else:
        with tf.name_scope('embedding'):
            if use_w:
                w = get_variable(init, name='w', shape=[input_dim,])
                tf.add_to_collection("embeddings", w)
                xw = tf.gather(w, inputs)
                if apply_mask:
                    xw = xw * mask
                # tf.add_to_collection("embeddings", xw)
            if use_v:
                v = get_variable(init_type=init, name='v', shape=[input_dim, factor])
                tf.add_to_collection("embeddings", v)
                xv = tf.gather(v, inputs)
                if apply_mask:
                    xv = xv * tf.expand_dims(mask, 2)
                # tf.add_to_collection("embeddings", xv)
            if third_order:
                third_v = get_variable(init_type=init, name='thiird_v', shape=[input_dim, factor])
                tf.add_to_collection("embeddings", third_v)
                xps = tf.gather(third_v, inputs)
                if apply_mask:
                    xps = xps * tf.expand_dims(mask, 2)
                # tf.add_to_collection("embeddings", xps)
            # if order is not None:
            #     for i in range(order):
            #         xp = get_variable(init, name='vp%d'%i, shape=[input_dim, embedsize[i]])
            #         xps.append(tf.gather(xp, inputs))
            #         if apply_mask:
            #             xps[-1] *= tf.expand_dims(mask,2)
            if use_b:
                b = get_variable('zero', name='b', shape=[1])
                tf.add_to_collection("embeddings", b)
                # tf.add_to_collection("embeddings", b)
    return xw, xv, b, xps


def linear(xw):
    with tf.name_scope('linear'):
        l = tf.squeeze(tf.reduce_sum(xw, 1))
    return l


def output(x):
    with tf.name_scope('output'):
        if type(x) is list:
            logits = sum(x)
        else:
            logits = x
        outputs = tf.nn.sigmoid(logits)
    return logits, outputs


def row_col_fetch(xv_embed, num_inputs):
    """
    for field-aware embedding
    :param xv_embed: batch * num * (num - 1) * k
    :param num_inputs: num
    :return:
    """
    rows = []
    cols = []
    for i in range(num_inputs - 1):
        for j in range(i + 1, num_inputs):
            rows.append([i, j - 1])
            cols.append([j, i])
    with tf.name_scope('lookup'):
        # batch * pair * k
        xv_p = tf.transpose(
            # pair * batch * k
            tf.gather_nd(
                # num * (num - 1) * batch * k
                tf.transpose(xv_embed, [1, 2, 0, 3]),
                rows),
            [1, 0, 2])
        xv_q = tf.transpose(
            tf.gather_nd(
                tf.transpose(xv_embed, [1, 2, 0, 3]),
                cols),
            [1, 0, 2])
    return xv_p, xv_q


def row_col_expand(xv_embed, num_inputs):
    """
    for universal embedding and field-aware param
    :param xv_embed: batch * num * k
    :param num_inputs:
    :return:
    """
    rows = []
    cols = []
    for i in range(num_inputs - 1):
        for j in range(i + 1, num_inputs):
            rows.append(i)
            cols.append(j)
    with tf.name_scope('lookup'):
        # batch * pair * k
        xv_p = tf.transpose(
            # pair * batch * k
            tf.gather(
                # num * batch * k
                tf.transpose(
                    xv_embed, [1, 0, 2]),
                rows),
            [1, 0, 2])
        # batch * pair * k
        xv_q = tf.transpose(
            tf.gather(
                tf.transpose(
                    xv_embed, [1, 0, 2]),
                cols),
            [1, 0, 2])
    return xv_p, xv_q


def batch_kernel_product(xv_p, xv_q, kernel=None, add_bias=True, factor=None, num_pairs=None, reduce_sum=True, mask=None):
    """
    :param xv_p: batch * pair * k
    :param xv_q: batch * pair * k
    :param kernel: k * pair * k
    :param add_bias:
    :param bias: pair
    :param init:
    :param factor:
    :param num_pairs:
    :return:
    """
    with tf.name_scope('inner'):
        if kernel is None:
            # kernel = get_variable(init, name='kernel', shape=[factor, num_pairs, factor])
            maxval = np.sqrt(3. / factor)
            minval = -maxval
            # stddev = np.sqrt(3. / factor)
            kernel = get_variable('uniform', name='kernel', shape=[factor, num_pairs, factor], minval=minval, maxval=maxval)
            # kernel = get_variable('tnormal', name='kernel', shape=[factor, num_pairs, factor], mean=0, stddev=stddev)
        if add_bias:
            bias = get_variable(0, name='bias', shape=[num_pairs])
        else:
            bias = None
        # batch * 1 * pair * k
        xv_p = tf.expand_dims(xv_p, 1)
        # batch * pair
        prods = tf.reduce_sum(
            # batch * pair * k
            tf.multiply(
                # batch * pair * k
                tf.transpose(
                    # batch * k * pair
                    tf.reduce_sum(
                        # batch * k * pair * k
                        tf.multiply(
                            xv_p, kernel),
                        -1),
                    [0, 2, 1]),
                xv_q),
            -1)
        if add_bias:
            prods += bias
        if reduce_sum:
            prods = tf.reduce_sum(prods, 1)
    return prods, kernel, bias


def batch_mlp(h, node_in, num_pairs, init, net_sizes, net_acts, net_keeps, add_bias=True,
              reduce_sum=True, layer_norm=False, batch_norm=False, apply_mask=False, mask=None):
    """
    :param h: batch * pair * 2k
    :param num_pairs:
    :param init:
    :param net_sizes:
    :param net_acts:
    :param net_keeps:
    :param add_bias
    :return:
    """
    with tf.name_scope('net'):
        # pair * batch * 2k
        h = tf.transpose(h, [1, 0, 2])
        if apply_mask:
            if not type(mask) is np.float64:
                mask = tf.expand_dims(tf.transpose(mask), 2)
        net_kernels = []
        net_biases = []
        for i in range(len(net_sizes)):
            with tf.name_scope('layer_%d' % i):
                _w = get_variable(init, name='w_%d' % i, shape=[num_pairs, node_in, net_sizes[i]])
                _wx = tf.matmul(h, _w)
                net_kernels.append(_w)
                if layer_norm:
                    _wx = layer_normalization(_wx, reduce_dim=[0, 2], out_dim=[num_pairs, 1, net_sizes[i]], bias=False)
                elif batch_norm:
                    _wx = batch_normalization(_wx, reduce_dim=[0, 1], out_dim=[num_pairs, 1, net_sizes[i]], bias=False)
                if add_bias:
                    _b = get_variable(0, name='b_%d' % i, shape=[num_pairs, 1, net_sizes[i]])
                    _wx += _b
                    net_biases.append(_b)
                h = tf.nn.dropout(
                    activate(_wx, net_acts[i]),
                    net_keeps[i])
                node_in = net_sizes[i]
                if apply_mask:
                    # pair * batch * n
                    if not type(mask) is np.float64:
                        h = h * mask
        # batch * pair * ?
        h = tf.transpose(h, [1, 0, 2])
        if reduce_sum:
            h = tf.squeeze(tf.reduce_sum(h, 1))
    return h, net_kernels, net_biases


# def batch_product_net(xv_p, xv_q, num_pairs, factor, init, net_sizes, net_acts, net_keeps, add_bias=True):
#     """
#     batch net with pnn structure
#     :param xv_p:
#     :param xv_q:
#     :param num_pairs:
#     :param factor:
#     :param init:
#     :param net_sizes:
#     :param net_acts:
#     :param net_keeps:
#     :param add_bias:
#     :return:
#     """
#     with tf.name_scope('product_net'):
#         op = tf.reshape(
#             tf.multiply(
#                 tf.expand_dims(xv_p, 3),
#                 tf.expand_dims(xv_q, 2)),
#             [-1, num_pairs, factor * factor])
#         n = tf.concat([xv_p, xv_q, op], 2)
#     return batch_net(n, factor * 2 + factor ** 2, num_pairs, init, net_sizes, net_acts, net_keeps, add_bias=add_bias)


def batch_normalization(x, reduce_dim=0, out_dim=None, scale=None, bias=None):
    if type(reduce_dim) is int:
        reduce_dim = [reduce_dim]
    if type(out_dim) is int:
        out_dim = [out_dim]
    with tf.name_scope('batch_norm'):
        batch_mean, batch_var = tf.nn.moments(x, reduce_dim, keep_dims=True)
        x = (x - batch_mean) / tf.sqrt(batch_var)
        if scale is not False:
            scale = scale if scale is not None else tf.Variable(tf.ones(out_dim), dtype=dtype, name='g')
        if bias is not False:
            bias = bias if bias is not None else tf.Variable(tf.zeros(out_dim), dtype=dtype, name='b')
        if scale is not False and bias is not False:
            return x * scale + bias
        elif scale is not False:
            return x * scale
        elif bias is not False:
            return x + bias
        else:
            return x


def layer_normalization(x, reduce_dim=1, out_dim=None, scale=None, bias=None):
    if type(reduce_dim) is int:
        reduce_dim = [reduce_dim]
    if type(out_dim) is int:
        out_dim = [out_dim]
    with tf.name_scope('layer_norm'):
        layer_mean, layer_var = tf.nn.moments(x, reduce_dim, keep_dims=True)
        x = (x - layer_mean) / tf.sqrt(layer_var)
        if scale is not False:
            scale = scale if scale is not None else tf.Variable(tf.ones(out_dim), dtype=dtype, name='g')
        if bias is not False:
            bias = bias if bias is not None else tf.Variable(tf.zeros(out_dim), dtype=dtype, name='b')
        if scale is not False and bias is not False:
            return x * scale + bias
        elif scale is not False:
            return x * scale
        elif bias is not False:
            return x + bias
        else:
            return x


def bin_mlp(init, layer_sizes, layer_acts, layer_keeps, h, node_in, batch_norm=False, layer_norm=False, training=True,
            res_conn=False):
    layer_kernels = []
    layer_biases = []
    x_prev = None
    for i in range(len(layer_sizes)):
        with tf.name_scope('hidden_%d' % i):
            wi = get_variable(init, name='w_%d' % i, shape=[node_in, layer_sizes[i]])
            bi = get_variable(0, name='b_%d' % i, shape=[layer_sizes[i]])
            print(wi.shape, bi.shape)
            print(layer_acts[i], layer_keeps[i])

            h = tf.matmul(h, wi)
            if i < len(layer_sizes) - 1:
                if batch_norm:
                    h = tf.layers.batch_normalization(h, training=training, reuse=tf.AUTO_REUSE, scale=False,
                                                      center=False, name='mlp_bn_%d' % i)
                    # h = batch_normalization(h, out_dim=layer_sizes[i], bias=False)
                elif layer_norm:
                    h = layer_normalization(h, out_dim=layer_sizes[i], bias=False)
            # h = tf.matmul(h, wi)
            h = h + bi
            if res_conn:
                if x_prev is None:
                    x_prev = h
                elif layer_sizes[i-1] == layer_sizes[i]:
                    h += x_prev
                    x_prev = h

            h = tf.nn.dropout(
                activate(
                    h, layer_acts[i]),
                layer_keeps[i])
            node_in = layer_sizes[i]
            layer_kernels.append(wi)
            layer_biases.append(bi)
    return h, layer_kernels, layer_biases


def load_fm(fm_path, fm_step, fm_data):
    fm_abs_path = os.path.join(
        os.path.join(
            os.path.join(
                os.path.join(
                    os.path.join(
                        os.path.join(
                            os.path.dirname(
                                os.path.dirname(
                                    os.path.abspath(__file__))),
                            'log'),
                        fm_data),
                    'FM'),
                fm_path),
            'checkpoints'),
        'model.ckpt-%d' % fm_step)
    reader = tf.train.NewCheckpointReader(fm_abs_path)
    print('load fm', reader.debug_string())
    fm_dict = {'w': reader.get_tensor('embedding/w'),
               'v': reader.get_tensor('embedding/v'),
               'b': reader.get_tensor('embedding/b')}
    return fm_dict
