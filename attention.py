from keras.layers import Layer
import keras.backend as K
import tensorflow as tf

class Attention(Layer):
    def __init__(self, method=None, **kwargs):
        self.supports_masking = True
        if method not in ['lba', 'ga', 'cba', None]:
            raise ValueError('attention method is not supported')
        self.method = method
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            self.att_size = input_shape[0][-1]
            self.query_dim = input_shape[1][-1]
            if self.method in ['ga', 'cba']:
                self.Wq = self.add_weight(
                    name='kernal_query_features', 
                    shape=(self.query_dim, self.att_size), 
                    initializer='glorot_normal', 
                    trainable=True)
        else:
            self.att_size = input_shape[-1]

        if self.method == 'cba':
            self.Wh = self.add_weight(
                name='kernal_hidden_features', 
                shape=(self.att_size, self.att_size), 
                initializer='glorot_normal', 
                trainable=True)
        if self.method in ['lba', 'cba']:
            self.v = self.add_weight(
                name='query_vector', 
                shape=(self.att_size, 1), 
                initializer='zeros', 
                trainable=True)

        super(Attention, self).build(input_shape)

    def call(self, inputs, mask=None):
        # 若传入 [memory, query]
        if isinstance(inputs, list) and len(inputs) == 2:
            memory, query = inputs
            if self.method is None:
                return memory[:, -1, :]
            elif self.method == 'cba':
                hidden = tf.matmul(memory, self.Wh) + tf.expand_dims(tf.matmul(query, self.Wq), axis=1)
                hidden = tf.tanh(hidden)
                s = tf.squeeze(tf.matmul(hidden, self.v), axis=-1)
            elif self.method == 'ga':
                s = tf.reduce_sum(tf.expand_dims(tf.matmul(query, self.Wq), axis=1) * memory, axis=-1)
            else:
                s = tf.squeeze(tf.matmul(memory, self.v), axis=-1)
            if mask is not None:
                mask = mask[0]
        else:
            # 若传入单个 memory tensor
            if isinstance(inputs, list):
                if len(inputs) != 1:
                    raise ValueError('inputs length should not be larger than 2')
                memory = inputs[0]
            else:
                memory = inputs
            if self.method is None:
                return memory[:, -1, :]
            elif self.method == 'cba':
                hidden = tf.matmul(memory, self.Wh)
                hidden = tf.tanh(hidden)
                s = tf.squeeze(tf.matmul(hidden, self.v), axis=-1)
            elif self.method == 'ga':
                raise ValueError('general attention needs the second input')
            else:
                s = tf.squeeze(tf.matmul(memory, self.v), axis=-1)

        s = tf.nn.softmax(s)
        if mask is not None:
            s *= tf.cast(mask, tf.float32)
            sum_by_time = tf.reduce_sum(s, axis=-1, keepdims=True)
            s = s / (sum_by_time + K.epsilon())
        return tf.reduce_sum(memory * tf.expand_dims(s, axis=-1), axis=1)

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            att_size = input_shape[0][-1]
            batch = input_shape[0][0]
        else:
            att_size = input_shape[-1]
            batch = input_shape[0]
        return (batch, att_size)


class SimpleAttention(Layer):
    def __init__(self, method=None, **kwargs):
        self.supports_masking = True
        if method not in ['lba', 'ga', 'cba', None]:
            raise ValueError('attention method is not supported')
        self.method = method
        super(SimpleAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):
            self.att_size = input_shape[0][-1]
            self.query_dim = input_shape[1][-1] + self.att_size
        else:
            self.att_size = input_shape[-1]
            self.query_dim = self.att_size

        if self.method in ['cba', 'ga']:
            self.Wq = self.add_weight(
                name='kernal_query_features', 
                shape=(self.query_dim, self.att_size),
                initializer='glorot_normal', 
                trainable=True)
        if self.method == 'cba':
            self.Wh = self.add_weight(
                name='kernal_hidden_features', 
                shape=(self.att_size, self.att_size), 
                initializer='glorot_normal', 
                trainable=True)
        if self.method in ['lba', 'cba']:
            self.v = self.add_weight(
                name='query_vector', 
                shape=(self.att_size, 1), 
                initializer='zeros', 
                trainable=True)

        super(SimpleAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        query = None
        if isinstance(inputs, list):
            memory = inputs[0]
            if len(inputs) > 1:
                query = inputs[1]
            elif len(inputs) > 2:
                raise ValueError('inputs length should not be larger than 2')
            if isinstance(mask, list):
                mask = mask[0]
        else:
            memory = inputs

        input_shape = K.int_shape(memory)
        if len(input_shape) > 3:
            memory = tf.reshape(memory, (-1,) + input_shape[2:])
            if mask is not None:
                mask = tf.reshape(mask, (-1,) + input_shape[2:-1])
            if query is not None:
                raise ValueError('query is not supported in this mode')

        last = memory[:, -1, :]
        memory = memory[:, :-1, :]
        if query is None:
            query = last
        else:
            query = tf.concat([query, last], axis=-1)

        if self.method is None:
            if len(input_shape) > 3:
                output_shape = K.int_shape(last)
                return tf.reshape(last, (-1, input_shape[1], output_shape[-1]))
            else:
                return last
        elif self.method == 'cba':
            hidden = tf.matmul(memory, self.Wh) + tf.expand_dims(tf.matmul(query, self.Wq), axis=1)
            hidden = tf.tanh(hidden)
            s = tf.squeeze(tf.matmul(hidden, self.v), axis=-1)
        elif self.method == 'ga':
            s = tf.reduce_sum(tf.expand_dims(tf.matmul(query, self.Wq), axis=1) * memory, axis=-1)
        else:
            s = tf.squeeze(tf.matmul(memory, self.v), axis=-1)

        s = tf.nn.softmax(s)
        if mask is not None:
            mask = mask[:, :-1]
            s *= tf.cast(mask, tf.float32)
            sum_by_time = tf.reduce_sum(s, axis=-1, keepdims=True)
            s = s / (sum_by_time + K.epsilon())
        result = tf.concat([tf.reduce_sum(memory * tf.expand_dims(s, axis=-1), axis=1), last], axis=-1)
        if len(input_shape) > 3:
            output_shape = K.int_shape(result)
            return tf.reshape(result, (-1, input_shape[1], output_shape[-1]))
        else:
            return result

    def compute_mask(self, inputs, mask=None):
        if isinstance(inputs, list):
            memory = inputs[0]
        else:
            memory = inputs
        if len(K.int_shape(memory)) > 3 and mask is not None:
            return K.all(mask, axis=-1)
        else:
            return None

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            att_size = input_shape[0][-1]
            seq_len = input_shape[0][1]
            batch = input_shape[0][0]
        else:
            att_size = input_shape[-1]
            seq_len = input_shape[1]
            batch = input_shape[0]
        if len(input_shape) > 3:
            if self.method is not None:
                shape1 = (batch, seq_len, att_size * 2)
            else:
                shape1 = (batch, seq_len, att_size)
            return shape1
        else:
            if self.method is not None:
                shape1 = (batch, att_size * 2)
            else:
                shape1 = (batch, att_size)
            return shape1