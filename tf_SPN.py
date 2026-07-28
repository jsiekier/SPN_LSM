import tensorflow as tf
import numpy as np



class SPN_clf(tf.keras.Model):
    def __init__(self
                 ,num_classes,
                 all_spn_x_y_model,
                 all_prior,
                 bin_borders=None,
                 bin_widths_scaled=None,
                 dtype=tf.dtypes.float32):
        super(SPN_clf, self).__init__()
        self.bin_borders=bin_borders
        self.bin_widths_scaled=tf.constant(bin_widths_scaled, dtype=tf.float32)#bin_widths_scaled
        self.num_classes = num_classes
        self.all_spn_x_y = all_spn_x_y_model

        if num_classes>1:
            self.prior_weights = tf.expand_dims(tf.Variable(initial_value=all_prior, dtype=dtype, trainable=False,
                                             name='Root weights') , axis=0) # name="mean",
        else:
            self.prior_weights=[]
            for prior_id,prior_w in enumerate(all_prior):
                self.prior_weights.append(
                    tf.expand_dims(
                        tf.Variable(initial_value=prior_w, dtype=dtype, trainable=False,name='Root weights_'+str(prior_id)), axis=0) )

    def spn_clf(self,embedding,training):
        if self.num_classes>1:
            inputs = []
            #tf_weights = tf.expand_dims(self.prior_weights, axis=0)
            for label_id, sub_spn in enumerate(self.all_spn_x_y):
                y=np.full((embedding.shape[0],1),label_id)
                spn_input = tf.concat([tf.cast(y, dtype=tf.float32), embedding], axis=-1)
                inputs.append(sub_spn(spn_input,training=training))

            children_prob = tf.concat(inputs, axis=1)  # children
            # print('childrenprob shape ',children_prob.shape)
            # Calculate log-sum-exp
            log_enumerator = children_prob + tf.math.log(self.prior_weights)
            p_x = tf.reduce_logsumexp(log_enumerator, axis=1,keepdims=True)
            p_y_x = log_enumerator - p_x
            return p_y_x,p_x
        else:


            inputs_smooth = [tf.math.log(tf.ones(shape=(embedding.shape[0])) * self.bin_widths_scaled[0])]
            y_1 = np.full((embedding.shape[0], 1), 1)
            spn_input_1 = tf.concat([tf.cast(y_1, dtype=tf.float32), embedding], axis=-1)
            y_0 = np.full((embedding.shape[0], 1), 0)
            spn_input_0 = tf.concat([tf.cast(y_0, dtype=tf.float32), embedding], axis=-1)
            for spn_id, (sub_spn, sub_priors) in enumerate(zip(self.all_spn_x_y, self.prior_weights)):
                res = []
                for sub_sub_spn_id, sub_sub_spn_input in enumerate([spn_input_0, spn_input_1]):
                    spn_log_pred = sub_spn[sub_sub_spn_id](sub_sub_spn_input, training=training)
                    res.append(spn_log_pred)

                res = tf.concat(res, axis=1)
                res = res + tf.math.log(sub_priors)
                p_x = tf.reduce_logsumexp(res, axis=1, keepdims=True)
                p_y_x = res - p_x
                inputs_smooth.append( p_y_x[:, 1] + tf.math.log(self.bin_widths_scaled[spn_id + 1]))

            log_enumerator = tf.stack(inputs_smooth, axis=1)
            log_y = tf.reduce_logsumexp(log_enumerator, axis=1, keepdims=False)

            return log_y, 0#log_p_x
    '''
    def spn_regression_exp(self,embedding,training):
        batch_size = tf.shape(embedding)[0]
        inputs_smooth = [tf.math.log(tf.ones(shape=(batch_size))*self.bin_widths_scaled[0])]
        inputs_decision_boundary=[tf.ones(shape=(batch_size))*self.bin_widths_scaled[0]]
        y_1 = np.full((embedding.shape[0], 1), 1)
        spn_input_1 = tf.concat([tf.cast(y_1, dtype=tf.float32), embedding], axis=-1)
        y_0 = np.full((embedding.shape[0], 1), 0)
        spn_input_0 = tf.concat([tf.cast(y_0, dtype=tf.float32), embedding], axis=-1)
        for spn_id,(sub_spn, sub_priors) in enumerate(zip(self.all_spn_x_y,self.prior_weights)):
            res=[]
            for sub_sub_spn_id,sub_sub_spn_input in enumerate([spn_input_0,spn_input_1]):
                spn_log_pred=sub_spn[sub_sub_spn_id](sub_sub_spn_input, training=training)
                res.append(spn_log_pred)
                #if sub_sub_spn_id
                # Compute decision (argmax over log-probs)
            res=tf.concat(res, axis=1)
            res=res + tf.math.log(sub_priors)
            p_x = tf.reduce_logsumexp(res, axis=1,keepdims=True)
            p_y_x = res - p_x
            inputs_smooth.append(p_y_x[:,1] + tf.math.log(self.bin_widths_scaled[spn_id+1]))  # *tf.math.log(sub_sub_prior)

            decision = tf.argmax(res, axis=1)
            inputs_decision_boundary.append(tf.cast(decision, tf.float32) * self.bin_widths_scaled[spn_id+1])
        log_enumerator = tf.stack(inputs_smooth, axis=1)
        log_p_y_x = tf.reduce_logsumexp(log_enumerator, axis=1, keepdims=False)

        inputs_decision_boundary = tf.stack(inputs_decision_boundary, axis=1)  # children
        inputs_decision_boundary = tf.reduce_sum(inputs_decision_boundary, axis=1, keepdims=False)
        return log_p_y_x,inputs_decision_boundary
    '''

    def spn_regression_exp(self, embedding, training):
        batch_size = tf.shape(embedding)[0]

        inputs_smooth = [
            tf.math.log(tf.ones(shape=(batch_size,), dtype=tf.float32) * self.bin_widths_scaled[0])
        ]
        inputs_decision_boundary = [
            tf.ones(shape=(batch_size,), dtype=tf.float32) * self.bin_widths_scaled[0]
        ]

        y_1 = tf.ones((batch_size, 1), dtype=tf.float32)
        spn_input_1 = tf.concat([y_1, embedding], axis=-1)

        y_0 = tf.zeros((batch_size, 1), dtype=tf.float32)
        spn_input_0 = tf.concat([y_0, embedding], axis=-1)

        for spn_id, (sub_spn, sub_priors) in enumerate(zip(self.all_spn_x_y, self.prior_weights)):
            res = []
            for sub_sub_spn_id, sub_sub_spn_input in enumerate([spn_input_0, spn_input_1]):
                spn_log_pred = sub_spn[sub_sub_spn_id](sub_sub_spn_input, training=training)
                res.append(spn_log_pred)

            res = tf.concat(res, axis=1)
            res = res + tf.math.log(sub_priors)

            p_x = tf.reduce_logsumexp(res, axis=1, keepdims=True)
            p_y_x = res - p_x

            inputs_smooth.append(
                p_y_x[:, 1] + tf.math.log(self.bin_widths_scaled[spn_id + 1])
            )

            decision = tf.argmax(res, axis=1)
            inputs_decision_boundary.append(
                tf.cast(decision, tf.float32) * self.bin_widths_scaled[spn_id + 1]
            )

        log_enumerator = tf.stack(inputs_smooth, axis=1)
        log_p_y_x = tf.reduce_logsumexp(log_enumerator, axis=1, keepdims=False)

        inputs_decision_boundary = tf.stack(inputs_decision_boundary, axis=1)
        inputs_decision_boundary = tf.reduce_sum(inputs_decision_boundary, axis=1, keepdims=False)

        return log_p_y_x, inputs_decision_boundary

    def clf(self,embedding,training):
        #tf_weights = tf.expand_dims(self.prior_weights, axis=0) ?
        inputs = []
        for label_id, sub_spn in enumerate(self.all_spn_x_y):
            y = np.full((embedding.shape[0], 1), label_id)
            spn_input = tf.concat([tf.cast(y, dtype=tf.float32), embedding], axis=-1)
            inputs.append(sub_spn(spn_input, training=training))

        children_prob = tf.concat(inputs, axis=1)  # children
        # Calculate log-sum-exp
        log_enumerator = children_prob + tf.math.log(self.prior_weights)
        p_x = tf.reduce_logsumexp(log_enumerator, axis=1, keepdims=True)
        p_y_x = log_enumerator - p_x
        return p_y_x
