import tensorflow as tf



#Leraning rates schedule
class TransformerLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    A custom learning rate schedule for transformer models.

    Args:
        d_model: The dimensionality of the model.
        warmup_steps: Number of warmup steps.
        custom_learning_rate_multiplication_factor: Custom multiplication factor for the learning rate.

    Returns:
        The learning rate for the given step.
    """
    def __init__(self, d_model, warmup_steps=4000, custom_lernaing_rate_multiplication_factor=1):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.custom_factor=custom_lernaing_rate_multiplication_factor

        self.warmup_steps = warmup_steps

    def __call__(self, step):

        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        lr = (tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)) * self.custom_factor

        return lr

    def get_config(self):
        config = {
            'd_model':self.d_model,
            'warmup_steps':self.warmup_steps,            
        }
        return config


class ExponentialDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    A custom learning rate schedule using exponential decay with warmup.

    Args:
        warmup_steps: Number of warmup steps.
        initial_learning_rate: Initial learning rate.
        decay_steps: Decay steps.
        decay_rate: Decay rate.

    Returns:
        The learning rate for the given step.
    """ 
    def __init__(self, warmup_steps, inital_learning_rate, decay_steps, decay_rate):
        super().__init__()

        self.exp_decay = tf.keras.optimizers.schedules.ExponentialDecay(inital_learning_rate, decay_steps= decay_steps, decay_rate = decay_rate)
        self.warmup_steps=warmup_steps
        self.inital_learning_rate=inital_learning_rate
        self.decay_steps=decay_steps
        self.decay_rate=decay_rate

    def __call__(self, step):

        step = tf.cast(step, dtype=tf.float32)
        arg1 = self.exp_decay(step-self.warmup_steps)
        arg2 = (step/self.warmup_steps) * self.inital_learning_rate
        
        rate = tf.math.minimum(arg1, arg2) 
        return rate
    
    def get_config(self):
        config = {
            'inital_learning_rate':self.inital_learning_rate,
            'warmup_steps':self.warmup_steps,
            'decay_steps':self.decay_steps,
            'decay_rate':self.decay_rate,
            
        }
        return config

class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    A custom learning rate schedule using cosine decay with warmup.

    Args:
        warmup_steps: Number of warmup steps.
        initial_learning_rate: Initial learning rate.
        decay_steps: Decay steps.

    Returns:
        The learning rate for the given step.
    """
    def __init__(self, warmup_steps, inital_learning_rate, decay_steps):
        super().__init__()

        self.cos_decay = tf.keras.optimizers.schedules.CosineDecay(inital_learning_rate, decay_steps= decay_steps)
        self.warmup_steps=warmup_steps
        self.inital_learning_rate=inital_learning_rate
        self.decay_steps = decay_steps

    def __call__(self, step):

        step = tf.cast(step, dtype=tf.float32)
        arg1 = self.cos_decay(step-self.warmup_steps)
        arg2 = (step/self.warmup_steps) * self.inital_learning_rate
        
        return tf.math.minimum(arg1, arg2) 
    
    def get_config(self):
        config = {
            'inital_learning_rate':self.inital_learning_rate,
            'warmup_steps':self.warmup_steps,
            'decay_steps':self.decay_steps,
        }
        return config

class CosineDecayRestartsWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    A custom learning rate schedule using cosine decay restarts with warmup.

    Args:
        warmup_steps: Number of warmup steps.
        initial_learning_rate: Initial learning rate.
        decay_steps: Decay steps.
        t_mul: Time multiplication factor.
        m_mul: Decay multiplication factor.

    Returns:
        The learning rate for the given step.
    """
    def __init__(self, warmup_steps, inital_learning_rate, decay_steps, t_mul=2.0, m_mul=0.7):
        super().__init__()

        self.cos_decay = tf.keras.optimizers.schedules.CosineDecayRestarts(inital_learning_rate, decay_steps, t_mul, m_mul, 0.00001)
        self.decay_steps=decay_steps
        self.t_mul=t_mul
        self.m_mul=m_mul        
        self.warmup_steps=warmup_steps
        self.inital_learning_rate=inital_learning_rate

    def __call__(self, step):

        step = tf.cast(step, dtype=tf.float32)
        arg1 = self.cos_decay(step)
        arg2 = (step/self.warmup_steps) * self.inital_learning_rate
        
        return tf.math.minimum(arg1, arg2) 
    
    def get_config(self):
        config = {
            'inital_learning_rate':self.inital_learning_rate,
            'warmup_steps':self.warmup_steps,
            'decay_steps':self.decay_steps,
            't_mul':self.t_mul,
            'm_mul':self.m_mul,
        }
        return config


