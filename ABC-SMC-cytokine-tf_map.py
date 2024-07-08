import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
import multiprocessing
print(f"GPUS detected: {tf.config.list_physical_devices('GPU')}")
tf.device("/CPU:0")
# Define the parameters
def generate_random_params():
    return {
        "k1ap": pow(10, np.random.uniform(-7, 1)),
        "k1am": pow(10, np.random.uniform(-2, 1)),
        "k1bp": pow(10, np.random.uniform(-7, 1)),
        "k1bm": pow(10, np.random.uniform(-2, 1)),
        "k3ap": pow(10, np.random.uniform(-7, 1)),
        "k3am": pow(10, np.random.uniform(-2, 1)),
        "k3bp": pow(10, np.random.uniform(-7, 1)),
        "k3bm": pow(10, np.random.uniform(-2, 1)),
        "qx": pow(10, np.random.uniform(-3, 2)),
        "d1": pow(10, np.random.uniform(-5, -2)),
        "d3": pow(10, np.random.uniform(-5, -2)),
        "r10": np.random.normal(12.7, 6.35),
        "r20": np.random.normal(33.8, 16.9),
        "s10": np.random.normal(300, 100),
        "s30": np.random.normal(400, 100),
        "r1p": pow(10, np.random.normal(-2.34, 1.17)),
        "r1m": pow(10, np.random.normal(-2.82, 1.41)),
        "r2p": pow(10, np.random.uniform(-2, 3)),
        "r2m": pow(10, np.random.uniform(-3, 1)),
        "beta": pow(10, np.random.uniform(-5, -1)),
        "gamma": 0
    }

class ODEFunc(tf.keras.layers.Layer):
    def __init__(self, params):
        super(ODEFunc, self).__init__()
        self.params = params

    
    @tf.function()
    def call(self, t, R):
        params_values = tf.unstack(self.params)
        k1ap, k1am, k1bp, k1bm, k3ap, k3am, k3bp, k3bm, qx, d1, d3, r10, r20, s10, s30, r1p, r1m, r2p, r2m, beta, gamma = params_values

        dRdt = tf.stack([
            -r1p*R[0]*R[1] + r1m*R[2] - beta*R[0] - gamma*(R[20]+R[21])*R[0],
            -r1p*R[0]*R[1] + r1m*R[2],
            r1p*R[0]*R[1] - r1m*R[2] - 2*r2p*R[2]**2 + 2*r2m*R[3] - beta*R[2] - gamma*(R[20]+R[21])*R[2],
            r2p*R[2]**2 - r2m*R[3] - 2*k1ap*R[3]*R[4] + k1am*R[6] - 2*k3ap*R[3]*R[5] + k3am*R[7] + k1am*R[8] + k3am*R[9] - beta*R[3] - gamma*(R[20]+R[21])*R[3],
            -k1ap*R[4]*(R[6]+R[7]+R[8]+R[9]) + k1am*(2*R[10]+R[16]+R[11]+R[18]) - 2*k1ap*R[3]*R[4] + k1am*R[6] + d1*R[20],
            -k3ap*R[5]*(R[6]+R[7]+R[8]+R[9]) + k3am*(2*R[13]+R[16]+R[14]+R[17]) - 2*k3ap*R[3]*R[5] + k3am*R[7] + d3*R[21],
            2*k1ap*R[3]*R[4] - k1am*R[6] - k1ap*R[6]*R[4] + 2*k1am*R[10] - k3ap*R[6]*R[5] + k3am*R[16] - qx*R[6] + k1am*R[11] + k3am*R[18] - beta*R[6] - gamma*(R[20]+R[21])*R[6],
            2*k3ap*R[3]*R[5] - k3am*R[7] - k3ap*R[7]*R[5] + 2*k3am*R[13] - k1ap*R[7]*R[4] + k1am*R[16] - qx*R[7] + k3am*R[14] + k1am*R[17] - beta*R[7] - gamma*(R[20]+R[21])*R[7],
            -k1ap*R[4]*R[8] + k1am*R[11] - k3ap*R[5]*R[8] + k3am*R[17] + qx*R[6] - k1am*R[8] + 2*k1am*R[12] + k3am*R[19] - beta*R[8] - gamma*(R[20]+R[21])*R[8],
            -k3ap*R[5]*R[9] + k3am*R[14] - k1ap*R[4]*R[9] + k1am*R[18] + qx*R[7] - k3am*R[9] + 2*k3am*R[15] + k1am*R[19] - beta*R[9] - gamma*(R[20]+R[21])*R[9],
            k1ap*R[4]*R[6] - 2*k1am*R[10] - 2*qx*R[10] - beta*R[10] - gamma*(R[20]+R[21])*R[10],
            k1ap*R[8]*R[4] - k1am*R[11] + 2*qx*R[10] - (qx + k1am)*R[11] - beta*R[11] - gamma*(R[20]+R[21])*R[11],
            qx*R[11] - 2*k1am*R[12] - beta*R[12] - gamma*(R[20]+R[21])*R[12],
            k3ap*R[5]*R[7] - 2*k3am*R[13] - 2*qx*R[13] - beta*R[13] - gamma*(R[20]+R[21])*R[13],
            k3ap*R[9]*R[5] - k3am*R[14] + 2*qx*R[13] - (qx + k3am)*R[14] - beta*R[14] - gamma*(R[20]+R[21])*R[14],
            qx*R[14] - 2*k3am*R[15] - beta*R[15] - gamma*(R[20]+R[21])*R[15],
            k1ap*R[4]*R[7] - k1am*R[16] + k3ap*R[6]*R[5] - k3am*R[16] - 2*qx*R[16] - beta*R[16] - gamma*(R[20]+R[21])*R[16],
            qx*R[16] + k3ap*R[8]*R[5] - k3am*R[17] - qx*R[17] - k1am*R[17] - beta*R[17] - gamma*(R[20]+R[21])*R[17],
            qx*R[16] + k1ap*R[9]*R[4] - k1am*R[18] - qx*R[18] - k3am*R[18] - beta*R[18] - gamma*(R[20]+R[21])*R[18],
            qx*R[18] + qx*R[17] - k1am*R[19] - k3am*R[19] - beta*R[19] - gamma*(R[20]+R[21])*R[19],
            k1am*(R[8]+R[11]+R[17]+R[19]) + 2*k1am*R[12] - d1*R[20],
            k3am*(R[9]+R[14]+R[18]+R[19]) + 2*k3am*R[15] - d3*R[21],
        ])
        return dRdt



class NeuralODEModel(tf.keras.Model):
    def __init__(self, ode_func):
        super(NeuralODEModel, self).__init__()
        self.ode_func = ode_func

    @tf.function
    def call(self, t, R):
        return self.ode_func(t, R)
# @tf.function
def run_e2e(times):
    for num_instances in times:
        start_time = time.time()
        @tf.function
        def run_ode_instance(params_list, initial_conditions_list, time_points_list):
            def solve_single_instance(params, initial_conditions, time_points):
                ode_func = ODEFunc(params)
                neural_ode_model = NeuralODEModel(ode_func)
                solver = tfp.math.ode.DormandPrince(atol=1e-5, rtol=1e-5)
                results = solver.solve(neural_ode_model, initial_time=time_points[0], initial_state=initial_conditions, solution_times=time_points)
                return results.states

            # Use tf.map_fn to parallelize the execution
            solutions = tf.map_fn(lambda args: solve_single_instance(*args), (params_list, initial_conditions_list, time_points_list), dtype=tf.float32)
            return solutions

        # Generate multiple sets of random parameters and initial conditions
        # num_instances = 200
        params_list = [generate_random_params() for _ in range(num_instances)]
        initial_conditions_list = [np.random.rand(22).astype(np.float32) for _ in range(num_instances)]

        params_list_tf = tf.constant([list(params.values()) for params in params_list], dtype=tf.float32)
        initial_conditions_list_tf = tf.constant(initial_conditions_list, dtype=tf.float32)
        # Repeat the creation of time_points_tf num_instances times
        time_points_list_tf =tf.convert_to_tensor([tf.constant(np.linspace(0.0, 10.0, num=100).astype(np.float32), dtype=tf.float32) for _ in range(num_instances)])
        # Call run_ode_instance with the individual tensors
        solutions = run_ode_instance(params_list_tf, initial_conditions_list_tf, time_points_list_tf)
        print(f"total time for {num_instances} instances was {time.time() - start_time} seconds")
        # for i, solution in enumerate(solutions):
        #     print(f"Solution {i}:\n{solution}")

run_e2e([10, 25, 50, 100, 150, 200])