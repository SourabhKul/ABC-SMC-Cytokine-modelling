import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

tf.device("GPU:0")

k1ap = pow(10,np.random.uniform(-7, 1))
k1am = pow(10,np.random.uniform(-2, 1))
k1bp = pow(10,np.random.uniform(-7, 1))
k1bm = pow(10,np.random.uniform(-2, 1))
k3ap = pow(10,np.random.uniform(-7, 1))
k3am = pow(10,np.random.uniform(-2, 1))
k3bp = pow(10,np.random.uniform(-7, 1))
k3bm = pow(10,np.random.uniform(-2, 1))
qx   = pow(10,np.random.uniform(-3, 2))
d1   = pow(10,np.random.uniform(-5,-2))
d3   = pow(10,np.random.uniform(-5,-2))
r10   = np.random.normal(12.7,6.35)
r20   = np.random.normal(33.8,16.9)
s10   = np.random.normal(300,100)
s30   = np.random.normal(400,100)
r1p = pow(10,np.random.normal(-2.34, 1.17))
r1m = pow(10,np.random.normal(-2.82, 1.41))
r2p = pow(10,np.random.uniform(-2, 3))
r2m = pow(10,np.random.uniform(-3, 1))
beta = pow(10,np.random.uniform(-5,-1))
gamma = 0
outputs = [] #Empty list for the model outputs.
paramsa = [] #Empty list for the parameters.
paramsa.append(np.array((r1p,r1m,r2p,r2m,beta)))

parset = np.array((k1ap, k1am, k1bp, k1bm, k3ap, k3am, k3bp, k3bm, qx, d1, d3, r10, r20, s10, s30, r1p, r1m, r2p, r2m, beta, 0)) #Parameter set to feed into the integrator.
R0 = np.array([r10,10,0,0,s10,s30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) #Initial concentrations.
# R = integrate.odeint(dIL27_dt, R0, tt, args=(parset,)) #Integrate the mathematical model.
# n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13, n14, n15, n16, n17, n18, n19, n20, n21, n22, n23, n24, n25, n26, n27, n28, n29, n30, n31, n32, n33 = R.T #Model outputs for each vairable.
# pS1 = n12 + n13 + n17 + n18 + 2*n19 + n26 + n29 + n30 + n31 + n32 #Sum of variables containing pSTAT1.
# pS3 = n14 + n15 + n21 + n22 + 2*n23 + n27 + n28 + n30 + n31 + n33 #Sum of variables containing pSTAT3.
# outputs = [] #Empty list for the model outputs.
# paramsa = [] #Empty list for the parameters.
# Define the parameters
params = tf.constant([k1ap, k1am, k3ap, k3am, qx, d1, d3, r10, s10, s30, r1p, r1m, r2p, r2m, beta, gamma], dtype=tf.float32)

class ODEFunc(tf.keras.layers.Layer):
    def __init__(self, params):
        super(ODEFunc, self).__init__()
        self.params = params

    def call(self, t, R):
        k1ap, k1am, k3ap, k3am, qx, d1, d3, r10, s10, s30, r1p, r1m, r2p, r2m, beta, gamma = self.params

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

# Define the Neural ODE model
class NeuralODEModel(tf.keras.Model):
    def __init__(self, ode_func):
        super(NeuralODEModel, self).__init__()
        self.ode_func = ode_func

    def call(self, t, R):
        return self.ode_func(t, R)

# Initial conditions and time span
R0 = tf.constant(R0, dtype=tf.float32)
t = tf.linspace(0.0, 10.0, num=100)  # Example time points

# Create the ODE function and model
ode_func = ODEFunc(params)
neural_ode_model = NeuralODEModel(ode_func)

# Use the tfp.math.ode solver to solve the ODE
solver = tfp.math.ode.DormandPrince(atol=1e-5, rtol=1e-5)
print("running solver now")
results = solver.solve(neural_ode_model, initial_time=t[0], initial_state=R0, solution_times=t)

# Extract the solution
solution = results.states.numpy
print(solution())