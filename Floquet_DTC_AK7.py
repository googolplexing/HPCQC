import os
import numpy as np
import matplotlib.pyplot as plt
import random
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.circuit.library import Measure
from qiskit.compiler import transpile
from iqm.qiskit_iqm import IQMProvider, IQMFakeAphrodite
import time

Q50_CORTEX_URL = os.getenv('Q50_CORTEX_URL')
#print(f"Q50_CORTEX_URL: {Q50_CORTEX_URL}")
provider = IQMProvider(Q50_CORTEX_URL, quantum_computer="q50")
#backend = provider.get_backend()
backend_FA = IQMFakeAphrodite()


num_qubits = 10
num_shots = 100         # q experiment runs
num_max_kicks = 40      # limit the kicks in result
num_gate_instances = 10
Jz = 1.                 # Floquet ZZ coupling
# Jzz = Jz * np.random.uniform(-1.5*np.pi, -0.5*np.pi)
Jzz = 0
h_z = 1.                # Floquet scrambling parameter
# hzz = np.random.uniform(-np.pi, np.pi)
hzz = 0
epsilon = 0.03         # deviation from 2pi kick
h_x = (1-epsilon)*np.pi   # transverse kick

#qc = QuantumCircuit(num_qubits)
#qc_loop = QuantumCircuit(num_qubits, num_qubits)
# circuit = QuantumCircuit(num_qubits, num_qubits)

Initial_state = 3       # 1:random, 2:Neel, 3:Polarized

autocorrelators = np.zeros(num_max_kicks)
init_bit_array = []

# helper function to get correlators from the string of results
def get_autocorrelation(counts, init_bit_array):
    total_shots = sum(counts.values())
    num_qub = len(list(counts.keys())[0])
    total_corr = 0

    for bitstring, count in counts.items():
        plus = 0
        minus = 0
        bit_array_little = np.array(list(bitstring), dtype=int)
        bit_array = bit_array_little[::-1]
        for wire in range(num_qubits):
            if bit_array[wire] == init_bit_array[wire]:
                plus += 1
            else:
                minus += 1
       
        temp_corr = (plus - minus) * count 

        total_corr += temp_corr
        
    return total_corr / (total_shots * num_qub)

# define a bit array for qubit initialization
def init_qubits_array(num_qubits):
    # random bit array
    if Initial_state == 1:
        for _ in range(num_qubits):
            init_bit_array.append(random.randint(0, 1))

    # Neel state 0,1,0,1,...
    elif Initial_state == 2:    
        for wire in range(num_qubits):
            init_bit_array.append(wire % 2)

    # fully polarized state
    elif Initial_state == 3:
        for _ in range(num_qubits):
            init_bit_array.append(0)

    else:
        raise ValueError(f"State must be 1-3, you set it {Initial_state}")


# One Floquet period
def apply_one_floquet_period(qc, hz_angles, Jzz_angles):
    # Transverse imperfect pi kick
    for wire in range(num_qubits):
        qc.rx(h_x, wire)

    # Random longitudinal z fields
    for wire in range(num_qubits):
        qc.rz(hz_angles[wire], wire)

    # Nearest-neighbour ZZ interactions
    for wire in range(num_qubits - 1):
        qc.rzz(Jzz_angles[wire], wire, wire + 1)


# Build circuit with n Floquet periods

def build_circuit(num_kicks, hz_angles, Jzz_angles, init_bit_array):

    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # set the initial state by x flips
    for wire in range(num_qubits):
        if init_bit_array[wire] == 1:
            qc.x(wire)

    for _ in range(num_kicks):
        apply_one_floquet_period(qc, hz_angles, Jzz_angles)

    qc.measure(range(num_qubits), range(num_qubits))

    return qc

# MAIN here:
# get initialization bit array
init_qubits_array(num_qubits)
print(init_bit_array)
# run different gate instances
start_time = time.time()
for n in range(num_gate_instances):
    print(f"Run gate instance {n}")
    Jz_angles = np.random.uniform(-1.5*np.pi, -0.5*np.pi, num_qubits)
    # Jz_angles = np.full(num_qubits, -0.4)   # tähän jaadytetty Jz
    hz_angles = np.random.uniform(-np.pi, np.pi, num_qubits)
    # build the circuits
    circuits = [
        build_circuit(n_kicks, hz_angles, Jz_angles, init_bit_array)
        for n_kicks in range(num_max_kicks)
    ]

    # initialize backend
    simulator = AerSimulator()                          # init simulator
    #compiled_circuit = transpile(circuits, simulator)    # be ready for HW backend
    compiled_circuit = transpile(circuits, backend_FA, optimization_level=3)    # be ready for HW backend
    
    # run with the defined backend
    #job = simulator.run(compiled_circuit, shots=num_shots, memory=True)
    job = backend_FA.run(compiled_circuit, shots=num_shots, memory=True)
    result = job.result()
    # counts = result.get_counts()

    for i in range(num_max_kicks):
        autocorrelators[i] += get_autocorrelation(result.get_counts(i), init_bit_array)

autocorrelators = autocorrelators / num_gate_instances

end_time = time.time()

print(f"Run time for {num_gate_instances} gate instances: {end_time-start_time}")

with open(f"Floquet_autocorr_data_IS{Initial_state}_GI{num_gate_instances}_Qs{num_qubits}_e{epsilon}_{num_max_kicks}_{num_shots}.dat",
           "w", encoding="utf-8") as outfile:
    for n in range(len(autocorrelators)):
        outfile.write(f"{n:4} {autocorrelators[n]:10.4} \n")
   
# Compute fft to get amplitudes   
fft_result = np.fft.fft(autocorrelators)
fs = len(fft_result)+1      # Sampling rate 
T = 1.0/fs                   # Sampling interval
N = len(fft_result)         # Number of samples
t = np.linspace(0.0, N*T, N)
y = np.abs(fft_result) / np.sum(abs(fft_result)) # normalized spectrum

# just increase font size in plots
plt.rcParams.update({'font.size': 14})

 # plot autocorrelation results
plt.figure(figsize=(8, 5))
plt.ylim(-1,1)
plt.plot(range(1,len(autocorrelators)+1), autocorrelators, 'o-') 
#         label=f'$\\epsilon=${epsilon}, $J_z=${Jz}, $h_z=${h_z}')
plt.xlabel("Driving Periods ($T$ steps)")
plt.ylabel("$\\langle A(0) A(T)\\rangle$")
#plt.title(f"DTC (Ising chain {num_qubits} qubits), X-pulse drive, shots={num_shots}")
#plt.title(f"No Heisenberg chain, X-pulse drive, shots={num_shots}")
plt.grid(True)
plt.legend()
plt.show()

# plot fourier amplitudes
plt.figure(figsize=(8, 5))
plt.xlim(0.35,0.65)
#plt.xlim(0.,1.)
plt.plot(t, y, 'o-') 
#         label=f'$\\epsilon=${epsilon}, $J_z=${Jz}, $h_z=${h_z}')
plt.xlabel("Frequency (1/T)")
plt.ylabel("Amplitude")
# plt.title(f"DTC (Ising chain {num_qubits} qubits): X-pulse drive, shots={num_shots}")
#plt.title(f"No Heisenberg chain, X-pulse drive, shots={num_shots}")
plt.grid(True)
plt.legend()
plt.show()
