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

num_qubits = 10
num_shots = 100         # q experiment runs
num_max_kicks = 40      # limit the kicks in result
num_gate_instances = 10 # number of different random gate sets
epsilon = 0.03          # deviation from 2pi kick
h_x = (1-epsilon)*np.pi # transverse kick
Initial_state = 3       # 1:random, 2:Neel, 3:Polarized
Jz_on = True            # spin-spin interaction on
hz_on = True            # random z field on
Jz_frozen = False       # constant Jz coupling. If True, overrides random Jz
Jz_frozen_value = -0.04 # value of constant Jz
backend_code = 1        # 1:AerSimulator, 2:IQMFakeAphrodite, 3:Q50 
random_seed = 1234

def get_backend(backend_code):
    if backend_code == 1:
        backend = AerSimulator()
    elif backend_code == 2:
        backend = IQMFakeAphrodite()
    elif backend_code == 3:
        Q50_CORTEX_URL = os.getenv('Q50_CORTEX_URL')
        provider = IQMProvider(Q50_CORTEX_URL, quantum_computer="q50")
        backend = provider.get_backend()
    else:
        raise ValueError(f"backend_code must be 1-3. You provided {backend_code}")
    
    return backend

# helper function to get correlators from the string of results
def get_autocorrelation(counts, init_bit_array):
    # edge_init_bits = [init_bit_array[0], init_bit_array[num_qubits-1]]
    total_shots = sum(counts.values())
    num_qub = len(list(counts.keys())[0])
    total_corr = 0

    for bitstring, count in counts.items():
        plus = 0
        bit_array_little = np.array(list(bitstring), dtype=int)
        bit_array = bit_array_little[::-1]
        for wire in range(num_qub):
            if bit_array[wire] == init_bit_array[wire]:
                plus += 1
            else:
                plus -= 1

        total_corr += plus * count
        
    return total_corr / (total_shots * num_qub)


def get_autocorrelation_per_qubit(counts, init_bit_array):
    """
    Compute autocorrelation for each qubit separately.
    Returns a numpy array of length num_qubits.
    """
    total_shots = sum(counts.values())
    num_qub = len(list(counts.keys())[0])
    total_corr = np.zeros(num_qub, dtype=float)

    for bitstring, count in counts.items():
        bit_array_little = np.array(list(bitstring), dtype=int)
        bit_array = bit_array_little[::-1]
        matches = np.where(bit_array == init_bit_array, 1.0, -1.0)
        total_corr += matches * count

    return total_corr / total_shots


# define a bit array for qubit initialization
def init_qubits_array(num_qubits):
    init_bit_array = []
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
        raise ValueError(f"Initial state must be 1-3, you set it {Initial_state}")

    return init_bit_array

# One Floquet period
def apply_one_floquet_period(qc, hz_angles, Jzz_angles, epsilon_local=None):
    # Transverse imperfect pi kick
    if epsilon_local is None:
        epsilon_local = epsilon
    h_x_local = (1 - epsilon_local) * np.pi
    for wire in range(num_qubits):
        qc.rx(h_x_local, wire)

    # Random longitudinal z fields
    rng = random
    for wire in range(num_qubits):
        #hz_angle = random.uniform(-np.pi, np.pi)
        qc.rz(hz_angles[wire], wire)
        #qc.rz(hz_angle, wire)

    # Nearest-neighbour ZZ interactions
    for wire in range(num_qubits - 1):
        qc.rzz(Jzz_angles[wire] / 2, wire, wire + 1)

def apply_one_Floquet_period_conjugate(qc, hz_angles, Jzz_angles, epsilon_local=None):
    if epsilon_local is None:
        epsilon_local = epsilon
    h_x_local = (1 - epsilon_local) * np.pi

    # Nearest-neighbour ZZ interactions
    for wire in range(num_qubits - 1):
        qc.rzz(-Jzz_angles[wire]/2, wire, wire + 1)

    # Random longitudinal z fields
    for wire in range(num_qubits):
        qc.rz(-hz_angles[wire], wire)

    # Transverse imperfect pi kick
    for wire in range(num_qubits):
        qc.rx(-h_x_local, wire)

# Build circuit with num_kicks Floquet periods
def build_autocorr_circuit(num_kicks, hz_angles, Jzz_angles, init_bit_array, epsilon_local=None):

    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # set the initial state by x flips
    for wire in range(num_qubits):
        if init_bit_array[wire] == 1:
            qc.x(wire)

    for _ in range(num_kicks):
        apply_one_floquet_period(qc, hz_angles, Jzz_angles, epsilon_local=epsilon_local)

    qc.measure(range(num_qubits), range(num_qubits))

    return qc

# Build echo circuit with num_kicks Floquet periods
def build_echo_circuit(num_kicks, hz_angles, Jzz_angles, init_bit_array, epsilon_local=None):

    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # set the initial state by x flips
    for wire in range(num_qubits):
        if init_bit_array[wire] == 1:
            qc.x(wire)

    for _ in range(num_kicks):
        apply_one_floquet_period(qc, hz_angles, Jzz_angles, epsilon_local=epsilon_local)

    for _ in range(num_kicks):
        apply_one_Floquet_period_conjugate(qc, hz_angles, Jzz_angles, epsilon_local=epsilon_local)

    qc.measure(range(num_qubits), range(num_qubits))

    return qc

def build_echo_instance_and_run(Jz_angles, hz_angles, init_bit_array, epsilon_local=None):
    echo_values = []
    
    echo_circuits = [
        build_echo_circuit(n_kicks, hz_angles, Jz_angles, init_bit_array, epsilon_local=epsilon_local)
        for n_kicks in range(num_max_kicks)
    ]
 
    backend = get_backend(backend_code)
    compiled_echo_circuit = transpile(echo_circuits, backend, optimization_level=3)    # be ready for HW backend
    echo_job = backend.run(compiled_echo_circuit, shots=num_shots)
    echo_result = echo_job.result()

    for i in range(num_max_kicks):
        temp_autocorrelation = get_autocorrelation(echo_result.get_counts(i), init_bit_array)
        echo_values.append(np.sqrt(np.abs(temp_autocorrelation)))
    return echo_values


def build_echo_instance_and_run_per_qubit(Jz_angles, hz_angles, init_bit_array, epsilon_local=None):
    echo_per_qubit = np.zeros((num_qubits, num_max_kicks))

    echo_circuits = [
        build_echo_circuit(n_kicks, hz_angles, Jz_angles, init_bit_array, epsilon_local=epsilon_local)
        for n_kicks in range(num_max_kicks)
    ]

    backend = get_backend(backend_code)
    compiled_echo_circuit = transpile(echo_circuits, backend, optimization_level=3)
    echo_job = backend.run(compiled_echo_circuit, shots=num_shots)
    echo_result = echo_job.result()

    for i in range(num_max_kicks):
        temp_autocorrelation = get_autocorrelation_per_qubit(
            echo_result.get_counts(i), init_bit_array
        )
        echo_per_qubit[:, i] = np.sqrt(np.abs(temp_autocorrelation))

    return echo_per_qubit


def build_autocorr_instance_and_run(Jz_angles, hz_angles, init_bit_array, epsilon_local=None):
    autocorrelators = []
    circuits = [
        build_autocorr_circuit(n_kicks, hz_angles, Jz_angles, init_bit_array, epsilon_local=epsilon_local)
        for n_kicks in range(num_max_kicks)
    ]

    # initialize backend and run experiment
    backend = get_backend(backend_code)
    compiled_circuit = transpile(circuits, backend, optimization_level=3)    # be ready for HW backend
    job = backend.run(compiled_circuit, shots=num_shots)
    result = job.result()
 
    for i in range(num_max_kicks):
        autocorrelators.append(get_autocorrelation(result.get_counts(i), init_bit_array))

    return autocorrelators


def build_autocorr_instance_and_run_per_qubit(Jz_angles, hz_angles, init_bit_array, epsilon_local=None):
    autocorr_per_qubit = np.zeros((num_qubits, num_max_kicks))
    circuits = [
        build_autocorr_circuit(n_kicks, hz_angles, Jz_angles, init_bit_array, epsilon_local=epsilon_local)
        for n_kicks in range(num_max_kicks)
    ]

    backend = get_backend(backend_code)
    compiled_circuit = transpile(circuits, backend, optimization_level=3)
    job = backend.run(compiled_circuit, shots=num_shots)
    result = job.result()

    for i in range(num_max_kicks):
        autocorr_per_qubit[:, i] = get_autocorrelation_per_qubit(
            result.get_counts(i), init_bit_array
        )

    return autocorr_per_qubit


def run_autocorr_and_echo(Jz_angles, hz_angles, init_bit_array, epsilon_local=None):
    autocorrelators = build_autocorr_instance_and_run(
        Jz_angles, hz_angles, init_bit_array, epsilon_local=epsilon_local
    )
    echo_values = build_echo_instance_and_run(
        Jz_angles, hz_angles, init_bit_array, epsilon_local=epsilon_local
    )
    return autocorrelators, echo_values


def run_autocorr_and_echo_per_qubit(Jz_angles, hz_angles, init_bit_array, epsilon_local=None):
    autocorrelators = build_autocorr_instance_and_run_per_qubit(
        Jz_angles, hz_angles, init_bit_array, epsilon_local=epsilon_local
    )
    echo_values = build_echo_instance_and_run_per_qubit(
        Jz_angles, hz_angles, init_bit_array, epsilon_local=epsilon_local
    )
    return autocorrelators, echo_values


def output_stem():  
    if Jz_on == True:
        Jz = 1
    else:
        Jz = 0
    if hz_on == True:
        hz = 1
    else:
        hz = 0
    if Jz_frozen == True:
        Jz_val = Jz_frozen_value
    else:
        Jz_val = 0
    stem = f"BE{backend_code}_{Jz}_{hz}_{Jz_val}_IS{Initial_state}_GI{num_gate_instances}_Qs{num_qubits}_e{epsilon}_{num_max_kicks}_{num_shots}"
    return stem

def autocorr_data_to_file(autocorrelators):
    with open(f"Floquet_autocorr_data_{output_stem()}.dat",
               "w", encoding="utf-8") as outfile:
        for n in range(len(autocorrelators)):
            outfile.write(f"{n:4} {autocorrelators[n]} \n")
   
def echo_data_to_file(echo_data):
    with open(f"Floquet_echo_data_{output_stem()}.dat",
               "w", encoding="utf-8") as outfile:
        for n in range(len(echo_data)):
            outfile.write(f"{n:4} {echo_data[n]} \n")

def plot_results(autocorrelators, echo_data):
    # Compute fft to get amplitudes   
    fft_result = np.fft.fft(autocorrelators)
    fs = len(fft_result)+1      # Sampling rate 
    T = 1.0/fs                   # Sampling interval
    N = len(fft_result)         # Number of samples
    t = np.linspace(0.0, N*T, N)
    y = np.abs(fft_result) / np.sum(abs(fft_result)) # normalized spectrum

    normalized_data = []
    for n in range(len(autocorrelators)):
        normalized_data.append(autocorrelators[n]/echo_data[n])

    # just increase font size in plots
    plt.rcParams.update({'font.size': 14})

    # plot autocorrelation results
    plt.figure(figsize=(8, 5))
    plt.ylim(-1,1)
    plt.plot(range(1,len(autocorrelators)+1), autocorrelators, 'o-', 
            label=f'$A(0)A(T)$')
    plt.plot(range(1,len(echo_data)+1), echo_data, 'o-', 
            label=f'Echo $A_0$')
    plt.plot(range(1,len(normalized_data)+1), normalized_data, 'o-', 
            label=f'$A(0)A(T)/A_0$')
    
    plt.xlabel("Driving Periods ($T$ steps)")
    #plt.ylabel("$\\langle A(0) A(T)\\rangle$")
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
# end plot_results()

def get_normalized_data(autocorrelators, echo):
    norm_data = []
    for n in range(len(autocorrelators)):
        norm_data.append(autocorrelators[n] / echo[n])
    return norm_data


def get_normalized_data_per_qubit(autocorrelators_per_qubit, echo_per_qubit):
    return np.divide(
        autocorrelators_per_qubit,
        echo_per_qubit,
        out=np.zeros_like(autocorrelators_per_qubit, dtype=float),
        where=echo_per_qubit != 0,
    )


def get_normalized_fft_central_value(normalized_autocorrelations):
    """
    Return the normalized Fourier amplitude at the central frequency, f ~= 0.5.
    """
    fft_result = np.fft.fft(normalized_autocorrelations)
    amplitudes = np.abs(fft_result)
    total_amplitude = np.sum(amplitudes)
    if total_amplitude == 0:
        return 0.0

    normalized_amplitudes = amplitudes / total_amplitude
    freqs = np.fft.fftfreq(len(normalized_autocorrelations))
    central_index = int(np.argmin(np.abs(np.abs(freqs) - 0.5)))
    return normalized_amplitudes[central_index]


def get_normalized_fft_central_values_per_qubit(normalized_autocorrelations_per_qubit):
    """
    Return the central Fourier amplitude for each qubit separately.
    Input shape should be (num_qubits, num_kicks).
    """
    return np.array([
        get_normalized_fft_central_value(qubit_series)
        for qubit_series in normalized_autocorrelations_per_qubit
    ])


def get_central_fft_variance_over_qubits(normalized_autocorrelations_per_qubit):
    central_values = get_normalized_fft_central_values_per_qubit(
        normalized_autocorrelations_per_qubit
    )
    return np.var(central_values)


def get_random_angles(rng):
    if Jz_on == True:
        Jz_angles = rng.uniform(-1.5*np.pi, -0.5*np.pi, num_qubits)
    else:
        Jz_angles = np.zeros(num_qubits)
    if Jz_frozen == True:
        Jz_angles = np.full(num_qubits, Jz_frozen_value)

    if hz_on == True:
        hz_angles = rng.uniform(-np.pi, np.pi, num_qubits)
    else:
        hz_angles = np.zeros(num_qubits)

    return Jz_angles, hz_angles


def compute_variance_for_epsilon(epsilon_local, init_bit_array):
    total_autocorrelators_per_qubit = np.zeros((num_qubits, num_max_kicks))
    total_echo_values_per_qubit = np.zeros((num_qubits, num_max_kicks))
    rng = np.random.default_rng(random_seed)

    for instance_index in range(num_gate_instances):
        print(f"epsilon={epsilon_local:.3f}, gate instance {instance_index}")
        Jz_angles, hz_angles = get_random_angles(rng)
 
        temp_autocorrelators_per_qubit, temp_echo_values_per_qubit = run_autocorr_and_echo_per_qubit(
            Jz_angles, hz_angles, init_bit_array, epsilon_local=epsilon_local
        )

        total_autocorrelators_per_qubit += temp_autocorrelators_per_qubit
        total_echo_values_per_qubit += temp_echo_values_per_qubit

    normal_autocorrelators_per_qubit = total_autocorrelators_per_qubit / num_gate_instances
    normal_echo_values_per_qubit = total_echo_values_per_qubit / num_gate_instances

    normalized_data_per_qubit = get_normalized_data_per_qubit(
        normal_autocorrelators_per_qubit, normal_echo_values_per_qubit
    )

    return get_central_fft_variance_over_qubits(normalized_data_per_qubit)


def plot_variance_vs_epsilon(epsilon_values, variance_values):
    np.savetxt(
        f"echo_variance_vs_epsilon_{output_stem()}.dat",
        np.column_stack((epsilon_values, variance_values)),
        header="epsilon variance",
    )
    plt.figure(figsize=(8, 5))
    plt.plot(epsilon_values, variance_values, 'o-', label='Var(central FFT)')
    plt.xlabel(r'$\epsilon$')
    plt.ylabel('Variance of central FFT values over qubits')
    plt.title('Echo-normalized central FFT variance vs. epsilon')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"echo_variance_vs_epsilon_{output_stem()}.png", dpi=300)
    plt.show()


def main():
    epsilon_values = np.arange(0.0, 0.5 + 1e-12, 0.01)
    variance_values = []
    init_bit_array = init_qubits_array(num_qubits)
    print(init_bit_array)
    start_time = time.time()

    for epsilon_local in epsilon_values:
        variance_values.append(compute_variance_for_epsilon(epsilon_local, init_bit_array))

    end_time = time.time()
    variance_values = np.array(variance_values)
    print(f"Run time for epsilon sweep with {num_gate_instances} gate instances: {end_time-start_time}")
    plot_variance_vs_epsilon(epsilon_values, variance_values)

    # end main()

# run the main:
if __name__ == "__main__":
    main()
