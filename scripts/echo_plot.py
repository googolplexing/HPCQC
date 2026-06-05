import matplotlib.pyplot as plt 
import numpy as np

# Load echo circuit data
echo_data = np.load("C:\\Users\\inkahilt\\time crystals\\results\\echo_yonyli4.npz", allow_pickle=True)
echo_autocorrelation = echo_data["autocorrelators"]
echo_init_state = echo_data["init_state"]
#echo_t = echo_data["frequencies"]
#echo_y = echo_data["spectrum"]
echo_params = echo_data["params"].item()   

#print("Loaded parameters:", echo_params)
print("Initial state:", echo_init_state)

# Load forward circuit data
forward_data = np.load("C:\\Users\\inkahilt\\time crystals\\results\\forward_yonyli4.npz", allow_pickle=True)
forward_autocorrelation = forward_data["auto"]

# Normalize
normalized = forward_autocorrelation / echo_autocorrelation

# just increase font size in plots
plt.rcParams.update({'font.size': 14})
 # plot autocorrelation results
plt.figure(figsize=(8, 5))
plt.plot(range(1,len(normalized)+1), normalized, marker='o', color='darkseagreen', label="$\\langle A(0)A(T)\\rangle / A_0$")
plt.plot(range(1,len(forward_autocorrelation)+1), forward_autocorrelation, marker='o', color='hotpink', label='Autocorrelation')
plt.plot(range(1,len(echo_autocorrelation)+1), echo_autocorrelation, 'o-', color='black', label='Echo') 

plt.xlabel("Driving Periods ($T$ steps)")
plt.ylabel("$\\langle A(0) A(T)\\rangle$")
plt.title("Disorder Averaged Autocorrelation")
#plt.title(f"DTC (Ising chain {num_qubits} qubits), X-pulse drive, shots={num_shots}")
#plt.title(f"No Heisenberg chain, X-pulse drive, shots={num_shots}")
plt.grid(True)
plt.legend(loc='lower right', fontsize=10)
plt.show()
