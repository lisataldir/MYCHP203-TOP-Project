import numpy as np
import matplotlib.pyplot as plt

x = 2**np.arange(5)

# On récupère les données mpi
mpi_med = []
for n in range(5):
    with open(f'../output_files/output_mpi{2**n}.txt', 'r') as f:
        for line in f:
            tmp = [float(num) for num in line.split()]
        mpi_med.append(np.median(tmp)*(2**n))

mpi_med = np.array(mpi_med)

# Plot
fig, ax = plt.subplots()
ax.plot(x, mpi_med, marker='*', color='navy', label='Médiane')
ax.plot(x, x[0]*np.ones(5), linestyle='--', color='#cb3717', label='Accélération idéale')
ax.set_xlabel('Quantité de travail')
ax.set_ylabel('Temps (en ms)')
ax.legend()
ax.grid(True)
plt.savefig(f'weak_scalability.png')

# On récupère les données omp
omp_med = []
for n in range(5):
    with open(f'../output_files/output_omp{2**n}.txt', 'r') as f:
        for line in f:
            tmp = [float(num) for num in line.split()]
        omp_med.append(np.median(tmp))

omp_med = np.array(omp_med) 
ideal_omp = omp_med[0] / x

# Plot
fig, ax = plt.subplots()
ax.plot(x, omp_med, marker='*', color='navy', label='Médiane')
ax.plot(x, ideal_omp, linestyle='--', color='#cb3717', label='Accélération idéale')
ax.set_xlabel('Nombre de threads')
ax.set_ylabel('Temps (en ms)')
ax.legend()
ax.grid(True)
plt.savefig(f'strong_scalability.png')