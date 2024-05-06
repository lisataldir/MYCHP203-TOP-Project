import numpy as np
import matplotlib.pyplot as plt

x_omp = 2**np.arange(5)
x_mpi = np.arange(1,9)

# On récupère les données mpi v2
mpi_sum = []
for n in range(1,9):
    with open(f'../output_files/output_mpi{n}_v2.txt', 'r') as f:
        for line in f:
            tmp = [float(num) for num in line.split()]
        mpi_sum.append(np.sum(tmp[:][2]))
print(mpi_sum)
mpi_sum = np.array(mpi_sum)

# Plot
fig, ax = plt.subplots()
ax.plot(x_mpi, mpi_sum, marker='*', color='navy', label='Accélération mesurée')
ax.plot(x_mpi, mpi_sum[0]*np.ones(8), linestyle='--', color='#cb3717', label='Accélération idéale')
ax.set_xlabel("Nombre de coeurs")
ax.set_ylabel('Temps (en ms)')
ax.legend()
ax.grid(True)
plt.savefig(f'weak_scalability_init.png')

# On récupère les données omp
omp_med = []
for n in range(5):
    with open(f'../output_files/output_omp{2**n}.txt', 'r') as f:
        for line in f:
            tmp = [float(num) for num in line.split()]
        omp_med.append(np.median(tmp))

omp_med = np.array(omp_med) 
ideal_omp = omp_med[0] / x_omp

# Plot
fig, ax = plt.subplots()
ax.plot(x_omp, omp_med, marker='*', color='navy', label='Accélération mesurée')
ax.plot(x_omp, ideal_omp, linestyle='--', color='#cb3717', label='Accélération idéale')
ax.set_xlabel('Nombre de threads')
ax.set_ylabel('Temps (en ms)')
ax.legend()
ax.grid(True)
plt.savefig(f'strong_scalability.png')