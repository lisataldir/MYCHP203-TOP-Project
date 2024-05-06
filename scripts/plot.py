import numpy as np
import matplotlib.pyplot as plt

x_omp = 2**np.arange(5)
x_mpi = np.arange(1,9)

# On récupère les données mpi v0 (elles s'appellent v2 mais c'est les v0)
mpi_sum = []
for n in range(1,9):
    with open(f'../output_files/output_mpi{n}_v2.txt', 'r') as f:
        for line in f:
            tmp = [float(num) for num in line.split()]
        mpi_sum.append(np.sum(tmp[:][2]))

mpi_sum = np.array(mpi_sum)

# Plot
fig, ax = plt.subplots()
ax.plot(x_mpi, mpi_sum, marker='*', color='navy', label='Accélération mesurée')
ax.plot(x_mpi, mpi_sum[0]*np.ones(8), linestyle='--', color='#cb3717', label='Accélération idéale')
ax.set_xlabel("Nombre de processus MPI")
ax.set_ylabel('Temps (en ms)')
ax.legend()
ax.grid(True)
plt.savefig(f'weak_scalability_init.png')

# On récupère les données mpi après retrait de la barriere
mpi_sum3 = []
for n in range(5):
    with open(f'../output_files/output_mpi{2**n}_v3.txt', 'r') as f:
        for line in f:
            tmp = [float(num) for num in line.split()]
        mpi_sum3.append(np.sum(tmp[:][2]))

mpi_sum3 = np.array(mpi_sum3)

# Plot
fig, ax = plt.subplots()
ax.plot(x_omp, mpi_sum3, marker='*', color='navy', label='Accélération mesurée')
ax.plot(x_omp, mpi_sum3[0]*np.ones(5), linestyle='--', color='#cb3717', label='Accélération idéale')
ax.set_xlabel("Nombre de processus MPI")
ax.set_ylabel('Temps (en ms)')
ax.legend()
ax.grid(True)
plt.savefig(f'weak_scalability_final.png')


# On récupère les données omp v0
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
ax.set_xlabel('Nombre de threads OpenMP')
ax.set_ylabel('Temps (en ms)')
ax.set_title('Strong scalability après parallélisation de solve_jacobi()')
ax.legend()
ax.grid(True)
plt.savefig(f'strong_scalability_v0_0.png')

fig, ax = plt.subplots()
ax.plot(x_omp, omp_med[0]/omp_med, marker='*', color='navy', label='Accélération mesurée')
ax.plot(x_omp, x_omp, linestyle='--', color='#cb3717', label='Accélération idéale')
ax.set_xlabel('Nombre de threads OpenMP')
ax.set_ylabel('Accélération')
ax.set_title('Strong scalability après parallélisation de solve_jacobi()')
ax.legend()
ax.grid(True)
plt.savefig(f'strong_scalability_v0_1.png')



# On récupère les données omp v1 après parallélisation de mesh_copy_core
omp_med2 = []
for n in range(5):
    with open(f'../output_files/output_omp{2**n}_v1.txt', 'r') as f:
        for line in f:
            tmp = [float(num) for num in line.split()]
        omp_med2.append(np.median(tmp[:][2]))

omp_med2 = np.array(omp_med) 

# Plot
fig, ax = plt.subplots()
ax.plot(x_omp, omp_med2, marker='*', color='navy', label='Accélération mesurée')
ax.plot(x_omp, ideal_omp, linestyle='--', color='#cb3717', label='Accélération idéale')
ax.set_xlabel('Nombre de threads OpenMP')
ax.set_ylabel('Temps (en ms)')
ax.set_title('Strong scalability après parallélisation de mesh_copy_core()')
ax.legend()
ax.grid(True)
plt.savefig(f'strong_scalability_v1_0.png')

fig, ax = plt.subplots()
ax.plot(x_omp, omp_med2[0]/omp_med2, marker='*', color='navy', label='Accélération mesurée')
ax.plot(x_omp, x_omp, linestyle='--', color='#cb3717', label='Accélération idéale')
ax.set_xlabel('Nombre de threads OpenMP')
ax.set_ylabel('Accélération')
ax.set_title('Strong scalability après parallélisation de mesh_copy_core()')
ax.legend()
ax.grid(True)
plt.savefig(f'strong_scalability_v1_1.png')