import subprocess
import matplotlib.pyplot as plt

def run_sequential():
    result = subprocess.run(['python', 'Lowe_algorithm.py'], stdout=subprocess.PIPE)
    execution_time = float(result.stdout.decode('utf-8').split('\n')[-2])
    return execution_time

def run_parallel(num_processes):
    result = subprocess.run(['python', 'parallel_Input_Lowe_algorithm.py', str(num_processes)], stdout=subprocess.PIPE)
    execution_time = float(result.stdout.decode('utf-8').split('\n')[-2])
    return execution_time

speedup = []
sequential_time = run_sequential()
for i in range(1, 17):
    parallel_time = run_parallel(i)
    speedup.append(round(sequential_time / parallel_time, 4))
    print('Speedup with {} processes: {:.4f}'.format(i, speedup[-1]))

plt.plot(range(1, 17), speedup, label='Speedup', marker='o')
plt.plot(range(1,17), range(1,17), label="Ideal", marker='o', color='coral')
plt.xlabel('Number of processes')
plt.ylabel('Speedup')
plt.title('Speedup vs Number of processes')
plt.legend()
plt.show()