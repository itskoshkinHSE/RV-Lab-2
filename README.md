```bash
mpicc -o rv-lab-2 main-columns.c
mpiexec -np 1 ./rv-lab-2 2500 2500
mpiexec -np 4 ./rv-lab-2 2500 2500
mpiexec -oversubscribe -np 9 ./rv-lab-2 2500 2500
mpiexec -oversubscribe -np 16 ./rv-lab-2 2500 2500
```