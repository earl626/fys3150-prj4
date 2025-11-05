#post-processing and validation
import numpy as np
import matplotlib.pyplot as plt
import subprocess

#ex: sweep temperature and compare <E>/N and <|M|>/N
L = 2
temps = np.linspace(1.0, 2.5, 16)
cycles = 1000000
ordered = 1

E_vals, M_vals = [], []

for T in temps:
    #run prob4.cpp and get output
    res = subprocess.run(["./ising", str(L), str(T), str(cycles), str(ordered)],
                         capture_output=True, text=True)
    for line in res.stdout.splitlines():
        if "<E>/N" in line:  E_vals.append(float(line.split("=")[1]))
        if "<|M|>/N" in line: M_vals.append(float(line.split("=")[1]))

plt.figure(figsize=(7,4))
plt.plot(temps, E_vals, "o-", label=r"$\langle \epsilon \rangle$")
plt.plot(temps, M_vals, "s-", label=r"$\langle |m| \rangle$")
plt.xlabel("T (J/kB)")
plt.ylabel("Per spin averages")
plt.legend()
plt.tight_layout()
plt.show()
