import numpy as np

N = 4
x = np.arange(N**2, dtype=np.float64) * 2 + 1
y = (np.arange(N**2, dtype=np.float64) + 1) * 2

x = np.reshape(x, (N, N))
y = np.reshape(y, (N, N))
z = x + 1j * y
z = z

print("Input: ")
print(z)
print("FFT: ")
a = np.fft.fft2(z)
print(a)

print("IFFT: ")
print(np.fft.ifft2(a))
