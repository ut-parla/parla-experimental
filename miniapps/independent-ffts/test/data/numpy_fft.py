import numpy as np
import pyfftw

N = 4
x = np.arange(N**2, dtype=np.float32) * 2 + 1
y = (np.arange(N**2, dtype=np.float32) + 1) * 2

x = np.reshape(x, (N, N))
y = np.reshape(y, (N, N))
z = x + 1j * y

a = pyfftw.empty_aligned((N, N), dtype=np.complex128, n=16)
a[:] = z
b = pyfftw.interfaces.numpy_fft.fft(a)
print(b)

z = np.asarray(z, dtype=np.complex64)


a = pyfftw.empty_aligned((N, N), dtype=np.complex64, n=8)
a[:] = z
b = pyfftw.interfaces.numpy_fft.fft(a)
print(b)

print("FFT: ")
fftz = np.fft.fft2(z)
# fftz = np.asarray(fftz, dtype=np.complex64)

print(np.allclose(fftz, b))

print(fftz)

fftz.tofile(f"{N}_fft.bin")
