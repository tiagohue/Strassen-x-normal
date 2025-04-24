import numpy as np
import time

def multiplicar_matrizes(A, B):
    resultado = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                resultado[i][j] += A[i][k] * B[k][j]

    return resultado

def strassen(A, B):
    n = A.shape[0]

    if n == 1:
        return A * B

    if n % 2 != 0:
        A = np.pad(A, ((0,1), (0,1)), mode='constant')
        B = np.pad(B, ((0,1), (0,1)), mode='constant')
        n += 1

    mid = n // 2

    A11 = A[:mid, :mid]
    A12 = A[:mid, mid:]
    A21 = A[mid:, :mid]
    A22 = A[mid:, mid:]

    B11 = B[:mid, :mid]
    B12 = B[:mid, mid:]
    B21 = B[mid:, :mid]
    B22 = B[mid:, mid:]

    M1 = strassen(A11 + A22, B11 + B22)
    M2 = strassen(A21 + A22, B11)
    M3 = strassen(A11, B12 - B22)
    M4 = strassen(A22, B21 - B11)
    M5 = strassen(A11 + A12, B22)
    M6 = strassen(A21 - A11, B11 + B12)
    M7 = strassen(A12 - A22, B21 + B22)

    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    C = np.vstack((np.hstack((C11, C12)), np.hstack((C21, C22))))
    return C

def benchmark(n):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)

    normal_start = time.time()
    print(multiplicar_matrizes(A, B))
    normal_time = time.time() - normal_start

    strassen_start = time.time()
    print(strassen(A, B))
    strassen_time = time.time() - strassen_start
    
    print(f"Benchmark para matrizes {n}x{n}...")
    print(f"Tempo da multiplicacao padrao: {normal_time:.4f} segundos")
    print(f"Tempo da multiplicacao com Strassen: {strassen_time:.4f} segundos")

    diferenca = normal_time - strassen_time
    print(f"Tempo normal - Tempo Strassen: {diferenca:.6f}")

if __name__ == "__main__":
    benchmark(128)
