import numpy as np

def LU_Decomposition_With_Total_Pivoting(A):
    A = A.astype('float64')
    n = len(A)
    P = np.eye(n, dtype = 'float64')
    Q = np.eye(n, dtype = 'float64')
    L = np.eye(n, dtype = 'float64')
    U = A.copy()
    for k in range(0, n-1):
        p, q =  np.unravel_index(abs(U[k:, k:]).argmax(), U[k:, k:].shape)
        p = p + k
        q = q + k

        if p != k:
            P_step = np.eye(n, dtype = 'float64')
            U[[p, k], k:n] = U[[k, p], k:n]
            L[[p, k], 0:k] = L[[k, p], 0:k]
            P_step[[p, k]] = P_step[[k, p]]
            P = np.dot(P_step, P)

        if q != k:
            Q_step = np.eye(n, dtype = 'float64')
            U[:, [q, k]] = U[:, [k, q]]
            Q_step[[q, k]] = Q_step[[k, q]]
            Q = np.dot(Q, Q_step)
        
        for j in range(k+1, n):
            L[j, k] =  U[j, k] / U[k, k]
            U[j, k:n] = U[j, k:n] - L[j, k] * U[k, k:n]

    return L, U, P, Q


def LU_TP_Solver(L, U, P, Q, b):
    L = L.astype('float64')
    U = U.astype('float64')
    P = P.astype('float64')
    Q = Q.astype('float64')
    b = b.astype('float64')
    n = len(U)
    # Forward substitution Lz = Pb
    b = np.dot(P, b)
    for k in range(1, n):
        b[k] = b[k] - np.dot(L[k, 0:k], b[0:k])
        
    # Backward substitution Uy = z
    for k in range(n - 1, -1, -1):
       b[k] = (b[k] - np.dot(U[k, k + 1:n], b[k + 1:n])) / U[k, k]
    
    # Finally finding x = Qy   
    b = np.dot(Q, b)
    return b

def generate_A_b(n):
    # Initializing A and b and filling them by 0
    A = np.zeros((n, n), dtype = 'float64')
    b = np.zeros(n, dtype = 'float64')
    
    # Generating A and b
    for i in range(1, n + 1):
        b[i - 1] =  ((1 + i)**n - 1) / i
        for j in range(1, n + 1):
            A[i - 1, j - 1] = (1 + i)**(j - 1)
    
    return A, b
    

def coefficients_Error(true_coefficients, computed_coefficients):
    difference_Vector = abs(true_coefficients - computed_coefficients)
    
    # L_1 Error
    L_1_Absolute_Error = difference_Vector.sum()
    L_1_Relative_Error = L_1_Absolute_Error / abs(true_coefficients).sum()
    
    # L_infinity Error
    L_infinity_Absolute_Error = difference_Vector.max()
    L_infinity_Relative_Error = L_infinity_Absolute_Error / abs(true_coefficients).max()
    
    # L_2 Error
    sum_Of_Squared_Elements = 0 
    for element in difference_Vector:
        sum_Of_Squared_Elements = sum_Of_Squared_Elements + element**2
    L_2_Absolute_Error = np.sqrt(sum_Of_Squared_Elements)
    sum_Of_Squared_Elements = 0
    for element in true_coefficients:
        sum_Of_Squared_Elements = sum_Of_Squared_Elements + element**2
    L_2_Relative_Error = L_2_Absolute_Error / np.sqrt(sum_Of_Squared_Elements)
    
    return L_1_Absolute_Error, L_1_Relative_Error, L_2_Absolute_Error, L_2_Relative_Error, L_infinity_Absolute_Error, L_infinity_Relative_Error
    

def main():
    list_n = [4, 5, 6, 7, 8, 9]
    for n in list_n:
        A, b = generate_A_b(n)
        
        L, U, P, Q = LU_Decomposition_With_Total_Pivoting(A)
        computed_coefficients = LU_TP_Solver(L, U, P, Q, b.copy())

        true_coefficients = np.ones(n, dtype = 'float64')
 
        error_L_1_a, error_L_1_r, error_L_2_a, error_L_2_r, error_L_infinity_a, error_L_infinity_r = coefficients_Error(true_coefficients, computed_coefficients)
        
        print(f"n = {n}\n")
        print(f"Computed Coefficients: {computed_coefficients}\n")
        print(f"True Coefficients: {true_coefficients}\n")
        print(f"L_1 Error: Absolute = {error_L_1_a}   Relative = {error_L_1_r}\n")
        print(f"L_2 Error: Absolute = {error_L_2_a}   Relative = {error_L_2_r}\n")
        print(f"L_infinity Error: Absolute = {error_L_infinity_a}   Relative = {error_L_infinity_r}\n-----------")
     
if __name__ == "__main__":
    main()
