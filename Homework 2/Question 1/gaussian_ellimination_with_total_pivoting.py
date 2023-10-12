import numpy as np

def Gaussian_Elimination_With_Total_Pivoting(A, b):
    A = A.astype('float64')
    b = b.astype('float64')
    n = len(b)
    rootIndices = np.arange(n, dtype = int)
    for k in range(n - 1):
        p, q =  np.unravel_index(abs(A[k:, k:]).argmax(), A[k:, k:].shape)
        p = p + k
        q = q + k

        # Index interchanging if needed
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[k], b[p] = b[p], b[k]
            
        if q != k:
            A[:, [k, q]] = A[:, [q, k]]
            rootIndices[k], rootIndices[q] = rootIndices[q], rootIndices[k]
            
        # Elimination
        for j in range(k + 1, n):
            alpha = A[j, k] / A[k, k]
            A[j, k:] = A[j, k:]- alpha * A[k, k:]
            b[j] = b[j] - alpha * b[k]

    # Back substitution phase
    for k in range(n - 1, -1, -1):
        b[k] = (b[k] - np.dot(A[k, k + 1:n], b[k + 1:n])) / A[k, k]
    
    # Bringing order of elements of x to the proper order
    x = np.zeros(n, dtype = 'float64')
    for k in range(n):
        x[k] = b[np.where(rootIndices == k)]
    
    return x


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
        
        computed_coefficients = Gaussian_Elimination_With_Total_Pivoting(A.copy(), b.copy())

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
