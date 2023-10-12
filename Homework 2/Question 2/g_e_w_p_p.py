import numpy as np

def Gaussian_Elimination_With_Partial_Pivoting(A, b):
    A = A.astype('float64')
    b = b.astype('float64')
    n = len(b)
    for k in range(n - 1):
        p = np.argmax(np.abs(A[k: , k])) + k

        # Index interchanging if needed
        if p != k:
            A[[k, p]] = A[[p, k]]
            b[k], b[p] = b[p], b[k]

        # Elimination
        for j in range(k + 1, n):
            if A [j , k] != 0.0:
                alpha = A[j, k] / A[k, k]
                A[j, k:] = A[j, k:]- alpha * A[k, k:]
                b[j] = b[j] - alpha * b[k]

    # Back substitution phase
    for k in range(n - 1, -1, -1):
        b[k] = (b[k] - np.dot(A[k, k + 1:n], b[k + 1:n])) / A[k, k]

    return b
    

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
    list_epsilon = [1e2, 1e-1, 1e-9]
    for epsilon in list_epsilon:
        A = np.array([[epsilon, 1], [1, 1]]).astype('float64')
        b = np.array([1, 2]).astype('float64')
        
        computed_x = Gaussian_Elimination_With_Partial_Pivoting(A.copy(), b.copy())
     
        true_x = np.array([-1 / (epsilon - 1), (2 * epsilon - 1) / (epsilon - 1)]).astype('float64')
 
        error_L_1_a, error_L_1_r, error_L_2_a, error_L_2_r, error_L_infinity_a, error_L_infinity_r = coefficients_Error(true_x, computed_x)
        
        print(f"epsilon = {epsilon}\n")
        print(f"Computed x_1 = {computed_x[0]:.8f}", f"  Computed x_2 = {computed_x[1]:.8f}", "\n")
        print(f"\"True\" x_1 = {true_x[0]:.8f}", f"  \"True\" x_2 = {true_x[1]:.8f}",  "\n")
        print(f"L_1 Error: Absolute = {error_L_1_a}   Relative = {error_L_1_r}\n")
        print(f"L_2 Error: Absolute = {error_L_2_a}   Relative = {error_L_2_r}\n")
        print(f"L_infinity Error: Absolute = {error_L_infinity_a}   Relative = {error_L_infinity_r}\n-----------")
     
if __name__ == "__main__":
    main()