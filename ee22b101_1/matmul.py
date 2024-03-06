def matmul(A, B):
    if len(A[0])==len(B): #checks if the matrices are of the sort Mx"N" and "N"xP
        C = [[0 for p in range(len(B[0]))] for p in range(len(A))] #defining C's size
        for i in range(0, len(A)):
            for j in range(0, len(B[0])):
                for k in range(0, len(A[0])):
                    C[i][j] += A[i][k] * B[k][j] #looping the multiplied elements from the two matrices
        return C
    else:
        raise ValueError("can't be multiplied") #rasining a ValueError so as to satisfy the given testcase under def test_axis_mismatch



