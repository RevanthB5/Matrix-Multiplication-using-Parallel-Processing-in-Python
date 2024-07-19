import numpy
import time
from multiprocessing import Process, Queue, cpu_count
import random

def generate_random_matrix(rows, cols):
    """Generate a matrix with random numbers"""
    return numpy.random.randint(100, size=(rows, cols))

def compare_to_our_approach(matrix1, matrix2):
    """Compare the result and execution time to multiprocessing approach and naive matrix multiplication"""
    
    # Start the timer for our approach
    start_time_our = time.perf_counter()
    our_result = parallel_mul(matrix1, matrix2)
    end_time_our = time.perf_counter()
    
    # Start the timer for naive matrix multiplication
    start_time_naive = time.perf_counter()
    naive_result = naive_matrix_mult(matrix1, matrix2)
    end_time_naive = time.perf_counter()
    
    # Calculate and print the time taken by each process
    time_our = end_time_our - start_time_our
    time_naive = end_time_naive - start_time_naive
    print(f"\nTime taken by our approach: {time_our} seconds")
    print(f"Time taken by naive matrix multiplication: {time_naive} seconds")
    
    # Calculate the speedup factor
    speedup = time_naive / time_our
    
    print(f"\nOur approach is {speedup:.2f} times faster than naive matrix multiplication.\n")
  
# worker function
def worker(matrix1Part, matrix2, result_queue):
    result_queue.put(numpy.matmul(matrix1Part, matrix2))

def input_matrix():
    """Take a matrix as input from the user."""
    
    # Get the number of rows and columns
    while True:
        try:
            rows = int(input("Enter the number of rows: "))
            cols = int(input("Enter the number of columns: "))
            if rows <= 0 or cols <= 0:
                print("Number of rows and columns should be greater than 0. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input! Please enter a positive integer.")
    
    # Initialize an empty matrix
    matrix = numpy.zeros((rows, cols))
    
    # Get the matrix elements
    for i in range(rows):
        for j in range(cols):
            while True:
                try:
                    matrix[i, j] = float(input(f"Enter element ({i+1}, {j+1}): "))
                    break
                except ValueError:
                    print("Invalid input, Please enter a number.")
    
    return matrix

#parallel muliplication using parallel processing 
def parallel_mul(matrix1,matrix2):
  
    # Splitting the first matrix
    num_processes = cpu_count()
    split_matrix_1 = numpy.array_split(matrix1,num_processes)

    # Queue to store the results
    resultQueue = Queue()

    # Start the processes
    processes = []
    for i in split_matrix_1:
        p = Process(target=worker,args=(i,matrix2,resultQueue))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for j in processes:
        j.join()

    # Combine the results
    result_parts = []
    while not resultQueue.empty():
        result_parts.append(resultQueue.get())

    finalResult = numpy.concatenate(result_parts)
    return finalResult

#naive matrix multiplication approach
def naive_matrix_mult(matrix1, matrix2):
    """Perform naive matrix multiplication"""
    n1, m1 = matrix1.shape
    n2, m2 = matrix2.shape
    assert m1 == n2, "Matrix dimensions do not match for multiplication"

    result = numpy.zeros((n1, m2))
    for i in range(n1):
        for j in range(m2):
            for k in range(m1):
                result[i][j] += matrix1[i][k] * matrix2[k][j]
    return result



# Input two matrices n*n matrices'
option = input("Enter 0 to generate random matrices or 1 to input matrices: ")
option = int(option)  # Convert the input to an integer.

if option == 0:
    n = int(input("Enter the size of the matrices: "))
    matrix1 = generate_random_matrix(n, n)
    matrix2 = generate_random_matrix(n, n)

elif option == 1:
    matrix1 = input_matrix()
    matrix2 = input_matrix()

else:
    print("Enter a valid choice (0 or 1)!")

# Print the inputs and the product of the two matrices
print("\nInput matrix 1: \n",matrix1)
print("\nInput matrix 2: \n",matrix2)

print("\nfinal result after multiplying the above matrices:\n",parallel_mul(matrix1,matrix2))

# Compare the result to naive matrix multiplication approach
compare_to_our_approach(matrix1, matrix2)