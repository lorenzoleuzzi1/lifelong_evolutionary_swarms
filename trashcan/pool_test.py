import numpy as np
import time
# from multiprocessing import Pool
import os

# pool = Pool(processes=os.cpu_count())

# def f(x):
#     time.sleep(0.001)
#     return 1.5 * 2 - x

# def my_func_par(large_list):
#     pool.map(f, large_list)

# def my_func_seq(large_list):
#     [f(i) for i in large_list]

# my_list = np.arange(1, 10000)

# s = time.time()
# my_func_par(my_list)
# print('Parallel time: ' + str(time.time() - s))

# s = time.time()
# my_func_seq(my_list)
# print('Sequential time: ' + str(time.time() - s))

from multiprocessing import Pool

def square_number(number):
    time.sleep(0.001)
    return number * number

if __name__ == '__main__':
    print('Number of CPUs: ' + str(os.cpu_count()))
    numbers = np.arange(1, 10000)
    # Create a Pool with 4 worker processes
    start_p = time.time()
    pool = Pool(processes=os.cpu_count()) 
    # Map the square_number function to the numbers list
    results = pool.map(square_number, numbers)
    print('Parallel time: ' + str(time.time() - start_p))

    start_s = time.time()
    # Sequential
    results = [square_number(number) for number in numbers]
    print('Sequential time: ' + str(time.time() - start_s))