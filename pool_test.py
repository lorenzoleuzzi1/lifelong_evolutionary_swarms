import multiprocessing
from concurrent.futures import ProcessPoolExecutor

def work_function(x):
    return x**2

def main():
    data = range(100000)

    # Using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as executor:
        results_executor = list(executor.map(work_function, data))
        print(f"Results from ProcessPoolExecutor: {results_executor[:5]}")

    # Using multiprocessing.Pool
    with multiprocessing.Pool(processes=4) as pool:
        results_pool = pool.map(work_function, data)
        print(f"Results from multiprocessing.Pool: {results_pool[:5]}")

    

if __name__ == '__main__':
    main()
