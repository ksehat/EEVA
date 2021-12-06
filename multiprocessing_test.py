from my_geneticalgorithm import MyGeneticAlgorithm as mga
import concurrent.futures
import math
import multiprocessing

PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419]

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True
def main():
    pool = multiprocessing.Pool()
    result = pool.map(is_prime, PRIMES)
    print(result)

if __name__ == '__main__':
    main()

