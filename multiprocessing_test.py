
import concurrent.futures

def power_by_two(x):
    return x*x

def list_power_by_two(list_of_params):
    for i in range(len(list_of_params)):
        print(power_by_two(list_of_params[i]))

with concurrent.futures.ProcessPoolExecutor() as executor:
    executor.map(function_name, power_by_two)