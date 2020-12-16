import time

def log_time(fun):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = fun(*args, **kwargs)
        t1 = time.time()

        print(f'Time {fun.__name__}: {t1 - t0} s') 
        return result
    return wrapper