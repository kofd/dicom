import os
from itertools import islice
from concurrent.futures import ThreadPoolExecutor


def intchars(i, length):
    '''
    Properly format an integer for string comparisons. This function
    will buffer the start of the string with 0's until it has the desired length.

    :param i: The integer.
    :param length: The length of the resulting string
    :return: A string of length=length representing the integer.
    '''
    string = str(i)
    return '0' * (length - len(string)) + string


def ibatch(iterable, size, fn=lambda btc: btc):
    '''
    Batch a stream of values. If the final batch is smaller than the batch
    size it will be thrown out.

    :param iterable: An iterable to be batched.
    :param size: The batch size being generated.
    :param fn: A helper function to allow batches to be something other than a list.
    :return: A generator that will yield batches of batch_size=size mapped onto the helper function.
    '''
    stream = iter(iterable)
    while True:
        btc = list(islice(stream, 0, size))
        if len(btc) < size:
            break
        yield fn(btc)


def imap_async(fn, iterable, max_workers=None):
    '''
    Map a stream of values with a function in parallel. This function is intended to prevent
    the system from running away and computing all values in what could be an infinite stream.
    This function will work ahead until it gets max_workers ahead of the currently yeilded
    datum, then wait for more values to be yielded before continuing.

    :param fn: A function to map the values in the stream.
    :param iterable: An iterable to map, could be an infinite stream.
    :param max_workers: The number of workers operating at any given time.
    :return: A generator that will yield mapped values as they are needed.
    '''
    max_workers = max_workers or (os.cpu_count() or 1) * 5
    stream = iter(iterable)
    with ThreadPoolExecutor(max_workers) as pool:
        futures = [pool.submit(fn, datum) for datum in islice(stream, 0, max_workers)]
        for datum in stream:
            result, futures = futures[0].result(), futures[1:]
            futures.append(pool.submit(fn, datum))
            yield result
        for future in futures:
            yield future.result()

