import requests
import shutil
import tempfile
import threading
import os
import glob
import tqdm
import multiprocessing
from multiprocessing import Pool
import time
import concurrent.futures

# CHANGEME: Set the directory to the dataset
dir = "/Users/malteebnerlightly/Documents/datasets/clothing-dataset-small"


# Prints the lightly-serve command
lightly_serve_command = f"lightly-serve input_mount={dir} lightly_mount={dir}"
print("lightly-serve command:")
print(lightly_serve_command)


# get all image files in the directory (recursively)
image_files = list(glob.glob(os.path.join(dir, "**", "*.jpg"), recursive=True))
print(f"Found {len(image_files)} image files in the directory")

if True:
    # Download the image to a tempdir using a ThreadPoolExecutor.
    # The connection errors are counted and printed at the end.
    def stresstest_threaded_semaphore(actual_copy: bool = True):
        tmp_dir = tempfile.mkdtemp()
        print(f"Downloading {len(image_files)} images with function {stresstest_threaded_semaphore.__name__}(actual_copy={actual_copy}).")

        # Counter for connection errors
        connection_errors = 0
        error_lock = threading.Lock()

        def download_image_threaded(url, path, actual_copy: bool):
            nonlocal connection_errors
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                time.sleep(0.1)
                with open(path, "wb") as out_file:
                    if actual_copy:
                        shutil.copyfileobj(response.raw, out_file)
            except requests.exceptions.ConnectionError:
                with error_lock:
                    connection_errors += 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for image_file in image_files:
                relpath = os.path.relpath(image_file, dir)
                url = f"http://localhost:3456/{relpath}"
                path = os.path.join(tmp_dir, relpath)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                future = executor.submit(download_image_threaded, url, path, actual_copy)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                future.result()

        print(f"Downloaded {len(image_files)} images to {tmp_dir}")
        print(f"Number of connection errors: {connection_errors}")

if True:
    # Download the image to a tempdir without using any thread limit.
    # The connection errors are counted and printed at the end.
    def stresstest_threaded_unbounded(actual_copy: bool = True):
        tmp_dir = tempfile.mkdtemp()
        print(f"Downloading {len(image_files)} images with function {stresstest_threaded_unbounded.__name__}(actual_copy={actual_copy}).")

        # Counter for connection errors
        connection_errors = 0
        error_lock = threading.Lock()

        def download_image_threaded(url, path, actual_copy: bool):
            nonlocal connection_errors
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                time.sleep(0.1)
                with open(path, "wb") as out_file:
                    if actual_copy:
                        shutil.copyfileobj(response.raw, out_file)
            except requests.exceptions.ConnectionError:
                with error_lock:
                    connection_errors += 1

        threads = []
        for image_file in tqdm.tqdm(image_files):
            relpath = os.path.relpath(image_file, dir)
            url = f"http://localhost:3456/{relpath}"
            path = os.path.join(tmp_dir, relpath)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            thread = threading.Thread(target=download_image_threaded, args=(url, path, actual_copy))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        print(f"Downloaded {len(image_files)} images to {tmp_dir}")
        print(f"Number of connection errors: {connection_errors}")

if True:


    def worker(image_files_chunk, tmp_dir, actual_copy, error_counter):

        def download_image_threaded(url, path, actual_copy: bool):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                time.sleep(0.1)
                with open(path, "wb") as out_file:
                    if actual_copy:
                        shutil.copyfileobj(response.raw, out_file)
            except requests.exceptions.ConnectionError:
                with error_counter.get_lock():
                    error_counter.value += 1
            
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for image_file in image_files_chunk:
                relpath = os.path.relpath(image_file, dir)
                url = f"http://localhost:3456/{relpath}"
                path = os.path.join(tmp_dir, relpath)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                future = executor.submit(download_image_threaded, url, path, actual_copy)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                future.result()

    def stresstest_threaded_unbounded_multiprocess(actual_copy: bool = True):
        tmp_dir = tempfile.mkdtemp()
        print(f"Downloading {len(image_files)} images with function {stresstest_threaded_unbounded_multiprocess.__name__}(actual_copy={actual_copy}).")

        # Create a multiprocessing value for connection errors
        connection_errors = multiprocessing.Value('i', 0)

        # Split image files into chunks for each process
        num_processes = 2
        chunk_size = len(image_files) // num_processes
        image_files_chunks = [image_files[i:i + chunk_size] for i in range(0, len(image_files), chunk_size)]

        start = time.time()

        processes = []
        for chunk in image_files_chunks:
            p = multiprocessing.Process(target=worker, args=(chunk, tmp_dir, actual_copy, connection_errors))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        end: float = time.time()

        print(f"Downloaded {len(image_files)} images in {end - start:.3f}s")
        print(f"Number of connection errors: {connection_errors.value}")



# Call the functions
if __name__ == '__main__':
    pass
    #freeze_support()
    #stresstest_threaded_semaphore(actual_copy=True)
    #stresstest_threaded_unbounded(actual_copy=True)
    #
    #stresstest_threaded_semaphore(actual_copy=False)
    #stresstest_threaded_unbounded(actual_copy=False)

    multiprocessing_stresstest=True
    if multiprocessing_stresstest:
        stresstest_threaded_unbounded_multiprocess(actual_copy=True)
        #stresstest_threaded_unbounded_multiprocess(actual_copy=False)
