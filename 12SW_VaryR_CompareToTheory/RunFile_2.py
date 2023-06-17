from Params import *

import subprocess
import threading

import time

import os,shutil

starttime = time.time()

if not os.path.isdir(SaveDirName):
    os.mkdir(SaveDirName)
    print("Created Directory")

shutil.copy("Params.py",SaveDirName)


# Maximum number of concurrent threads
max_threads = 40

# Lock to synchronize thread access
lock = threading.Lock()

# Function to execute a command
def execute_command(r):
    command = ['nice', '-n', '18', 'python', 'Script.py', '-r', str(r)]
    with lock:
        print("Executing:", command)
    subprocess.Popen(command).wait()

# List to store the running threads
threads = []

# Iterate over the CList
for r in rlist:
    # Wait until a thread is available
    while len(threads) >= max_threads:
        threads = [thread for thread in threads if thread.is_alive()]

    # Create a new thread and start executing the command
    thread = threading.Thread(target=execute_command, args=(r,))
    thread.start()
    threads.append(thread)

# Wait for all threads to finish
for thread in threads:
    thread.join()


endtime = time.time()


print("Time taken:",endtime-starttime)
