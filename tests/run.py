import sys
import os
import subprocess
#for files in this dirictory that have a name _test
# python3 run as python3 {filename}
# C# ?
# C++ g++
# run and capture the exit code
code = 0
print("Starting Testing Suite:\n")
i = 0
for c,n in enumerate(os.listdir("./tests/")):
    if "_test" in n:
        print("#",i," starting: ", n )
        i +=1
        result = subprocess.run(["python3", f"./tests/{n}"])
        code = result.returncode
        if (code != 0):
            print("fail at ", i )
            code = i+1
            os.environ["status"] = str(code)
            break
else:
    os.environ["status"] = str(code)
print("status = ",os.environ["status"])
# print("runner = ",os.environ["WEBHOOK"])
sys.exit(code)
