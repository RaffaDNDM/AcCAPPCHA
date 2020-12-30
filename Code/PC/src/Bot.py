import subprocess
import sys
import colorama
import msvcrt

process = subprocess.Popen('python3 AcCAPPCHA.py -t -plot', shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

print(process.communicate(b'foo\r\n')[0])
#process.stdin.write(b'\r')
print(process.communicate(b'\r')[0])
print(process.communicate(b'i')[0])
print(process.communicate(b'a')[0])
print(process.communicate(b'o')[0])
print(process.communicate(b'\r\n')[0])