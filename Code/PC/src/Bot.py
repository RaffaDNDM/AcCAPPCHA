import subprocess
import sys
import colorama
import msvcrt
from time import sleep

def popen_bot(username, password):
    #Subprocess that redirects pipes
    process = subprocess.Popen('python3 AcCAPPCHA.py -t -plot', shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    #Wait until username could be inserted
    sleep(4)
    #Write username and password
    output = process.communicate(username.encode() + b'\r\n'+ \
                              password.encode() + b'\r\n')[0]
    
    print(output.decode())

from pynput.keyboard import Key, Controller

def input_bot(username, password):
    #Object for control of keyboard events
    keyboard = Controller()

    def press_release(char):
        keyboard.press(char)
        keyboard.release(char)

    #Wait that username could be inserted
    sleep(4)

    #username insertion
    for x in username:
        press_release(x)
    
    press_release(Key.enter)

    #Trials for password insertion
    count = 0
    while(count<3):
        sleep(5)
        
        for x in password:
            press_release(x)

        press_release(Key.enter)
        count += 1


def main():
    username = 'RaffaDNDM'
    password = 'hello35'
    
    popen_bot(username, password)
    #input_bot(username, password)

if __name__=='__main__':
    main()