import numpy as np
import pyaudio
import wave
import sys
import scipy
import scipy.fftpack as fftpk
from time import sleep
import threading
from pynput import keyboard
import datetime
from termcolor import cprint
import os
import utility

class AcquireAudio:
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    DATA_FOLDER = 'dat/audio/'
    WAVE_OUTPUT_FILENAME = None
    CHECK_TYPING = True
    FIRST_KEY = True
    KILLED = False

    def __init__(self, audio_dir, times_key):
        if not(os.path.exists(audio_dir) and os.path.isdir(audio_dir)):
            os.mkdir(self.DATA_FOLDER)

        if audio_dir:
            self.DATA_FOLDER = utility.uniform_dir_path(audio_dir)
    
        self.already_acquired()

        input('Print something to start the acquisition of audio')
        sleep(2)
        self.mutex = threading.Lock()
        self.count = 0
        self.TIMES_KEY_PRESSED = times_key


    def already_acquired(self):    
        #Initialize LETTERS with number of files already in the subfolder ('a', ...)
        subfolders = os.listdir(self.DATA_FOLDER)
        subfolders.sort()

        count=0
        self.LETTERS = {}
        special_chars = {}
        cprint('\nNum of already acquired audio samples for letters', 'blue')
        cprint('_________________________________________________', 'blue')

        for subfolder in subfolders:
            if os.path.isdir(self.DATA_FOLDER+subfolder):
                num_already_acquired = len([x for x in os.listdir(self.DATA_FOLDER+subfolder) if x.endswith('.wav')])
                self.LETTERS[subfolder] = num_already_acquired
                if len(subfolder)==1:
                    cprint(f'{subfolder}', 'green', end=' ', attrs=['bold',])
                    cprint('---->', 'yellow', end=' ')
                    delimiter = '   '

                    if count==3:
                        delimiter = '\n'
                        count = 0
                    else:
                        count = count + 1
                    
                    print('{:2d}'.format(num_already_acquired), end=delimiter)

                else:
                    special_chars[subfolder] = num_already_acquired

        if count==0:
            print('', end='\r')        
        else:
            print('', end='\n')
            count=0

        for subfolder in special_chars:
            cprint(f'{subfolder}', 'green', end=' ', attrs=['bold',])
            cprint('---->', 'yellow', end=' ')
            print('{:2d}'.format(self.LETTERS[subfolder]))

        cprint('_________________________________________________', 'blue')


    def audio_logging(self):
        p = pyaudio.PyAudio()

        stream = p.open(format=self.FORMAT,
                    channels=self.CHANNELS,
                    rate=self.RATE,
                    input=True,
                    frames_per_buffer=self.CHUNK)
        
        cprint('\n*** recording ***', 'green', attrs=['bold'])

        frames = []

        while self.CHECK_TYPING:
            if self.KILLED:
                exit(0)

            data = stream.read(self.CHUNK)
            frames.append(data)

        '''
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
        '''

        stream.stop_stream()
        stream.close()
        p.terminate()

        cprint('\n*** End recording ***', 'green', attrs=['bold'])

        while not self.WAVE_OUTPUT_FILENAME:
            sleep(1)

        wf = wave.open(self.DATA_FOLDER+self.WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(p.get_sample_size(self.FORMAT))
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        self.mutex.acquire()
        try:
            self.WAVE_OUTPUT_FILENAME = None
            self.CHECK_TYPING = True
            self.count = 0
            self.FIRST_KEY = True
        finally:
            self.mutex.release()


    def press_key(self, key):
        key_string = self.key_definition(key)

        if self.count < self.TIMES_KEY_PRESSED:
            self.count = self.count + 1
            print('\r'+str(self.count), end='')

            sleep(1)
            self.mutex.acquire()
            try:
                if key_string in self.LETTERS.keys():
                    self.WAVE_OUTPUT_FILENAME = key_string+'/'+str(self.LETTERS[key_string])+'.wav'
                    self.LETTERS[key_string] = self.LETTERS[key_string] + 1
                else:
                    self.LETTERS[key_string] = 0
                    self.WAVE_OUTPUT_FILENAME = key_string+'/'+str(self.LETTERS[key_string])+'.wav'
                    os.mkdir(self.DATA_FOLDER+key_string)

            finally:
                self.mutex.release()

            if self.FIRST_KEY:
                self.mutex.acquire()
                try:
                    self.FIRST_KEY = False
                finally:
                    self.mutex.release()

            if self.count == self.TIMES_KEY_PRESSED:
                self.mutex.acquire()
                try:
                    self.CHECK_TYPING = False
                    exit(0)
                finally:
                    self.mutex.release()


    def key_definition(self, key):
        #Obtain string of key inserted
        try:
            key_string = str(key.char)
        except AttributeError:
            #Special key pressed
            if key == key.alt:
                key_string= 'ALT'
            elif key == key.alt_gr:
                key_string= 'ALT_GR'
            elif key == key.backspace:
                key_string= 'BACKSPACE'
            elif key == key.caps_lock:
                key_string= 'CAPS_LOCK'
            elif key == key.ctrl or key == key.ctrl_l or key == key.ctrl_r:
                key_string= 'CTRL'
            #elif key == key.cmd or key.cmd_r or key.cmd_l:
            #    key_string= 'CMD'
            elif key == key.delete:
                key_string= 'DELETE'
            elif key == key.down:
                key_string= 'DOWN'
            elif key == key.end:
                key_string= 'END'
            elif key == key.esc:
                key_string= 'ESC'
            elif key == key.enter:
                key_string= 'ENTER'
            elif key == key.home:
                key_string= 'HOME'
            elif key == key.insert:
                key_string= 'INSERT'
            elif key == key.left:
                key_string= 'LEFT'
            elif key == key.menu:
                key_string= 'MENU'
            elif key == key.num_lock:
                key_string= 'NUM_LOCK'
            elif key == key.page_down:
                key_string= 'PAGE_DOWN'
            elif key == key.page_up:
                key_string= 'PAGE_UP'
            elif key == key.pause:
                key_string= 'PAUSE'
            elif key == key.print_screen:
                key_string= 'PRINT_SCREEN'
            elif key == key.right:
                key_string= 'RIGHT'
            elif key == key.scroll_lock:
                key_string= 'SCROLL_LOCK'
            elif key == key.space:
                key_string = 'SPACE'
            elif key == key.tab:
                key_string= 'TAB'
            elif key == key.up:
                key_string= 'UP'
            elif key == key.shift or key.shift_r or key.shift_l:
                key_string= 'SHIFT'
            else:
                key_string = str(key)

            '''
            #Fn tast disable in Dell PC
            elif key == key.f1:
                key_string= 'F1'
            elif key == key.f2:
                key_string= 'F2'
            elif key == key.f3:
                key_string= 'F3'
            elif key == key.f4:
                key_string= 'F4'
            elif key == key.f5:
                key_string= 'F5'
            elif key == key.f6:
                key_string= 'F6'
            elif key == key.f7:
                key_string= 'F7'
            elif key == key.f8:
                key_string= 'F8'
            elif key == key.f9:
                key_string= 'F9'
            elif key == key.f10:
                key_string= 'F10'
            elif key == key.f11:
                key_string= 'F11'
            elif key == key.f12:
                key_string= 'F12'
            elif key == key.f13:
                key_string= 'F13'
            elif key == key.f14:
                key_string= 'F14'
            elif key == key.f15:
                key_string= 'F15'
            elif key == key.f16:
                key_string= 'F16'
            elif key == key.f17:
                key_string= 'F17'
            elif key == key.f18:
                key_string= 'F18'
            elif key == key.f19:
                key_string= 'F19'
            elif key == key.f20:
                key_string= 'F20' 
            '''

        if key_string=='.':
            key_string ='POINT'
        elif key_string=='/':
            key_string ='SLASH'
        elif key_string=='\\':
            key_string ='BACKSLASH'
        elif key_string=='*':
            key_string ='STAR'
        elif key_string=='+':
            key_string ='PLUS'
        elif key_string=='-':
            key_string ='MINUS'
        elif key_string==',':
            key_string ='COMMA'
        elif key_string=="'":
            key_string ='APOSTROPHE'

        return key_string
        
            
    def record(self):
        try:
            while True:
                cprint(f'\nType first letter {self.TIMES_KEY_PRESSED} times', 'green', attrs=['bold'])

                audiologger = threading.Thread(target=self.audio_logging)
                audiologger.start()
                with keyboard.Listener(on_press=self.press_key) as listener:
                    #Manage keyboard input
                    listener.join()

                audiologger.join()

        except KeyboardInterrupt:
            self.KILLED = True
            cprint('\nClosing the program', 'red', attrs=['bold'], end='\n\n')