import os

files = [x for x in os.listdir('.')]
letters = [chr(ord('a')+x) for x in range(0,25)]

subfolder = 'Complete_acquisition/'
os.mkdir(subfolder)

for x in letters:
    if x not in files:
        os.mkdir(subfolder+x)
