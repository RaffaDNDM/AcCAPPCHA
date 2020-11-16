'''
Return directory path with '/' at the end
'''
def uniform_dir_path(directory):
    if directory.endswith('/') or directory.endswith('\\'):
        return directory
    else:
        return directory+'/'