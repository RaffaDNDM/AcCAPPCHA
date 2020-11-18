def swap(m, n):
    '''
    Invert order of args

    Args:
        m (int): first number
        n (int): second number

    Returns:
        n (int): second number
        m (int): first number
    '''
    return n, m


def uniform_dir_path(directory):
    '''
    Return directory path with '/' at the end

    Args:
        directory (str): directory path that you want to uniform

    Returns:
        directory (str): modified directory path that ends with '/'
    '''
    if directory.endswith('/') or directory.endswith('\\'):
        return directory
    else:
        return directory+'/'


def key_definition(key):
    '''
    Evaluate which key has been pressed

    Args:
        key (key): pynput key

    Returns:
        key_string (str): string that correspond to the pressed key
    '''
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
        #Fn tast disable in Dell PC
        #elif key == key.f1:
        #    key_string= 'F1'
        #elif key == key.f2:
        #    key_string= 'F2'
        #elif key == key.f3:
        #    key_string= 'F3'
        #elif key == key.f4:
        #    key_string= 'F4'
        #elif key == key.f5:
        #    key_string= 'F5'
        #elif key == key.f6:
        #    key_string= 'F6'
        #elif key == key.f7:
        #    key_string= 'F7'
        #elif key == key.f8:
        #    key_string= 'F8'
        #elif key == key.f9:
        #    key_string= 'F9'
        #elif key == key.f10:
        #    key_string= 'F10'
        #elif key == key.f11:
        #    key_string= 'F11'
        #elif key == key.f12:
        #    key_string= 'F12'
        #elif key == key.f13:
        #    key_string= 'F13'
        #elif key == key.f14:
        #    key_string= 'F14'
        #elif key == key.f15:
        #    key_string= 'F15'
        #elif key == key.f16:
        #    key_string= 'F16'
        #elif key == key.f17:
        #    key_string= 'F17'
        #elif key == key.f18:
        #    key_string= 'F18'
        #elif key == key.f19:
        #    key_string= 'F19'
        #elif key == key.f20:
        #    key_string= 'F20'
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