# OpenRover
# Keys.py

# Handles accessing the local machine keyboard.

import sys
import time

# Variables exported
up_key_pressed = False
down_key_pressed = False
left_key_pressed = False
right_key_pressed = False
esc_key_pressed = False

def keyboard_hook(key):
    global up_key_pressed, down_key_pressed, left_key_pressed, right_key_pressed, esc_key_pressed
    pressed = True
    if key.event_type == "up":
        pressed = False
    if key.name == "up":
        up_key_pressed = pressed
    if key.name == "down":
        down_key_pressed = pressed
    if key.name == "left":
        left_key_pressed = pressed
    if key.name == "right":
        right_key_pressed = pressed
    if( key.name == "esc" ):
        esc_key_pressed = pressed

def on_press(key):
    global up_key_pressed, down_key_pressed, left_key_pressed, right_key_pressed, esc_key_pressed
    from pynput import keyboard
    if( key == keyboard.Key.up ): up_key_pressed = True
    if( key == keyboard.Key.down ): down_key_pressed = True
    if( key == keyboard.Key.left ): left_key_pressed = True
    if( key == keyboard.Key.right ): right_key_pressed = True
    if( key == keyboard.Key.esc ): esc_key_pressed = True

def on_release(key):
    global up_key_pressed, down_key_pressed, left_key_pressed, right_key_pressed, esc_key_pressed
    from pynput import keyboard
    if( key == keyboard.Key.up ): up_key_pressed = False
    if( key == keyboard.Key.down ): down_key_pressed = False
    if( key == keyboard.Key.left ): left_key_pressed = False
    if( key == keyboard.Key.right ): right_key_pressed = False
    if( key == keyboard.Key.esc ): esc_key_pressed = False

def keyboard_listener():
    print( "Starting keyboard." )

    # There are two different libraries to access the keyboard.
    # Pynput works well but doesn't start automatically on linux system startup because there's no Xlib available.
    # Keyboard works well but requires sudo. Your choice.
    USE_PYNPUT = False
    if USE_PYNPUT: # Use 'pynput' module
        try:
            from pynput import keyboard
            with keyboard.Listener(
                    on_press=on_press,
                    on_release=on_release) as listener:
                listener.join()
        except:
            print( "Error: Cannot access keyboard. Please install pynput and linux desktop." )
            print( sys.exc_info() )
    else: # Use 'keyboard' module instead
        try:
            import keyboard
            keyboard.hook( keyboard_hook )
        except:
            print( "Error: Cannot access keyboard. Please install with 'sudo pip install keyboard'." )
            print( sys.exc_info() )

# Run
try:
    import _thread
    _thread.start_new_thread( keyboard_listener, () )
except:
    print( "Error: Cannot start keyboard listener. Please install python threads or use Python 3." )
