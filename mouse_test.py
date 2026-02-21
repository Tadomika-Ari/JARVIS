from pynput.mouse import Controller, Button
import time

mouse = Controller()

time.sleep(2)
mouse.position = (500, 400)
time.sleep(1)
mouse.click(Button.left)
time.sleep(1)
mouse.scroll(0, -3)
time.sleep(1)
mouse.scroll(0, 5)