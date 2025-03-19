import RPi.GPIO as GPIO
import time

# Define GPIO pins
IN1 = 23
IN2 = 24
ENA = 5

# Setup GPIO mode
GPIO.setmode(GPIO.BCM)
GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(ENA, GPIO.OUT)

# Initialize PWM on ENA pin
pwm = GPIO.PWM(ENA, 1000)  # 1000 Hz frequency
pwm.start(0)

def rotate_left(speed=100):
    """Rotate motor left at the given speed (0-100)."""
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    pwm.ChangeDutyCycle(speed)

def rotate_right(speed=100):
    """Rotate motor right at the given speed (0-100)."""
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    pwm.ChangeDutyCycle(speed)

def stop_motor():
    """Stop the motor."""
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    pwm.ChangeDutyCycle(0)

try:
    while True:
        command = input("Enter command (left, right, stop, trash, recycle): ").strip().lower()
        if command == "left":
            rotate_left()
        elif command == "right":
            rotate_right()
        elif command == "stop":
            stop_motor()
        elif command == "trash":
            rotate_right()
            time.sleep(3.5)
            stop_motor()
            time.sleep(1)
            rotate_left()
            time.sleep(3.5)
            stop_motor()
        elif command == "recycle":
            rotate_left()
            time.sleep(3.5)
            stop_motor()
            time.sleep(1)
            rotate_right()
            time.sleep(3.5)
            stop_motor()
        else:
            print("Invalid command. Please enter 'left', 'right', 'trash', 'recycle' or 'stop'.")
except KeyboardInterrupt:
    pass
finally:
    pwm.stop()
    GPIO.cleanup()
