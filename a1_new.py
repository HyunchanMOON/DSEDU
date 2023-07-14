import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)

servo_pin = 18
pir_pin = 21

GPIO.setup(servo_pin, GPIO.OUT)
GPIO.setup(pir_pin, GPIO.IN)

pwm = GPIO.PWM(servo_pin, 50)
pwm.start(0)

def move_to_neutral():
    pwm.ChangeDutyCycle(0)
    time.sleep(0.5)

def get_average_value(pir_pin, num_samples=5):
    total_cnt = 0
    for _ in range(num_samples):
        if GPIO.input(pir_pin) == GPIO.HIGH:
            total_cnt += 1
    return True if total_cnt > 3 else False
    
try:
    while True:
        average_value = get_average_value(pir_pin, num_samples=5)
        if average_value is True:
            print("Motion detected")
            move_to_neutral()

            high_time = 30
            while high_time < 125:
                pwm.ChangeDutyCycle(high_time / 10.0)
                time.sleep(0.02)
                high_time += 1

            high_time = 124
            while high_time > 30:
                pwm.ChangeDutyCycle(high_time / 10.0)
                time.sleep(0.02)
                high_time -= 1

            move_to_neutral()
        else:
            print("No motion detected")
            pwm.ChangeDutyCycle(0)
            time.sleep(0.5)

except KeyboardInterrupt:
    pass

pwm.stop()
GPIO.cleanup()
