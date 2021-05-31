import os

print('Enter run script num')
a = input()

print('run script', a)

if a == '0':
    os.system("python3 main_loss3.py --lr 6e-5 --rho 6")
    os.system("python3 main_loss3.py --lr 6e-5 --rho 10")

elif a == '1':
    os.system("python3 main_loss3.py --lr 3e-4 --rho 1")
    os.system("python3 main_loss3.py --lr 3e-4 --rho 3")
