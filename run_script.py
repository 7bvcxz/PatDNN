import os

print('Enter run script num')
a = input()

print('run script', a)

if a == '0':
    os.system("python3 main_loss3.py --lr 1e-4 --rho 1e-1")
    os.system("python3 main_loss3.py --lr 1e-4 --rho 6")

elif a == '1':
    os.system("python3 main_loss3.py --lr 1e-4 --rho 10")
    os.system("python3 main_loss3.py --lr 6e-5 --rho 1e-1")
