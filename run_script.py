import os

print('run script :')
a = input()

if a == '0':
    os.system("python3 main_loss3.py --lr 1e-4 --rho 2")
    os.system("python3 main_loss3.py --lr 1e-4 --rho 2.5")

elif a == '1':
    os.system("python3 main_swp.py --lr 1e-4 --rho 1e-1")
    os.system("python3 main_swp.py --lr 6e-5 --rho 1e-1")
