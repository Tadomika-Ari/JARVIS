#!/usr/bin/env python3

from main import *

def init():
    print("\n")
    print("Bonjour utilisateur\n")
    print("Que voulez vous faire ?\n")
    print("0 : Yolo")
    choice = input("donne ton choix :")

    if (int(choice) == 0):
        main

if __name__ == "__main__":
    init()
