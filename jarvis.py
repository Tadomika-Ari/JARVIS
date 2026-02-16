#!/usr/bin/env python3

from hand_detection_yolo import Yolo

def init():
    print("\n")
    print("Bonjour utilisateur\n")
    print("Que voulez vous faire ?\n")
    print("0 : Yolo")
    choice = input("donne ton choix :")

    if (int(choice) == 0):
        Yolo()

if __name__ == "__main__":
    init()
