#!/usr/bin/env python3

import cv2
import numpy as np
import torch

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam")
        return
    
    print("Détection de mains par couleur de peau")
    print("Appuyez sur 'q' pour quitter")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir en HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Définir la plage de couleur de peau
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Créer un masque
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Appliquer des opérations morphologiques
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Trouver les contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Dessiner les contours des mains détectées
        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 5000:  # Filtrer les petits contours
                    # Dessiner le contour
                    cv2.drawContours(frame, [cnt], 0, (0, 255, 0), 2)
                    
                    # Obtenir la boîte englobante
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(frame, "Main", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Afficher le résultat
        cv2.imshow('Detection de mains', frame)
        
        # Quitter avec 'q' ou 'h'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
     
    #liberer mémoire
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
