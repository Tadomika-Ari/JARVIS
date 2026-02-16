#!/usr/bin/env python3

import cv2
from ultralytics import YOLO

def main():
    # Charger le modèle YOLO pré-entraîné
    # Il peut détecter les personnes, on peut affiner pour les mains
    model = YOLO('yolov8n.pt')  # nano version (rapide)
    
    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Erreur : impossible d'ouvrir la webcam")
        return
    
    print("Détection avec YOLO + PyTorch")
    print("Appuyez sur 'q' pour quitter")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip horizontal pour effet miroir
        frame = cv2.flip(frame, 1)
        
        # Prédiction avec YOLO
        results = model(frame, verbose=False)
        
        # Dessiner les détections
        annotated_frame = results[0].plot()
        
        # Afficher le résultat
        cv2.imshow('YOLO Detection', annotated_frame)
        
        # Quitter avec 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
