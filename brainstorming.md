1. È possibile adattare DIFFI su Extended Isolation Forest?
2. ma effettivamente quante foreste, quanti alberi?
3. approccio più corretto per la injection delle threshold

-> Scaling delle `two most important features usage` utilizzando il max, per omogeneizzare la scala, mentre per F1 score è stata normalizzata con sum (ma in pratica ho cambiato con max anche qua)
    - Ho pensato di fare così per le `two most important features usage` perchè per ogni albero vorrei avere la percentuale d'uso, dunque il valore massimo scalerà ad 1 mentre il valore minimo a 0 (non so se sia effettivamente l'approccio più corretto)
    - Faccendo la normalizzazione con sum, ho notato che 


