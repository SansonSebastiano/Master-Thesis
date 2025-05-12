1. È possibile adattare DIFFI su Extended Isolation Forest? (X)
2. ma effettivamente quante foreste, quanti alberi? -> fine-tuning
3. average usage per isolation tree o per isolation forest o per l'intero modello (granularità)? (X)

-> capire quali potrebbero essere le feature significative: osservare le heatmap
    -> strategia per selezionarle?

POSSIBILITA' DI SCARTARE
1) avg usage (score totale) < threshold [percentile] -> si potrebbero provare valori differenti di percentile per valutare quale sia il migliore?
2) teniamo le feature separate e togliamo gli alberi per cui almeno un feature usage è sotto alla threshold
3) e togliere gli alberi per cui TUTTE le feature utili sono sotto una threshold? come definire la treshold?

ATTUALI MODALITÀ DI SCARTO
1. Percentile
2. 