| Modelo | Condición | Umbral | Exactitud | Precisión | Recall | F1 | AUC | ms/audio |
|---|---|---|---|---|---|---|---|---|
| CNN | A · preentrenado | 0.5 | 0.613 | 0.989 | 0.193 | 0.323 | 0.794 | 0.6 |
| CNN | A · preentrenado | validación | 0.727 | 0.748 | 0.649 | 0.695 | 0.794 | 0.6 |
| CNN | B · preentrenado + afinado | 0.5 | 0.991 | 0.981 | 1.000 | 0.990 | 1.000 | 0.6 |
| CNN | B · preentrenado + afinado | validación | 0.970 | 1.000 | 0.937 | 0.968 | 1.000 | 0.6 |
| CNN | C · solo español | 0.5 | 0.999 | 0.998 | 1.000 | 0.999 | 1.000 | 0.6 |
| CNN | C · solo español | validación | 0.982 | 1.000 | 0.963 | 0.981 | 1.000 | 0.6 |
| LSTM | A · preentrenado | 0.5 | 0.631 | 0.901 | 0.258 | 0.401 | 0.818 | 3.4 |
| LSTM | A · preentrenado | validación | 0.732 | 0.736 | 0.688 | 0.711 | 0.818 | 3.4 |
| LSTM | B · preentrenado + afinado | 0.5 | 0.989 | 1.000 | 0.976 | 0.988 | 1.000 | 3.6 |
| LSTM | B · preentrenado + afinado | validación | 0.994 | 1.000 | 0.987 | 0.993 | 1.000 | 3.6 |
| LSTM | C · solo español | 0.5 | 0.998 | 1.000 | 0.996 | 0.998 | 1.000 | 3.6 |
| LSTM | C · solo español | validación | 0.978 | 1.000 | 0.954 | 0.977 | 1.000 | 3.6 |
