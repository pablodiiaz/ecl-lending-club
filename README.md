# Pérdida esperada (ECL) – Lending Club

## El problema

El Comité de Riesgos de un banco necesita estimar la **pérdida esperada (ECL, Expected Credit Loss)** de una cartera de préstamos personales de Lending Club (más de 2,1 millones de préstamos) para dotar provisiones. Para aceptar la estimación, exige:

1. **Provisión suficiente**: el ECL total debe superar la pérdida real de la cartera.
2. **Pocos préstamos infraprovisionados**: como mucho un 10% de los préstamos puede tener un ECL por debajo de su pérdida real.
3. Cumplido lo anterior, **el ECL total más bajo posible** (provisionar de más inmoviliza capital) y la **mayor precisión préstamo a préstamo** (MAE y MAPE).

## Enfoque

Se usa el enfoque estándar de riesgo de crédito:

**ECL = PD × LGD × EAD**

- **PD** (Probability of Default): probabilidad de impago de cada préstamo.
- **EAD** (Exposure at Default): cuánto se debe en el momento del impago.
- **LGD** (Loss Given Default): qué parte de esa deuda no se recupera.

## Estructura

| Notebook | Contenido |
|---|---|
| [`01_eda.ipynb`](01_eda.ipynb) | Carga de datos, clasificación de variables (originación / performance / prohibidas), EDA estructural y frente al impago, transformaciones fijas, split train / validación / test y variables nuevas |
| [`02_modelo_pd.ipynb`](02_modelo_pd.ipynb) | Outliers e imputación, selección de variables (IV y VIF), logística frente a LightGBM, ajuste de hiperparámetros, modelo de PD final y su calibración |
| [`03_ead_lgd_ecl.ipynb`](03_ead_lgd_ecl.ipynb) | Construcción y validación de los targets de EAD y LGD, sus modelos, ECL, margen conservador y resultado final en test |

Los notebooks se encadenan por ficheros en `data/processed/`:

```
X.csv, target.csv ──► 01_eda ──► train.pkl, val.pkl, test.pkl, variables.json
                                        │
                                        ├──► 02_modelo_pd ──► pd_val_test.pkl
                                        │                          │
                                        └──────────────────────────┴──► 03_ead_lgd_ecl ──► resultado final
```

## Decisiones principales

- **Sin leakage**: solo variables de originación como predictores. Las variables de performance se usan únicamente para construir los targets de EAD y LGD, y `grade` está excluida.
- **Todo lo que aprende de los datos se ajusta solo con train**: outliers, imputación, encodings y modelos. La validación se usa para el early stopping y para calibrar el margen. Test se usa una única vez, al final.
- **Madurez del préstamo**: las añadas recientes aún no han tenido tiempo de impagar ni de recobrar. Las variables de madurez (`meses_en_cartera`, `pct_plazo_transcurrido`) separan el riesgo del cliente del tiempo de observación.
- **Modelos**: LightGBM en las tres piezas. Para la EAD y la LGD, con objetivo `cross_entropy` sobre el CCF (EAD / importe) y la LGD, que están entre 0 y 1.
- **Margen conservador**: cobertura selectiva de los préstamos con mayor PD / (LGD × EAD), calibrada en validación y comprobada con bootstrap y con validación en dos mitades.

## Resultado en test

| | ECL total | ECL / pérdida real | % préstamos con ECL < pérdida | MAE | MAPE |
|---|---|---|---|---|---|
| Sin margen | 557,5 M€ | 1,004 | 12,79% | 1.977 € | 81,9% |
| **Con margen** | **850,4 M€** | **1,531** | **9,73%** | **2.452 €** | **92,4%** |

## Cómo ejecutarlo

1. Descargar el dataset de Kaggle *lending-club-loan-data-cleared* y dejar `X.csv` y `target.csv` en la raíz del proyecto.
2. Ejecutar los notebooks **en orden**: `01_eda` → `02_modelo_pd` → `03_ead_lgd_ecl`.

`02_modelo_pd` es el más lento: unos 15–20 minutos, sobre todo por la búsqueda de hiperparámetros y el reentrenamiento con todo train.

**Requisitos** (versiones usadas): Python 3.13, pandas 2.3, numpy 2.3, scikit-learn 1.9, lightgbm 4.7, matplotlib 3.10.
