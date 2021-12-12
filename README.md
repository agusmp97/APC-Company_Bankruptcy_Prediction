# Pràctica Kaggle APC UAB 2021-22
### Nom: Agustín Martínez
### DATASET: Company Bankruptcy Prediction
### URL: [kaggle](https://www.kaggle.com/fedesoriano/company-bankruptcy-prediction)

## Resum
El dataset utilitza dades extretes del Taiwan Economic Journal entre els anys 1999 i 2009. Així, per determinar quan una empresa és fallida (Bankruptcy = 1) s'utilitzen les regulacions comercials definides a la Borsa de Valors de Taiwan.

El conjunt de dades està format per 6819 registres, amb 96 atributs cadascun d'ells. D'aquests, 95 atributs s'utilitzaran com a característiques d'entrada al model, mentre que l'atribut "Bankrupt?" és l'atribut objectiu o *target* que es vol predir.

Concretament, la llista d'atributs d'aquest dataset és:

 0   Bankrupt?                                                 6819 non-null   int64  
 1    ROA(C) before interest and depreciation before interest  6819 non-null   float64
 2    ROA(A) before interest and % after tax                   6819 non-null   float64
 3    ROA(B) before interest and depreciation after tax        6819 non-null   float64
 4    Operating Gross Margin                                   6819 non-null   float64
 5    Realized Sales Gross Margin                              6819 non-null   float64
 6    Operating Profit Rate                                    6819 non-null   float64
 7    Pre-tax net Interest Rate                                6819 non-null   float64
 8    After-tax net Interest Rate                              6819 non-null   float64
 9    Non-industry income and expenditure/revenue              6819 non-null   float64
 10   Continuous interest rate (after tax)                     6819 non-null   float64
 11   Operating Expense Rate                                   6819 non-null   float64
 12   Research and development expense rate                    6819 non-null   float64
 13   Cash flow rate                                           6819 non-null   float64
 14   Interest-bearing debt interest rate                      6819 non-null   float64
 15   Tax rate (A)                                             6819 non-null   float64
 16   Net Value Per Share (B)                                  6819 non-null   float64
 17   Net Value Per Share (A)                                  6819 non-null   float64
 18   Net Value Per Share (C)                                  6819 non-null   float64
 19   Persistent EPS in the Last Four Seasons                  6819 non-null   float64
 20   Cash Flow Per Share                                      6819 non-null   float64
 21   Revenue Per Share (Yuan ¥)                               6819 non-null   float64
 22   Operating Profit Per Share (Yuan ¥)                      6819 non-null   float64
 23   Per Share Net profit before tax (Yuan ¥)                 6819 non-null   float64
 24   Realized Sales Gross Profit Growth Rate                  6819 non-null   float64
 25   Operating Profit Growth Rate                             6819 non-null   float64
 26   After-tax Net Profit Growth Rate                         6819 non-null   float64
 27   Regular Net Profit Growth Rate                           6819 non-null   float64
 28   Continuous Net Profit Growth Rate                        6819 non-null   float64
 29   Total Asset Growth Rate                                  6819 non-null   float64
 30   Net Value Growth Rate                                    6819 non-null   float64
 31   Total Asset Return Growth Rate Ratio                     6819 non-null   float64
 32   Cash Reinvestment %                                      6819 non-null   float64
 33   Current Ratio                                            6819 non-null   float64
 34   Quick Ratio                                              6819 non-null   float64
 35   Interest Expense Ratio                                   6819 non-null   float64
 36   Total debt/Total net worth                               6819 non-null   float64
 37   Debt ratio %                                             6819 non-null   float64
 38   Net worth/Assets                                         6819 non-null   float64
 39   Long-term fund suitability ratio (A)                     6819 non-null   float64
 40   Borrowing dependency                                     6819 non-null   float64
 41   Contingent liabilities/Net worth                         6819 non-null   float64
 42   Operating profit/Paid-in capital                         6819 non-null   float64
 43   Net profit before tax/Paid-in capital                    6819 non-null   float64
 44   Inventory and accounts receivable/Net value              6819 non-null   float64
 45   Total Asset Turnover                                     6819 non-null   float64
 46   Accounts Receivable Turnover                             6819 non-null   float64
 47   Average Collection Days                                  6819 non-null   float64
 48   Inventory Turnover Rate (times)                          6819 non-null   float64
 49   Fixed Assets Turnover Frequency                          6819 non-null   float64
 50   Net Worth Turnover Rate (times)                          6819 non-null   float64
 51   Revenue per person                                       6819 non-null   float64
 52   Operating profit per person                              6819 non-null   float64
 53   Allocation rate per person                               6819 non-null   float64
 54   Working Capital to Total Assets                          6819 non-null   float64
 55   Quick Assets/Total Assets                                6819 non-null   float64
 56   Current Assets/Total Assets                              6819 non-null   float64
 57   Cash/Total Assets                                        6819 non-null   float64
 58   Quick Assets/Current Liability                           6819 non-null   float64
 59   Cash/Current Liability                                   6819 non-null   float64
 60   Current Liability to Assets                              6819 non-null   float64
 61   Operating Funds to Liability                             6819 non-null   float64
 62   Inventory/Working Capital                                6819 non-null   float64
 63   Inventory/Current Liability                              6819 non-null   float64
 64   Current Liabilities/Liability                            6819 non-null   float64
 65   Working Capital/Equity                                   6819 non-null   float64
 66   Current Liabilities/Equity                               6819 non-null   float64
 67   Long-term Liability to Current Assets                    6819 non-null   float64
 68   Retained Earnings to Total Assets                        6819 non-null   float64
 69   Total income/Total expense                               6819 non-null   float64
 70   Total expense/Assets                                     6819 non-null   float64
 71   Current Asset Turnover Rate                              6819 non-null   float64
 72   Quick Asset Turnover Rate                                6819 non-null   float64
 73   Working capitcal Turnover Rate                           6819 non-null   float64
 74   Cash Turnover Rate                                       6819 non-null   float64
 75   Cash Flow to Sales                                       6819 non-null   float64
 76   Fixed Assets to Assets                                   6819 non-null   float64
 77   Current Liability to Liability                           6819 non-null   float64
 78   Current Liability to Equity                              6819 non-null   float64
 79   Equity to Long-term Liability                            6819 non-null   float64
 80   Cash Flow to Total Assets                                6819 non-null   float64
 81   Cash Flow to Liability                                   6819 non-null   float64
 82   CFO to Assets                                            6819 non-null   float64
 83   Cash Flow to Equity                                      6819 non-null   float64
 84   Current Liability to Current Assets                      6819 non-null   float64
 85   Liability-Assets Flag                                    6819 non-null   int64
 86   Net Income to Total Assets                               6819 non-null   float64
 87   Total assets to GNP price                                6819 non-null   float64
 88   No-credit Interval                                       6819 non-null   float64
 89   Gross Profit to Sales                                    6819 non-null   float64
 90   Net Income to Stockholder's Equity                       6819 non-null   float64
 91   Liability to Equity                                      6819 non-null   float64
 92   Degree of Financial Leverage (DFL)                       6819 non-null   float64
 93   Interest Coverage Ratio (Interest expense to EBIT)       6819 non-null   float64
 94   Net Income Flag                                          6819 non-null   int64
 95   Equity to Liability                                      6819 non-null   float64
dtypes: float64(93), int64(3)

Així, com es pot observar, la gran majoria dels atributs és de tipus float64, mentre que únicament 3 d'ells són de tipus int64. A més, cal destacar que, segurament degut a que són dades de caràcter oficial, no hi ha cap valor inexistent al conjunt de dades, la qual cosa aporta un gran avantatge ja que, encara que algunes d'elles puguin contenir errors de diversos tipus (d'introducció, de mesura, de càlcul, etc.), no s'han de fer suposicions ni omplir buits de dades.

Finalment, cal mencionar que originalment aquests atributs no estan normalitzats, pel que hi ha alguns valors que es troben entre 0 i 1 i altres que són vàries ordres de magnitud més grans, pel que s'han de processar per normalitzar-los, tal com s'explica a l'apartat "Preprocessat".



### Objectius del dataset
Com es pot intuir, l'objectiu principal d'aquest dataset és determinar quan, a partir de les dades econòmiques reportades per l'empresa, si aquesta empresa serà fallida en el futur o no.

## Experiments
Durant la realitzaició d'aquesta pràctica s'han realitzat diversos experiments per aconseguir l'objectiu proposat.
Inicialment, s'han processat les dades per poder tenir una visió global d'elles, comprendre quines són les seves característiques i fer-les comparables entre elles per poder entrenar correctament els models que, posteriorment, intentaran fer les prediccions a partir d'aquestes dades processades.

El següent pas, una vegada el processament de dades s'ha finalitzat, s'han creat els models classificadors, emprant la llibreria oferida per scikit learn. Concretament, els models implementats han estat un regressor logístic (LogisticRegression), una màquina de vectors de suport (SVC), un K Nearest Neighbors (KNeighborsClassifier) i un Random Forest (RandomForestClassifier).
Tots ells s'han utilitzat com a estimadors d'entrada del mecanisme de cerca d'hiperparàmetres GridSearch (GridSearchCV), el qual ha permès trobar els millors hiperparàmetres per cadascun dels classificadors per ajustar-se a les dades del dataset.

Finalment, s'ha creat una demostració (demo.py) del funcionament dels models entrenats per demostrar la seva eficàcia en el moment de fer prediccions amb dades que no han utilitzat per ser entrenats.

### Preprocessat
Abans de realitzar el preprocessat de les dades, una de les primeres accions que s'ha dur a terme ha estat comprovar si les dades del dataset estan balancejades, i no ho estan (només el 3,23% de les dades són d'empreses fallides).
Seguidament, s'ha creat una Matriu de Correlació de Pearson per determinar quins atributs tenen més relació amb l'atribut objectiu, i el resultat ha estat que cap atribut té una relació notable, ja sigui directa o inversa, en la predicció de l'atribut objeciu.

Després d'aquest anàlisi de les dades, s'ha començat amb el preprocessament de les dades. Concretament, el que s'ha fet és:
1. S'han eliminat els espais dels noms dels atributs
2. S'ha eliminat l'atribut NetIncomeFlag, ja que per a tots els registres el seu valor és 1.
3. S'ha comprovat si hi havia valors nuls al dataset i, en aquest cas, no hi ha cap.
4. S'han separat les dades en un conjunt d'entrenament (training) i de validació (test).
5. S'han normalitzat les dades. Per a les dades d'entrenament, s'ha utilitzat la funció `fit_transform()` de MinMaxScaler. En canvi, per a les dades de validació, s'ha utilitzat la funció `transform()` de MinMaxScaler.


### Model
| Model | Hiperparàmetres | Mètrica | Temps |
| -- | -- | -- | -- |
| Logistic Regressor | 'C': 1, 'penalty': 'l2', 'solver': 'lbfgs' | f1-score: Classe 0: 0.98, Classe 1: 0.09 | 341.66 s |
| Support Vectors Classifier | 'C': 1, 'degree': 3, 'kernel': 'linear' | f1-score: Classe 0: 0.98, Classe 1: 0.08 | 506.07 s |
| K Nearest Neighbors | 'n\_neighbors': 10, 'p': 1, 'weights': 'uniform' | f1-score: Classe 0: 0.98, Classe 1: 0.11 | 16.42 s |
| Random Forest Classifier | 'criterion': 'gini', 'max\_depth': 8, 'max\_features': 'sqrt', 'n\_estimators': 200 | f1-score: Classe 0: 0.98, Classe 1: 0.16 | 614.19 s |
| -- | -- | -- | -- |

## Demo
Per tal de fer una prova, es pot fer servir amb la següent comanda:
`python3 demo/demo.py --input here`

## Conclusions
El millor model que s'ha aconseguit ha estat el KNN, ja que, encara que els resultats de les prediccions són molt semblants als altres models, el temps d'entrenament del model és notablement inferior.
En comparació amb l'estat de l'art i els altres treballs que s'han analitzat, els models implementats en aquest projecte obtenen uns resultats similars als explorats. Això és degut a les poques dades d'empreses fallides que conté el dataset, pel que no es poden optimitzar gaire més els models sense fer un resampling de les dades.

Així, cal destacar que com les dades estan tan desbalancejades (només un 3,23% de les empreses són fallides), és complex per als classificadors predir amb un gran nombre d'encerts les empreses que són fallides, ja que no es disposen de dades suficients d'aquest tipus que permetin fer un entrenament exhaustiu.

## Idees per treballar en un futur
Com a millora important d'aquest projecte cal destacar la obtenció de dades d'empreses fallides. Això vol dir que, si no es poden obtenir suficients dades reals d'empreses fallides, s'ha d'utilitzar algun mecanisme de generació de dades fictícies per tal de balancejar el conjunt de dades i poder entrenar millor els models.

Durant el desenvolupament actual del projecte, s'han realitzat proves amb tècniques senzilles de *resampling* de les dades, el que vol dir que s'ha utilitzat la utilitat `resample` de `sklearn.utils`, però no s'han obtingut bons resultats, ja que el que fa aquesta utilitat és duplicar, aleatòriament, les mostres que compleixen una certa condició. Però, amb aquest mecanisme, i degut a que hi ha tan poques dades on l'empresa cau en fallida, respecte de les que no cauen en fallida, que els models entrenats amb aquest mecanisme patien un *overfitting* molt important, i els resultats, per tant, no eren vàlids per poder generalitzar amb el model.

## Llicència
El projecte s’ha desenvolupat sota llicència Apache 2.0.

