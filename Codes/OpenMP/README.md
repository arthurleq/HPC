# OpenMP

OpenMP est un modèle de programmation associé à la mémoire partagée, tel une API permettant de programmer sur ce type d'architecture.
Le principe est de créer plusieurs threads qui travaillent en parallèle.
Par exemple, si on travaille sur une boucle, il est possible de répartir cette dernière entre plusieurs threads.

Le but de ce dossier est d'éxpliquer le fonctionnement pratique de OpenMP, ainsi sont présent different fichier C++ présentant différents aspects de d'OpenMP. 

---

## `1_OpemMP_comparaison.cpp` — stratégies de synchronisation mémoire

### Objectif

Calculer la somme des éléments d'un tableau de 10 millions d'entiers, en parallèle, avec quatre stratégies différentes pour protéger la variable partagée `sum` — et voir laquelle est la plus rapide.

### Les 4 fonctions

| Fonction | Clause OpenMP | Quand la synchro a lieu | Coût |
|---|---|---|---|
| `sum_table_unprotected` | aucune | jamais | résultat **faux** |
| `sum_table_reduction` | `reduction(+:sum)` | une seule fois, à la fin | quasi nul |
| `sum_table_atomic` | `atomic` | à chaque itération (instruction matérielle) | faible mais non nul |
| `sum_table_critical` | `critical` | à chaque itération (verrou logiciel) | élevé |

**`sum_table_unprotected`** — plusieurs threads font `sum += table[i]` en même temps sur la même variable. C'est une *race condition* : le `+=` n'est pas une opération atomique (il se décompose en lecture, addition, écriture), donc des mises à jour se perdent quand deux threads lisent la même valeur de `sum` avant que l'un des deux n'ait fini d'écrire. Le résultat est incorrect et change d'une exécution à l'autre — cette fonction n'est là que pour illustrer le problème, jamais à utiliser en pratique.

**`sum_table_reduction`** — chaque thread reçoit une copie privée de `sum` (initialisée à 0, l'élément neutre de `+`), accumule dedans sans jamais toucher aux copies des autres, puis OpenMP fusionne toutes les copies en une seule fois à la sortie de la boucle. Comme il n'y a aucune synchronisation *pendant* la boucle, c'est quasiment gratuit : c'est la solution à privilégier dès que l'opération s'y prête (somme, produit, min, max, et, ou...).

**`sum_table_atomic`** — utilise une instruction machine dédiée (typiquement `lock xadd` sur x86) qui garantit l'atomicité au niveau du bus mémoire/cache, mais uniquement pour une opération simple comme `+=`. C'est rapide comparé à un verrou générique, mais ça reste une synchronisation matérielle à chaque itération.

**`sum_table_critical`** — section critique générique (verrou/mutex), capable de protéger n'importe quel bloc de code, pas seulement une opération scalaire. C'est l'outil le plus polyvalent mais aussi le plus lourd : chaque thread doit acquérir puis relâcher le verrou à chaque itération, ce qui sérialise fortement l'exécution.

### Ce qu'on doit observer à l'exécution

Le tri attendu, du plus rapide au plus lent : `reduction` < `atomic` < `critical`, avec `unprotected` qui donne un temps très court mais **un résultat faux** (à comparer avec `reel_sum`, calculé analytiquement).

### Compiler et exécuter

```bash
g++ -fopenmp -O2 1_OpemMP_comparaison.cpp -o comparaison_memoire
./comparaison_memoire
```

---

## `2_OpenMP_scheduling.cpp` — politiques d'ordonnancement (scheduling)

### Objectif

Remplir un tableau où `table[i]` est le nombre d'étapes de la suite de Collatz partant de `i+1`, en parallèle, avec quatre politiques de répartition des itérations entre threads.

### Pourquoi la suite de Collatz ?

Le coût de `collatz_steps(i)` varie énormément et de façon imprévisible selon `i` : certains nombres convergent vers 1 en quelques étapes, d'autres en prennent des centaines. C'est une charge de travail **irrégulière** — exactement ce qu'il faut pour rendre visible l'effet du scheduling. Avec une boucle où chaque itération coûterait pareil, tous les schedules donneraient des temps quasi identiques et la démo n'aurait aucun intérêt.

### Les 4 fonctions

| Fonction | Clause OpenMP | Décidé quand | Taille de bloc |
|---|---|---|---|
| `fill_static` | `static` (défaut) | avant la boucle | `n / nb_threads`, un seul bloc contigu par thread |
| `fill_static_chunk` | `static, 64` | avant la boucle | fixe, distribuée en round-robin |
| `fill_dynamic` | `dynamic, 64` | à l'exécution | fixe, distribuée à la demande |
| `fill_guided` | `guided` | à l'exécution | décroissante (grande au début, petite à la fin) |

**`fill_static`** — le schedule par défaut. L'espace d'itération est coupé en `P` blocs contigus (`P` = nombre de threads), chaque thread reçoit exactement un bloc, décidé une bonne fois pour toutes avant que la boucle ne démarre. Zéro overhead à l'exécution, mais si les itérations "lourdes" se regroupent dans un seul bloc (ce qui arrive facilement avec Collatz), ce thread devient le goulot d'étranglement pendant que les autres attendent, inactifs.

**`fill_static_chunk`** — même logique de découpage à l'avance, mais en petits blocs (chunks) de taille fixe distribués en round-robin entre threads (thread 0 reçoit les chunks 0, P, 2P..., thread 1 reçoit 1, P+1... etc). Ça mélange mieux le lourd et le léger entre threads, toujours sans aucune synchronisation pendant la boucle.

**`fill_dynamic`** — les chunks sont distribués à la demande, à l'exécution : dès qu'un thread termine son chunk, il revient piocher le suivant dans une file partagée. Très bon équilibrage pour une charge irrégulière comme celle-ci, mais chaque distribution coûte une (petite) synchronisation sur la file.

**`fill_guided`** — même principe que `dynamic` (distribution à l'exécution depuis une file partagée), mais la taille des chunks démarre grande et décroît géométriquement au fil de la boucle. Moins de distributions que `dynamic` (donc moins d'overhead), tout en s'adaptant finement vers la fin de la boucle — c'est précisément là que le déséquilibre coûte le plus cher en temps d'attente.

### Ce qu'on doit observer à l'exécution

Les 4 fonctions produisent la **même somme totale** (vérifié par `sum_table` en fin de `main`) — le schedule change *comment* le travail est réparti entre threads, jamais *ce qui* est calculé. En revanche les temps diffèrent : `static` (défaut) est généralement le plus déséquilibré sur ce type de charge, `dynamic` et `guided` compensent au prix d'un peu d'overhead, et `static, chunk=64` se situe souvent entre les deux.

### Compiler et exécuter

```bash
g++ -fopenmp -O2 2_OpenMP_scheduling.cpp -o comparaison_scheduling
./comparaison_scheduling
```

---

## Pour aller plus loin

- Fixer le nombre de threads pour des comparaisons reproductibles : `OMP_NUM_THREADS=4 ./comparaison_scheduling`.
- Dans le fichier scheduling, faire varier `chunk_size` (essayer 1, 8, 64, 512...) et regarder à partir de quelle taille l'overhead de `dynamic` dépasse le gain d'équilibrage.
- Faire varier `n` dans les deux fichiers : plus `n` est grand, plus l'overhead de synchronisation (par itération) pèse relativement moins lourd face au calcul lui-même.
- Croiser les deux fichiers : rien n'empêche d'ajouter `schedule(dynamic)` à une boucle qui utilise aussi `reduction` — les deux clauses répondent à des questions indépendantes (comment répartir le travail / comment agréger un résultat partagé).
