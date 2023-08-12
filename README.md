# Dependencies
- pandas `pip install pandas`
- NumPy `pip install numpy`
- prettyprint `pip install prettyprint`
- PrettyPrintTree `pip install prettyprinttree`

####

# Samples
## Iris
Iris dataset decision tree:
- features = 4
- samples = 150
- max depth = ∞

```
                                   root 
                                    │
                           petal_length ≤ 1.9 
    ┌──────────────────────────────┴─────────────────────────────┐
 Setosa                                                  petal_width ≤ 1.7 
                                    ┌────────────────────────────┴────────────────────────────┐
                           petal_length ≤ 4.9                                        petal_length ≤ 4.8 
                     ┌─────────────┴─────────────┐                                  ┌────────┴───────┐
             petal_width ≤ 1.6           petal_width ≤ 1.5                 sepal_length ≤ 5.9    Virginica 
               ┌─────┴─────┐           ┌─────────┴─────────┐                 ┌─────┴─────┐
          Versicolor   Virginica   Virginica      sepal_length ≤ 6.7    Versicolor   Virginica 
                                                    ┌─────┴─────┐
                                               Versicolor   Virginica 
```

#

Iris dataset multivariate decision tree:
- features = 4
- samples = 150
- max depth = ∞
- execution time ≈ 25 seconds

```
                                                                                                          root 
                                                                                                           │
                                                                                        sepal_width ≤ 2.0 ∨ petal_length ≥ 3.0 
                                                                                     ┌────────────────────┴────────────────────┐
                                                                  sepal_length ≤ 4.9 ∨ petal_length ≥ 4.8                   Setosa 
                                                      ┌──────────────────────────────┴──────────────────────────────┐
                                    sepal_width ≤ 2.2 ∨ petal_width ≥ 1.8                                      Versicolor 
                    ┌─────────────────────────────────┴─────────────────────────────────┐
 sepal_width ≤ 3.0 ∨ sepal_length ≥ 6.0                              sepal_length ≤ 4.9 ∨ petal_length ≥ 5.1 
     ┌──────┴─────┐                                                        ┌────────────┴───────────┐
 Virginica   Versicolor                                           sepal_length ≤ 6.0           Versicolor 
                                                             ┌────────────┴────────────┐
                                          sepal_width ≤ 2.4 ∨ sepal_length ≥ 6.0   Virginica 
                                               ┌─────┴─────┐
                                          Versicolor   Virginica 
```

## Heart Failure
Heart failure dataset decision tree:
- features = 10
- samples = 299
- max depth = 3

```
                                                         root 
                                                          │
                                                     serum ≤ 1.8 
                                    ┌─────────────────────┴─────────────────────┐
                              ejection ≤ 30                                serum ≤ 2.0 
                ┌───────────────────┴──────────────────┐                   ┌────┴────┐
           serum ≤ 0.8                            age ≤ 79.0               1   ejection ≤ 20 
       ┌────────┴────────┐                  ┌─────────┴─────────┐              ┌────┴────┐
 ejection ≤ 20   creatinine ≤ 154   creatinine ≤ 3966   creatinine ≤ 149       1   ejection ≤ 35 
 ┌─┴─┐           ┌─┴─┐              ┌─┴─┐               ┌─┴─┐                      ┌─┴─┐
 1   0           1   0              0   1               1   1                      0   1 
```

#

Heart failure dataset multivariate decision tree:
- features = 10
- samples = 299
- max depth = 3
- execution time ≈ 430 seconds

```
                                                              root 
                                                               │
                                                 ejection ≤ 25 ∨ serum ≥ 1.83 
                                 ┌────────────────────────────┴────────────────────────────┐
                    serum ≤ 1.0 ∨ ejection ≥ 25                            platelets ≤ 75000.0 ∨ age ≥ 80.0 
                        ┌────────┴───────┐                      ┌─────────────────────────┴─────────────────────────┐
           ejection ≤ 25 ∨ serum ≥ 2.1   1        creatinine ≤ 166 ∨ highbp ≥ 1                        serum ≤ 1.0 ∨ ejection ≥ 35 
               ┌────────┴────────┐                ┌─────────┴────────┐                             ┌────────────────┴────────────────┐
 creatinine ≤ 47 ∨ serum ≥ 6.8   1                1   age ≤ 50.0 ∨ creatinine ≥ 5882   serum ≤ 1.1 ∨ age ≥ 65.0   creatinine ≤ 131 ∨ platelets ≥ 334000.0 
 ┌─┴─┐                                                ┌─┴─┐                            ┌─┴─┐                      ┌─┴─┐
 1   0                                                1   0                            0   0                      1   0 
```

## Pumpkin
Pumpkin dataset decision tree:
- features = 12
- samples = 78
- max depth = ∞

```
                                                                                                     root 
                                                                                                      │
                                                                                        major_axis_length ≤ 486.2567 
                                                     ┌───────────────────────────────────────────────┴───────────────────────────────────────────────┐
                                       minor_axis_length ≤ 197.2946                                                                            Ürgüp Sivrisi 
       ┌────────────────────────────────────────────┴────────────────────────────────────────────┐
 Ürgüp Sivrisi                                                                     minor_axis_length ≤ 234.0947 
                                                        ┌───────────────────────────────────────┴──────────────────────────────────────┐
                                               perimeter ≤ 1064.63                                                                Çerçevelik 
                               ┌────────────────────────┴───────────────────────┐
                        extent ≤ 0.6326                                  extent ≤ 0.6601 
                       ┌───────┴──────┐            ┌────────────────────────────┴───────────────────────────┐
                 Ürgüp Sivrisi   Çerçevelik   Çerçevelik                                            solidity ≤ 0.9903 
                                                                                                ┌───────────┴───────────┐
                                                                                          area ≤ 80229            Ürgüp Sivrisi 
                                                                                    ┌──────────┴──────────┐
                                                                          perimeter ≤ 1134.806       Çerçevelik 
                                                                         ┌─────────┴─────────┐
                                                                  extent ≤ 0.7439       Çerçevelik 
                                                                 ┌───────┴──────┐
                                                           Ürgüp Sivrisi   Çerçevelik 
```

#

Pumpkin dataset multivariate decision tree:
- features = 12
- samples = 78
- max depth = ∞
- execution time ≈ 1200 seconds

```
                              root 
                               │
  minor_axis_length ≤ 197.2946 ∨ major_axis_length ≥ 492.8912 
       ┌───────────────────────┴───────────────────────┐
 Ürgüp Sivrisi                perimeter ≤ 1064.63 ∨ minor_axis_length ≥ 236.9788 
                                  ┌───────────────────┴───────────────────┐
                 area ≤ 65320 ∨ convex_area ≥ 68507        extent ≤ 0.6601 ∨ area ≥ 86344 
                      ┌──────┴──────┐                      ┌─────────────┴────────────┐
                 Çerçevelik   Ürgüp Sivrisi           Çerçevelik     extent ≤ 0.6945 ∨ solidity ≥ 0.9905 
                                                                         ┌────────────┴────────────┐
                                                                   Ürgüp Sivrisi   area ≤ 75637 ∨ solidity ≥ 0.9877 
                                                                                        ┌──────┴──────┐
                                                                                   Çerçevelik   Ürgüp Sivrisi 
```

## Raisin
Raisin dataset decision tree:
- features = 7
- samples = 112
- max depth = ∞

```
                                                                 root 
                                                                  │
                                                   major_axis_length ≤ 394.340189 
                                      ┌──────────────────────────┴──────────────────────────┐
                             convex_area ≤ 70274                             major_axis_length ≤ 452.7756953 
                           ┌──────────┴──────────┐                           ┌──────────────┴─────────────┐
                 extent ≤ 0.824319225      area ≤ 69746                area ≤ 78632                     Besni 
                 ┌────────┴────────┐       ┌────┴───┐        ┌──────────────┴──────────────┐
    eccentricity ≤ 0.566991437   Besni   Besni   Kecimen   Besni                 extent ≤ 0.725472034 
         ┌──────┴─────┐                                                            ┌──────┴──────┐
   area ≤ 48488    Kecimen                                                   area ≤ 83248      Besni 
   ┌────┴───┐                                                               ┌─────┴─────┐
 Besni   Kecimen                                                      area ≤ 81456   Kecimen 
                                                                      ┌────┴───┐
                                                                   Kecimen   Besni 
```

#

Raisin dataset multivariate decision tree:
- features = 7
- samples = 112
- max depth = ∞
- execution time ≈ 900 seconds

```
                                                                            root 
                                                                             │
                                                  extent ≤ 0.625069042 ∨ major_axis_length ≥ 419.3385795 
                            ┌───────────────────────────────────────────────┴──────────────────────────────────────────────┐
 major_axis_length ≤ 434.1002588 ∨ perimeter ≥ 1207.534                                                   convex_area ≤ 70274 ∨ area ≥ 73311 
   ┌──────────────┴─────────────┐                                                                         ┌───────────────┴───────────────┐
 Besni   area ≤ 84057 ∨ minor_axis_length ≥ 275.5829735                         perimeter ≤ 859.326 ∨ major_axis_length ≥ 407.9403285   Besni 
            ┌────┴───┐                                                                  ┌─────────────────┴─────────────────┐
         Kecimen   Besni                                  eccentricity ≤ 0.566991437 ∨ major_axis_length ≥ 323.5892546   Kecimen 
                                                                  ┌──────┴─────┐
                                                            area ≤ 78632    Kecimen 
                                                            ┌────┴───┐
                                                          Besni   Kecimen 
```
