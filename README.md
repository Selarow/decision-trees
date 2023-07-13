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

```
                                                                                                           root 
                                                                                                            │
                                                                                         sepal_width ≤ 2.0 ∨ petal_length ≥ 3.0 
                                                                                      ┌────────────────────┴────────────────────┐
                                                                   sepal_length ≤ 4.9 ∨ petal_length ≥ 4.8                   Setosa 
                                                       ┌──────────────────────────────┴──────────────────────────────┐
                                     sepal_width ≤ 2.2 ∨ petal_width ≥ 1.8                                      Versicolor 
                    ┌──────────────────────────────────┴─────────────────────────────────┐
 sepal_width ≤ 3.0 ∨ sepal_length ≥ 6.0                               sepal_length ≤ 4.9 ∨ petal_length ≥ 5.1 
     ┌──────┴─────┐                                                        ┌─────────────┴────────────┐
 Virginica   Versicolor                                 sepal_length ≤ 6.0 ∨ petal_width ≥ 1.7   Versicolor 
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
