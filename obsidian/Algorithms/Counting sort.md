
Complexity: $\mathcal{O}(n+k)$

[Visualization](https://www.cs.usfca.edu/~galles/visualization/CountingSort.html)
[Explanation](https://www.programiz.com/dsa/counting-sort)

Steps:

1) Find *max* element in array
2) Create **count** array of length (*max* + 1)
3) Store the count of each element with respective index in **count** array
4) Find and store [[Cumulative Sum]] in **count** array
5) Restore original array as on pic ![[Count sort image.png]]