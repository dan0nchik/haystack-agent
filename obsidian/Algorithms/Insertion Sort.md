
Complexity: $\mathcal{O}(n^2)$

[Visualization](https://visualgo.net/en/sorting)
[Explanation](https://www.geeksforgeeks.org/heap-sort/)

Steps:

for all elements starting from 2nd:
1) Store element in **key**
2) Compare with elements to the left
3) Place the **key** behind the smaller or in the start of the array
![File:Insertion-sort-example.gif - Wikimedia Commons](https://upload.wikimedia.org/wikipedia/commons/9/9c/Insertion-sort-example.gif)
Example:
```cpp
template<typename T>  
void insertion_sort(std::vector<T>& v)  
{  
    for (int step = 1; step < v.size(); ++step)  
    {  
        T key = v[step];  
        T j = step - 1;  
        while (key < v[j] and j >= 0)  
        {  
            v[j + 1] = v[j];  
            j--;  
        }  
        v[j + 1] = key;  
    }  
}
```