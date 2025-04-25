
Complexity: $\mathcal{O}(n\log{}n)$

[Visualization](https://www.cs.usfca.edu/~galles/visualization/HeapSort.html)
[Explanation](https://www.geeksforgeeks.org/heap-sort/)

Steps:

1) Create max [[Heap]] from unsorted array => the root is the largest number after heapifying
2) Swap the root with last element
3) Remove last element from heap
4) Heapify again until the heap is empty (all the roots will be in the array ascending)

Example:

```pseudo
void heapify(vector, rootIndex){
	l = leftChildIndex
	r = rightChildIndex
	max = (left if left > root OR right if right > root) AND (less than vector length)
	if (max is not root)
	{
		swap(root, max)
		heapify(vector, max) <- max is not the root so we repeat recursively
	}
}

void heap_sort(vector){
	// Build Max Heap
	for (i = N / 2 - 1; i >= 0; i--)
		heapify(vector, i)

	// Remove root from heap 
	for (i = N - 1; i > 0; i--){
		
		swap(vector[0], vector[i]) // move root to the end
		
		heapify(vector, i) // heapify remaining heap
	}
}
```