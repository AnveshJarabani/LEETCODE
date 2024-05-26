"""
Given a positive integer n,
generate an n x n matrix filled with elements from 1 to n2 in spiral order.
Example 1:
Input: n = 3
Output: [[1,2,3],[8,9,4],[7,6,5]]
Example 2:
Input: n = 1
Output: [[1]]
"""
def spiral_matrix(n: int) -> list[list[int]]:
    nums = [i for i in range(1,n*n+1)]
    matrix = [[] for _ in range(n)]
    for row in range(n):
        matrix[row] = nums[1:n+1]
        
