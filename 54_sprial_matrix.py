"""
spiral matrix - 
Given an m x n matrix, return all elements of the matrix in spiral order.
Example 1:
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [1,2,3,6,9,8,7,4,5]
Example 2:
Input: matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
Output: [1,2,3,4,8,12,11,10,9,5,6,7]
"""


def spiral_matrix(matrix: list[list[int]]) -> list[int]:
    result = []
    while matrix:
        result.extend(matrix.pop(0))
        if matrix and matrix[0]:
            for i in range(len(matrix)):
                result.append(matrix[i].pop())
        if matrix and matrix[0]:
            result.extend(matrix.pop()[::-1])
        if matrix and matrix[0]:
            for i in range(len(matrix) - 1, 0, -1):
                result.extend(matrix[i].pop(0))
    return result


print(
    spiral_matrix(matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    "Output: [1,2,3,6,9,8,7,4,5]",
)
print(
    spiral_matrix(matrix=[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]),
    "Output: [1,2,3,4,8,12,11,10,9,5,6,7]",
)
