"""
Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.
You must do it in place.
Example 1:
Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]
Example 2:
Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
"""


class solution:
    def set_matrix_zeroes(self, matrix: list[list[int]]) -> list[list[int]]:
        # scan rows and cols for zeroes and record them in the sets.
        rows, cols = set(), set()
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == 0:
                    rows.add(i)
                    cols.add(j)

        # Traverse all cells and keep making the value zero if the row/ col is in the sets.
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if i in rows or j in cols:
                    matrix[i][j] = 0
        return matrix


print(
    solution().set_matrix_zeroes(matrix=[[1, 1, 1], [1, 0, 1], [1, 1, 1]]),
    "Output: [[1,0,1],[0,0,0],[1,0,1]]",
)
print(
    solution().set_matrix_zeroes(matrix=[[0, 1, 2, 0], [3, 4, 5, 2], [1, 3, 1, 5]]),
    "Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]",
)
