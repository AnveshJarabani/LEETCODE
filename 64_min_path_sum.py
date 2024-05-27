"""
Given a m x n grid filled with non-negative numbers,
find a path from top left to bottom right,
which minimizes the sum of all numbers along its path.
Note: You can only move either down or right at any point in time.
Example 1:
Input: grid = [[1,3,1],[1,5,1],[4,2,1]]
Output: 7
Explanation: Because the path 1 → 3 → 1 → 1 → 1 minimizes the sum.
Example 2:
Input: grid = [[1,2,3],[4,5,6]]
Output: 12
Approach - 
Use backtracking. 
Traverse all paths from top left to the bottom right. 
"""


## BRUTE FORCE SOLUTION
def minimum_path_sum(grid: list[list[int]]) -> int:
    m, n = len(grid), len(grid[0])
    min_path_sum = float("inf")

    def backtrack(i, j, path_sum):
        nonlocal min_path_sum
        if i >= m or j >= n:
            return
        path_sum += grid[i][j]
        print(i, j, grid)
        if i == m - 1 and j == n - 1:
            min_path_sum = min(path_sum, min_path_sum)
            return
        backtrack(i + 1, j, path_sum)
        backtrack(i, j + 1, path_sum)

    backtrack(0, 0, 0)
    return min_path_sum


## OPTIMAL SOLUTION - DYNAMIC PROGRAMMING -
def minimum_path_sum(grid: list[list[int]]) -> int:
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
    return dp[-1][-1]


print(minimum_path_sum(grid=[[1, 3, 1], [1, 5, 1], [4, 2, 1]]), "Output: 7")
print(minimum_path_sum(grid=[[1, 2, 3], [4, 5, 6]]), "Output: 12")
