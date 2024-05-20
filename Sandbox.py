"""
Solve sudoku board - 
1. Create a function to check if the current number in consideration
is a valid option to specify in the sudoku board. 
If it is , then we enter it. 
We do this recursively for each element of "123456789"
If at any point it's a false, We return False, else keep going till the last value. 
"""


def solve_sudoku(board: list[list[str]]) -> list[list[str]]:
    # Find if the number can be used at the current position in the board
    def is_valid(board, row, col, num):
        for x in range(9):
            if board[row][x] == num:
                return False
            if board[x][col] == num:
                return False
        start_row, start_col = row - row % 3, col - col % 3
        for x in range(3):
            for y in range(3):
                if board[start_row + x][start_col + y] == num:
                    return False
        return True
    def solve(board):
        for row in range(9):
            for col in range(9):
                if board[row][col] == '.':
                    s = '123456789'
                    for num in s:
                        if is_valid(board,row,col,num):
                            board[row][col] = num
                            if solve(board):
                                return True
                        board[row][col] = '.'
                    return False
        return True
    
    if not board: return None
    if solve(board): return board
                    

        


board = [
    ["5", "3", ".", ".", "7", ".", ".", ".", "."],
    ["6", ".", ".", "1", "9", "5", ".", ".", "."],
    [".", "9", "8", ".", ".", ".", ".", "6", "."],
    ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
    ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
    ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
    [".", "6", ".", ".", ".", ".", "2", "8", "."],
    [".", ".", ".", "4", "1", "9", ".", ".", "5"],
    [".", ".", ".", ".", "8", ".", ".", "7", "9"],
]
test_result = [
    ["5", "3", "4", "6", "7", "8", "9", "1", "2"],
    ["6", "7", "2", "1", "9", "5", "3", "4", "8"],
    ["1", "9", "8", "3", "4", "2", "5", "6", "7"],
    ["8", "5", "9", "7", "6", "1", "4", "2", "3"],
    ["4", "2", "6", "8", "5", "3", "7", "9", "1"],
    ["7", "1", "3", "9", "2", "4", "8", "5", "6"],
    ["9", "6", "1", "5", "3", "7", "2", "8", "4"],
    ["2", "8", "7", "4", "1", "9", "6", "3", "5"],
    ["3", "4", "5", "2", "8", "6", "1", "7", "9"],
]
print(solve_sudoku(board))
print(solve_sudoku(board) == test_result)
