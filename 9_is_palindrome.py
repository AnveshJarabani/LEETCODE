"""
check if a number is palnidromic - 
If it has a negative -> return False
Else ->
str(input)[:length//2]==str(input)[l//2 + 1 ::-1]
"""


def is_palindrome(i: int) -> bool:
    if i < 0:
        return False
    print(str(i)[: len(str(i)) // 2], str(i)[len(str(i)) // 2 + 1 :][::-1])
    if len(str(i)) % 2 != 0:
        return str(i)[: len(str(i)) // 2] == str(i)[len(str(i)) // 2 + 1 :][::-1]
    return str(i)[: len(str(i)) // 2] == str(i)[len(str(i)) // 2 :][::-1]


i = 121
print(is_palindrome(i))
