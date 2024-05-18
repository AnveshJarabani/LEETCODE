"""
Longest valid paranthesis - 
Example 1:
Input: s = "(()"
Output: 2
Explanation: The longest valid parentheses substring is "()".
Example 2:
Input: s = ")()())"
Output: 4
Explanation: The longest valid parentheses substring is "()()".
Example 3:
Input: s = ""
Output: 0

Approach - 
use a stack to keep track of the valid paranthesis with the indexes of each value in the stack.
start with stack = [-1] or any non negative number.
Now, as we travers through the string with the index numbers, 
if the value of the char is '(', then we add the index to the stack
If it's a ')' then we pop from the stack. 
so once the max valid paranthesis breaks so to speak,
we just find the delta of current i to the last i on the stack.
Which gives us the max span. Then we keep track of max span by comparing with max(current,past)
"""


def max_paranthesis(s: str) -> int:
    stack = [-1]
    max_result = 0
    for i in range(len(s)):
        if s[i] == "(":
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                max_result = max(max_result, i - stack[-1])
    return max_result


s = "(()(((((())))))"
print(max_paranthesis(s))
