"""
valid paranthesis - 
Approach - 
stack. 
if the next value matches, remove it. 
if not, keep going until you hit the match and remove it. 
at the end the stack should be empty. 

"""

pairs = {")": "(", "}": "{", "]": "["}


class Solution:
    def valid_paranthesis(self, s: str) -> bool:
        stack = []
        for i in s:
            if stack and stack[-1] == pairs.get(i):
                stack.pop()
            else:
                stack.append(i)
        return len(stack) == 0


s = "()[]{}"
print(Solution().valid_paranthesis(s))
