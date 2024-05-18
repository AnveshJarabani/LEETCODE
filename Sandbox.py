"""
Adding two numbers that are linked list reverse orders - 

Since the numbers are reversed in the list
I just need to add each node values and 
carry -> send over to the next node as value. 
current sum // 10 -> ( to be kept in the current node value)
"""


class ListNode:
    def __init__(self,val=0,next=None):
        self.val=val
        self.next=next

def addTwoNumbers(l1:ListNode,l2:ListNode) -> ListNode:
    head = ListNode(0)
    current = head
    carry = 0
    while l1 or l2 or carry:
        sum = l1.val + l2.val
        l1 , l2 = l1.next , l2.next
        
        
        
    

