"""
rotate list - 
Given the head of a linked list, rotate the list to the right by k places.
Example 1:
Input: head = [1,2,3,4,5], k = 2
Output: [4,5,1,2,3]
Example 2:
Input: head = [0,1,2], k = 4
Output: [2,0,1]
"""


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


from typing import Optional


class Solution:
    def rotate_list(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head: 
            return None
        # Calculate the length of the node list
        length = 1
        cur_node = ListNode()
        cur_node.next = head
        while cur_node.next:
            cur_node = cur_node.next
            length += 1
        # now the cur_node is at the tail of the listnode
        k %= length
        cur_node.next = head #! attaching the tail to head here
        for _ in range(length - k - 1):
            cur_node = cur_node.next
        result = cur_node.next
        cur_node.next = None
        return result
        
