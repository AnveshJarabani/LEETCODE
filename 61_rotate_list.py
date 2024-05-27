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
        # Calculate the length of the node list
        cur_node = ListNode()
        cur_node.next = head
        length = 1
        while cur_node.next:
            cur_node = cur_node.next
            length += 1
        k %= length
        cur_node.next = head
        cur_node = cur_node.next
        for _ in range(length - k - 1):
            cur_node.next = cur_node
        result = cur_node.next
        cur_node.next = None
        return result
