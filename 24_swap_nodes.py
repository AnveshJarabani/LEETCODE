"""
swap nodes without modifying values - 
take the node and the next node 
just point the next node to the previous node. 
now the previs node should point to the next node. 
done. 
"""


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def swap_nodes(head):
    dummy = ListNode()
    dummy.next = head
    prev = dummy
    while prev.next and prev.next.next:
        node1 = prev.next
        node2 = prev.next.next
        prev.next, node1.next, node2.next = node2, node2.next, node1
    return dummy.next
