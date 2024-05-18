"""
Adding two numbers that are linked list reverse orders - 

Since the numbers are reversed in the list
I just need to add each node values and 
carry -> send over to the next node as value. 
current sum // 10 -> ( to be kept in the current node value)
"""


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def read_list(self):
        nodes = []
        node = self
        while node:
            nodes.append(str(node.val))
            node = node.next
        return "->".join(nodes)


def addTwoNumbers(l1: ListNode, l2: ListNode) -> ListNode:
    head = ListNode(0)
    current = head
    carry = 0
    while l1 or l2 or carry:
        sum = 0
        if l1:
            sum += l1.val
            l1 = l1.next
        if l2:
            sum += l2.val
            l2 = l2.next
        sum += carry
        carry = sum // 10
        current.next = ListNode(sum % 10)
        current = current.next
    return head.next


def build_listnode(l: list) -> ListNode:
    head = ListNode()
    current = head
    for i in l:
        current.next = ListNode(val=i)
        current = current.next
    return head.next


l1 = build_listnode([1, 2, 3, 4])
l2 = build_listnode([5, 6, 7, 8, 9])
l3 = addTwoNumbers(l1, l2)
print(ListNode.read_list(l1))
print(ListNode.read_list(l2))
print(ListNode.read_list(l3))
