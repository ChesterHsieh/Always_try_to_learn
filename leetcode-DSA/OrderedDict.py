class Node:
    def __init__(self,key, value, prev= None, next= None) -> None:
        self.key = key
        self.value = value
        self.prev = prev
        self.next = next

class OrderedDict:
    def __init__(self) -> None:
        self.head = Node(None,None)
        self.tail = Node(None,None)
        self.mem = {}
    
    def move_to_end(key):
        n = self.mem[key]
        n.prev.next, n.next.prev  = n.next, n.prev
        self.tail.prev.next, n.next, n.prev, self.tail.prev = n, self.tail, self.tail.prev, n

    def popleft():
        n = self.head.next 
        del self.mem[n.key]
        self.head, n.next.prev = n.next, self.head
    
    def __setitem__(self,key,val):
        self.mem[key] = Node(key,val)

    

class LRUCache:
    def __init__(self,cap) -> None:
        self.d = OrderedDict()
        self.cap = cap

    def get(key) -> int:
        if key in self.d.mem:
            self.d.move_to_end(key)
            return self.d.mem[key].val
        else:
            return -1

    def put(key,value) -> None:
        if key in self.d.mem:
            self.d.move_to_end(key)
            return self.d.mem[key].value
        else:
            if len(self.d.mem) <= self.cap:
                self.d[key] = val


if __name__ == "__main__":
