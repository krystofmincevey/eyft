class BoundedList:
    """
    BoundedList is a list-like data structure with a fixed maximum length.

    When the maximum length is exceeded upon calling append(),
    the first added element is removed.

    Attributes:
        max_len: The maximum number of elements that BoundedList can hold.
    """
    def __init__(self, max_len):
        """
        Initializes a BoundedList with a given maximum length.
        """
        self.max_len = max_len
        self._list = []

    def append(self, item):
        """
        Adds an item to the end of the BoundedList. If this would cause the
        length to exceed max_len, the first item is removed.
        """
        if len(self._list) == self.max_len:
            self._list.pop(0)
        self._list.append(item)

    def pop(self, index: int = 0):
        return self._list.pop(index)

    def __getitem__(self, index):
        """
        Returns the item at a given index.
        """
        return self._list[index]

    def __setitem__(self, index, value):
        """
        Sets the item at a given index.
        """
        self._list[index] = value

    def __len__(self):
        """
        Returns the current length of the BoundedList.
        """
        return len(self._list)

    def __str__(self):
        """
        Returns a string representation of the BoundedList.
        """
        return str(self._list)

    def __repr__(self):
        """
        Returns a formal string representation of the BoundedList.
        """
        return f'BoundedList({self._list}, maxlen={self.max_len})'
