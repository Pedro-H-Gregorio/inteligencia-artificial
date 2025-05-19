class Node:
    """
    Representa um nó em uma árvore binária de busca.
    Cada nó possui um valor (key) e referências para subárvores esquerda e direita.
    """
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None