from node import Node

ACTIONS = {"left": 1, "right": 2}

def build_graph(init: int, dest: int, root: Node = None):
    root = Node(init)
    tail = Node(dest)
    build_graph_recursive(root, tail)
    return root

def build_graph_recursive(inter, tail, nodes: list = [], inserted_values: list = []):
    if inter.key == tail.key:
        return inter
    inter.left = Node(inter.key + ACTIONS["left"])
    inter.right = Node(inter.key * ACTIONS["right"])
    if inter.left.key not in inserted_values:
        nodes.append(inter.left)
        inserted_values.append(inter.left.key)
    if inter.right.key not in inserted_values:
        nodes.append(inter.right)
        inserted_values.append(inter.right.key)
    return build_graph_recursive(nodes.pop(0), tail, nodes, inserted_values)


def init_bfs(graph, dest):
    node = graph
    return bfs(node, dest)
    

def bfs(root: Node, dest: int,found_nodes: list = [], visted_nodes: list = []):
    if root.left:
        found_nodes.append(root.left)
    if root.right:
        found_nodes.append(root.right)
    visted_nodes.append(root)
    if len(found_nodes) == 0 or root.key == dest:
        return visted_nodes
    return bfs(found_nodes.pop(0), dest, found_nodes, visted_nodes)

def count_nodes(root: Node, count: int = 0, nodes_found: list = []):
    if root.left:
        nodes_found.append(root.left)
    if root.right:
        nodes_found.append(root.right)
    count += 1
    if len(nodes_found) == 0:
        return count
    return count_nodes(nodes_found.pop(0), count)
    


print("Escolha um numero para iniciar a busca:")
init = int(input("Numero: "))
print("Escolha o numero da direita do numero inicial para o destino: ")
dest = int(input("Numero: "))

graph = build_graph(init, dest)
nodes_visited = bfs(graph, int(dest))
len_graph = count_nodes(graph)

print("Caminhos percorridos: ", end=" ")
[print(el.key, end=" ") for el in nodes_visited]
print()
print("Quantidade de nodos acessados: ", len(nodes_visited))
print("Posibilidades produzidas pelas ações: ", len_graph)
print("Custo associado: ", len(nodes_visited))