# Dijkstra's single source shortest path algorithm
# reference from this link: https://github.com/keon/algorithms
from reader import *
import numpy as np
from numpy import linalg as LA

class Dijkstra():
    def __init__(self, vertices, g, use_g=False):
        self.vertices = vertices
        self.graph = [[0 for column in range(vertices)] for row in range(vertices)]
        if g is not []:
            self.graph = g

    def set_graph(self, g):
        self.graph = g

    def min_distance(self, dist, min_dist_set):
        min_dist = float("inf")
        for v in range(self.vertices):
            if dist[v] < min_dist and min_dist_set[v] == False:
                min_dist = dist[v]
                min_index = v
        return min_index

    def dijkstra(self, src):
        dist = [float("inf")] * self.vertices
        dist[src] = 0
        min_dist_set = [False] * self.vertices
        for count in range(self.vertices):
            # minimum distance vertex that is not processed
            u = self.min_distance(dist, min_dist_set)
            # put minimum distance vertex in shortest tree
            min_dist_set[u] = True
            # Update dist value of the adjacent vertices
            for v in range(self.vertices):
                if self.graph[u][v] > 0 and min_dist_set[v] == False and dist[v] > dist[u] + self.graph[u][v]:
                    dist[v] = dist[u] + self.graph[u][v]
        return dist

    def dijkstra2node(self, src, dist):
        distlist = self.dijkstra(src)
        return distlist[dist]


if __name__ == '__main__':
    mesh, _, _, _ = read(int(1))  # build graph
    init_pos = mesh.vertices.astype(np.float32)
    init_pos = init_pos[:, :2]
    map = np.zeros((mesh.num_vertices, mesh.num_vertices))
    mesh.enable_connectivity()
    for p in range(mesh.num_vertices):
        adjv = mesh.get_vertex_adjacent_vertices(p)
        for j in range(adjv.shape[0]):
            n1 = p
            n2 = adjv[j]
            p1 = init_pos[n1, :]
            p2 = init_pos[n2, :]
            dp = LA.norm(p1-p2)
            map[n1][n2] = map[n2][n1] = dp
    maplist = map.tolist()
    print(map)
    print(maplist)
    # g = Dijkstra(9, [], False)
    # g.graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
    #            [4, 0, 8, 0, 0, 0, 0, 11, 0],
    #            [0, 8, 0, 7, 0, 4, 0, 0, 2],
    #            [0, 0, 7, 0, 9, 14, 0, 0, 0],
    #            [0, 0, 0, 9, 0, 10, 0, 0, 0],
    #            [0, 0, 4, 14, 10, 0, 2, 0, 0],
    #            [0, 0, 0, 0, 0, 2, 0, 1, 6],
    #            [8, 11, 0, 0, 0, 0, 1, 0, 7],
    #            [0, 0, 2, 0, 0, 0, 6, 7, 0]]
    # print(g.dijkstra(0))
    g2 = Dijkstra(mesh.num_vertices, maplist, False)
    dlist = g2.dijkstra(0)
    darray = np.asarray(dlist)
    print("2:  ", g2.dijkstra2node(0, 2))
    print("158:  ", g2.dijkstra2node(0, 158))
    # print(darray)


