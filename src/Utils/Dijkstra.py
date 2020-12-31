# Dijkstra's single source shortest path algorithm
# reference from this link: https://github.com/keon/algorithms
from .reader import *
import numpy as np
from numpy import linalg as LA


# class Dijkstra():
#     def __init__(self, vertices, g, use_g=False):
#         self.vertices = vertices
#         self.graph = g
#         if g is []:
#             raise Exception("Input g should not be None.")
#
#     def min_distance(self, dist, min_dist_set):
#         min_dist = float("inf")
#         for v in range(self.vertices):
#             if dist[v] < min_dist and min_dist_set[v] == False:
#                 min_dist = dist[v]
#                 min_index = v
#         return min_index
#
#     def dijkstra(self, src):
#         dist = [float("inf")] * self.vertices
#         dist[src] = 0
#         min_dist_set = [False] * self.vertices
#         for count in range(self.vertices):
#             # minimum distance vertex that is not processed
#             u = self.min_distance(dist, min_dist_set)
#             # put minimum distance vertex in shortest tree
#             min_dist_set[u] = True
#             # Update dist value of the adjacent vertices
#             for v in range(self.vertices):
#                 if self.graph[u][v] > 0 and min_dist_set[v] == False and dist[v] > dist[u] + self.graph[u][v]:
#                     dist[v] = dist[u] + self.graph[u][v]
#         return dist

class Dijkstra():
    def __init__(self, vertices, use_g=False):
        self.vertices = vertices
        # self.graph = g

    def min_distance(self, dist, min_dist_set):
        min_dist = float("inf")
        for v in range(self.vertices):
            if dist[v] < min_dist and min_dist_set[v] == False:
                min_dist = dist[v]
                min_index = v
        return min_index

    def dijkstra(self, src, graph):
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
                if graph[u][v] > 0 and min_dist_set[v] == False and dist[v] > dist[u] + graph[u][v]:
                    dist[v] = dist[u] + graph[u][v]
        return dist
