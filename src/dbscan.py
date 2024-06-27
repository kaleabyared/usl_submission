import random

import numpy as np


class Point(object):
    def __init__(self, coordinate):
        self.coordinate = coordinate
        self.cluster_idx = None
        
    def is_clustered(self):
        return self.cluster_idx is not None
    
    def cluster(self, cluster_idx):
        self.cluster_idx = cluster_idx


class DBSCAN(object):
    def __init__(self, epsi, min_points):
        self.epsi = epsi
        self.min_points = min_points
    
    @staticmethod
    def _compute_distance(x1, x2):
        return np.sum((x1 - x2)**2)**0.5

    def _find_neighbor_indices(self, core_point, other_points):
        neighbor_indices = []
        
        for idx in range(len(other_points)):
            if self._compute_distance(core_point.coordinate, other_points[idx].coordinate) <= self.epsi:
                neighbor_indices.append(idx)
                
        return neighbor_indices

    def fit(self, df, feature_cols=['Annual Income (k$)', 'Spending Score (1-100)']):
        points = []
        for idx, row in df.loc[:, feature_cols].iterrows():
            points.append(Point(row.to_numpy()))
                    
        free_point_indices = set(range(len(points)))

        core_point_indices = set()
        cluster_count = 0

        while len(free_point_indices) > 0:
            if len(core_point_indices) == 0:
                start_idx = random.choice(list(free_point_indices))
                core_point_indices.add(start_idx)
                free_point_indices.remove(start_idx)
                points[start_idx].cluster(cluster_count)

            while len(core_point_indices) > 0:
                picked_core_point_idx = core_point_indices.pop()
                
                neighbor_indices = self._find_neighbor_indices(points[picked_core_point_idx], points)
                
                if len(neighbor_indices) >= (self.min_points - 1):
                    for neighbor_idx in neighbor_indices:
                        if neighbor_idx in free_point_indices:
                            points[neighbor_idx].cluster(cluster_count)
                            if neighbor_idx in free_point_indices:
                                core_point_indices.add(neighbor_idx)
                                free_point_indices.remove(neighbor_idx)
                    
            cluster_count += 1
                    
        return points