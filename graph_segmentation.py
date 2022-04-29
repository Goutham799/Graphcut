import cv2
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
import math
from alive_progress import alive_bar
import pickle


class Graph:

	def __init__(self,graph):
		self.graph = graph # residual graph
		self.org_graph = [i[:] for i in graph]
		self. ROW = len(graph)
		self.COL = len(graph[0])


	# updating resudial graph
	def BFS(self,s, t, parent):

		# Mark all the vertices as not visited
		visited =[False]*(self.ROW)

		# Create a queue for BFS
		queue=[]

		# Mark the source node as visited and enqueue it
		queue.append(s)
		visited[s] = True

		# Standard BFS Loop
		while queue:

			#Dequeue a vertex from queue and print it
			u = queue.pop(0)

			# marking all unvisited adjacent vertexes of u
			for ind, val in enumerate(self.graph[u]):
				if visited[ind] == False and val > 0 :
					queue.append(ind)
					visited[ind] = True
					parent[ind] = u

		# returning true if sink is reached
		return True if visited[t] else False
		
	
	# Depth FIrst Traversal of the graph
	def dfs(self, graph,s,visited):
		visited[s]=True
		for i in range(len(graph)):
			if graph[s][i]>0 and not visited[i]:
				self.dfs(graph,i,visited)

	# Returns the min-cut of the given graph
	def ford_fulkerson(self, source, sink):

		# This array is filled by BFS and to store path
		parent = [-1]*(self.ROW)

		max_flow = 0 # There is no flow initially

		# Augment the flow while there is path from source to sink
		while self.BFS(source, sink, parent) :

			# Find minimum residual capacity of the edges along the
			# path filled by BFS. Or we can say find the maximum flow
			# through the path found.
			path_flow = float("Inf")
			s = sink
			while(s != source):
				path_flow = min (path_flow, self.graph[parent[s]][s])
				s = parent[s]

			# Add path flow to overall flow
			max_flow += path_flow

			# update residual capacities of the edges and reverse edges
			# along the path
			v = sink
			while(v != source):
				u = parent[v]
				self.graph[u][v] -= path_flow
				self.graph[v][u] += path_flow
				v = parent[v]

		visited=len(self.graph)*[False]
		self.dfs(self.graph,s,visited)

		# print the edges which initially had weights
		# but now have 0 weight
		for i in range(self.ROW):
			for j in range(self.COL):
				if self.graph[i][j] == 0 and\
				self.org_graph[i][j] > 0 and visited[i]:
					print(str(i) + " - " + str(j))




class Loader:
	def __init__(self, path):
		
		self.img = cv2.imread(path)
		img_grey = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
		self.img_normalised = cv2.normalize(img_grey, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		self.dim = self.img_normalised.shape

		
		self.c_mat = [[0] * self.dim[0] * self.dim[1]] * (self.dim[0] * self.dim[1] + 2)

	def inter_pixel_weight(self, vertex_from, vertex_to, sigma = 30):

		intensity_1 = self.img_normalised[vertex_from[0]][vertex_from[1]]
		intensity_2 = self.img_normalised[vertex_to[0]][vertex_to[1]]
		
		# connecting to nearby pixel only therefore considering 
		# eclud distance = 1
		# energy function simplified is as follows
		return int(math.exp(-abs(intensity_1 - intensity_2) / (2 * sigma^2)))

	#mapping image cordinates to c_mat x & y cordinate
	def map_with_c_mat(self,vertex_from, vertex_to):
		
		edge = (
			(vertex_from[0] * self.dim[1]) + vertex_from[1] + 1,
			(vertex_to[0] * self.dim[1]) + vertex_to[1]
		)
		
		return edge
		
	def build_c_mat(self, bbox):

		print('Calculating inter pixel egde weights....')
		with alive_bar(self.dim[0] - 1) as bar:
			for i in range(self.dim[0] - 1):
				for j in range(self.dim[1] - 1):
					
					pos = self.map_with_c_mat((i, j), (i, j + 1))
					self.c_mat[pos[0]][pos[1]] = self.inter_pixel_weight(
						(i,j), (i, j + 1)
					)

					pos = self.map_with_c_mat((i, j), (i + 1, j))
					self.c_mat[pos[0]][pos[1]] = self.inter_pixel_weight(
						(i,j), (i + 1, j)
					)

				bar()					

		print('Please Wait....')

		source_index = 0
		sink_index = len(self.c_mat) - 1
		# initializing all weights to sink as maximum and will update
		# weights in bounding box to lowest
		for i in range(len(self.c_mat[0])):
			self.c_mat[sink_index][i] = 1

		print('Calculating super nodes weights....')
		with alive_bar(
			(bbox.end_x - bbox.start_x) * (bbox.end_y - bbox.start_y)
		) as bar:
			for i in range(bbox.start_x, bbox.end_x + 1):
				for j in range(bbox.start_y, bbox.end_y + 1):
					pos = self.map_with_c_mat((0, 0), (i, j))
					
					#assigning all pixels in bbox with heighest weight to source 
					try:
						self.c_mat[source_index][pos[1]] = 1
						
						#assigning all pixels in bbox with kowest weight to sink
						self.c_mat[sink_index][pos[1]] = 0
					except:
						pass

					bar()

		return True

class Segment:
	def __init__(self, loader, rect):
		self.loader = loader
		self.rect = rect

		mask = np.zeros(self.loader.img.shape[:2], np.uint8)
		bg_model = np.zeros((1, 65), np.float64)
		fg_model = np.zeros((1, 65), np.float64)
		cv2.grabCut(loader.img, mask, (rect.start_x, rect.start_y, rect.end_x, rect.end_y), bg_model, fg_model, 5, cv2.GC_INIT_WITH_RECT)


		self.mask2 = np.where((mask == 2)| (mask == 0), 0, 1).astype('uint8')
		self.mask3 = np.where((mask == 2)| (mask == 0), 1, 0).astype('uint8')
		
		# img = self.loader.img * self.mask2[:,:, np.newaxis] + target_img * mask3[:,:, np.newaxis]
		# img = loader.img * self.mask2[:,:, np.newaxis]
		
		# plt.subplot(122)
		# plt.title("Segmented Image")

		# plt.imshow(img)
		# plt.subplot(121)
		
		# cv2.rectangle(loader.img, (rect.start_x, rect.start_y), (rect.end_x, rect.end_y), (255,255,255), 2)
		
		# plt.imshow(loader.img)
		# plt.title("Orignal Image")
		# plt.show()


	def show(self):
		img = self.loader.img * self.mask2[:,:, np.newaxis]
		
		plt.subplot(122)
		plt.title("Segmented Image")

		plt.imshow(img)
		plt.subplot(121)
		
		cv2.rectangle(self.loader.img, (self.rect.start_x, self.rect.start_y), (self.rect.end_x, self.rect.end_y), (255,255,255), 2)
		
		plt.imshow(self.loader.img)
		plt.title("Orignal Image")
		plt.show()

	def stich_to(self, target):
		
		target_img = cv2.imread(target)
		
		if target_img.shape == self.loader.img.shape:

			img = self.loader.img * self.mask2[:,:, np.newaxis] + target_img * self.mask3[:,:, np.newaxis]
			plt.subplot(122)
			plt.title("Segmented Image")

			plt.imshow(img)
			plt.subplot(121)
			
			cv2.rectangle(self.loader.img, (self.rect.start_x, self.rect.start_y), (self.rect.end_x, self.rect.end_y), (255,255,255), 2)
			
			plt.imshow(self.loader.img)
			plt.title("Orignal Image")
			plt.show()

		else:
			print("Please use source image and target image of same dimention")


class BoundingBox:
	def __init__(self, start_x, start_y, end_x, end_y):
		self.start_x = start_x
		self.start_y = start_y
		self.end_x = end_x
		self.end_y = end_y


graph = [[0, 16, 13, 0, 0, 0],
		[0, 0, 10, 12, 0, 0],
		[0, 4, 0, 0, 14, 0],
		[0, 0, 9, 0, 0, 20],
		[0, 0, 0, 7, 0, 4],
		[0, 0, 0, 0, 0, 0]]

print("Running ford fulkerson on a small sample graph")
print(graph)

g = Graph(graph)

source = 0; sink = 5

print("Following is the min cut generated by ford fulkerson algorithm")
g.ford_fulkerson(source, sink)

source_img = 's4.jpeg'
target_img = 's3.jpeg'

#for hut and hill images
bbox = BoundingBox(100, 30, 250, 180)

#for hand and mobile phone image
#bbox = BoundingBox(300, 120, 400, 250)

print("bounding box cordinates are", bbox)
print("Loading source image")



l = Loader(source_img)

print('Building Contingency Matrix')
l.build_c_mat(bbox)

print('Saving contingency matrix inn a pickle file')
with open('contingency', 'wb') as fp:
    pickle.dump(l.c_mat, fp)


# print('saving c mat in text file')
# with open('c_mat.txt', 'w') as f:
    
# 	with alive_bar(len(l.c_mat)) as bar:
		
# 		for item in l.c_mat[1:10]:
# 			f.write("%s\n" % item)
# 			bar()



segment = Segment(l, bbox)
segment.stich_to(target_img)
segment.show()



