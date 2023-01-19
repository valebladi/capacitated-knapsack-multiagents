import sys
import math	
import numpy as np
from  collections.abc import Iterable	#flatten lists
import os								# access operationg system dependent file paths
import matplotlib.pyplot as plt					# plotting solution process and graphs

''' 
Dynamic Knapsack Problem, returns the maximum value that can be put in a knapsack of capacity W given weight and value
'''

# part 0: settings
#filename = 'moabit87_2'						# select vrp file
filename = 'moabit214_2'						# select vrp file
#filename = 'monbijou-james-simon_2'				# select vrp file
show_gui = True
# part 1: MURMEL energy and time
## part 1.1: murmel energy
speed_murmel= 3.24							# time per distance in h per km (MURMEL) based on 0.9 m/s maximum speed in Urbanek bachelor thesis
energy_murmel_loc = 0.122						# energy consumption mobility of MURMEL in KWh per kmilometer
energy_murmel_bin = 0.0185						# energy consumption per bin in KWh
murmel_capacity = 50							# on average, a mothership visit is necessary every 'murmel_capacity's waypoint
## part 1.2: murmel time compression
time_adjust_bin = 120/3600						# seconds-hr, whole process of opening and closing trash can 		
time_compress  = 300/3600 						# seconds-hr
#time_emptying = 42.5/3600						# time to empty a dustbin in hr (MURMEL) based on 42.5s in Merle Simulation

# part 2: MOTHERSHIP energy and time
speed_mothership = 1/50							# time per distance in h per km (mothership)
energy_mothership_loc = 0.27					# energy consumption kWh per km of MOTHERSHIP

## part 3: MOTHERSHIP and MM battery swap and unload time
## part 3.1: battery swap
time_swapping_battery = 240/3600				# time consumption per battery swap between MS and MM 
## part 3.2: unloading trash
time_unloading_trash = 30/3600					# time consumption per complete unload between MM and MS

# part 4: battery capacity
battery_capacity = 0.96							# max battery capacity in kWh
# part 5: constant energy
constant_energy = 0.092							# constant energy consumption given CPU, sensor, etc

## part 0.1: global vairables 
nodes = []
value = []
dist = []
num_visited = [0,0]
num_visited_cap = [[0]]
fsize=10
params = {'legend.fontsize': fsize*0.8,
          'axes.labelsize': fsize*0.9,
          'axes.titlesize': fsize,
          'xtick.labelsize': fsize*0.8,
          'ytick.labelsize': fsize*0.8,
          'axes.titlepad': fsize*1.5}
plt.rcParams.update(params)

### part 1: open file and get waypoints
def file_oppening(filename):
	try:
		with open(filename + '.tsp', 'r') as tsp_file:
			tsp_file_data = tsp_file.readlines()
	except Exception as e:
		print('error!\nExiting..') # more exception details: str(e)
		sys.exit()
	# possible entries in specification part:
	specification_list = ['NAME', 'TYPE', 'COMMENT', 'DIMENSION', 'CAPACITY', 'GRAPH_TYPE', 'EDGE_TYPE', 'EDGE_WEIGHT_TYPE', 'EDGE_WEIGHT_FORMAT', 'EDGE_DATA_FORMAT', 'NODE_TYPE', 'NODE_COORD_TYPE', 'COORD1_OFFSET', 'COORD1_SCALE', 'COORD2_OFFSET', 'COORD2_SCALE', 'COORD3_OFFSET', 'COORD3_SCALE', 'DISPLAY_DATA_TYPE']
	specification = [None] * len(specification_list)
	node_data = False
	for data_line in tsp_file_data:
		data_line = data_line.replace('\n', '')
		if node_data:
			node = data_line.split()
			if len(node) == 4:
				try:
					node[0], node[1], node[2] ,node[3] = int(node[0]), float(node[1]), float(node[2]),int(node[3])
					nodes.append(node)
				except Exception as e: # not expected data format; try to continue parsing
					node_data = False
			else:
				node_data = False
		for i in range(len(specification_list)):
			if data_line.find(specification_list[i] + ': ') == 0:
				specification[i] = data_line.replace(specification_list[i] + ': ', '')
		if (data_line.find('NODE_COORD_SECTION') == 0):
			node_data = True

def flatten(x):
    if isinstance(x, Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]
		
def geographic_2d_distance(i, j,nodes_coor):
	R = 6371  # Earth radius in kilometers
	dLat = math.radians(nodes_coor[j,1] - nodes_coor[i,1])
	dLon = math.radians(nodes_coor[j,0] - nodes_coor[i,0])
	lat1 = math.radians(nodes_coor[i,1])
	lat2 = math.radians(nodes_coor[j,1])
	return 2 * R * math.asin(math.sqrt(math.sin(dLat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2)**2)) #converted in kilometers

# part 1.2: MM energy and time cost when moving from one node to another
def cost_murmel_distance(dist):
	energy_cost = 0
	time_cost = 0
	energy_cost = dist*energy_murmel_loc
	time_cost = dist/speed_murmel
	return (energy_cost, time_cost)

# part 1.3: MM energy and time cost when compresing garbage 
def cost_murmel_compressing(bins_empied):
	energy_cost = 0
	time_cost = 0
	energy_cost = energy_murmel_bin*bins_empied
	time_cost = (time_adjust_bin + time_compress+time_unloading_trash)*bins_empied
	return (energy_cost,time_cost)

# part 1.4: MS energy and time cost
def cost_mothership(dist):
	# added energy and time of passing the trash from MM to MS
	energy_cost = dist*energy_mothership_loc
	time_cost = dist*speed_mothership
	return (energy_cost,time_cost) 

# part 1.5: battery energy and time cost
def cost_battery(energy_cost,time_cost):
	battery_changes = math.ceil(energy_cost/battery_capacity)
	energy_cost = battery_changes
	time_cost = time_swapping_battery*battery_changes
	return (battery_changes,energy_cost,time_cost)

# part 2.1: update listed after nodes are visited
def update(t_val, t_wt, t_coor,t_nod,t_point):
	l_val = []
	l_wt = []
	l_coor = []
	l_point = []
	#flatten the irregular list 
	t_nod = flatten(t_nod)
	#remove the visited nodes, so next search is based on the unvisted nodes 
	for i in range (0,len(t_nod),2):
		if [t_nod[i],t_nod[i+1]] in t_coor:
			n = t_coor.index([t_nod[i],t_nod[i+1]])
			#have them in the visited lists
			l_val.append(t_val[n])
			l_wt.append(t_wt[n])
			l_coor.append(t_coor[n])
			l_point.append(t_point[n])
			#pop elements from the lists
			t_val.pop(n)
			t_wt.pop(n)
			t_coor.pop(n)
			t_point.pop(n)
	#sort coor from max to min value
	l_point = flatten(l_point)
	values_t = []
	if l_val and l_point:
		num_1 = []
		sort_val = np.argsort(l_val)
		for i in range (0,len(sort_val)):
			num_1.append([sort_val[i],l_point[i]])
		num_1= sorted(num_1, key=lambda x:x[0],reverse=True)
		a = []
		for num in num_1:
			a.append(int(num[1]))
			num_visited.append(int(num[1]))
		values_t = np.delete(value, num_visited, 1) # last arg colum 1 / row 0
	#return the value list of the visited node
	return (num_visited,values_t)

# part 2: coptimize path between value and cost
def knapSack(W, wt, val, n,coor, point_num):
	# build matrices for dynamic program
	K = [[0 for x in range(W + 1)] for x in range(n + 1)]
	coor_k = [[[0,0] for x in range(W + 1)] for x in range(n + 1)]
	# max capacity W given the weight and added value of the bins 
	for i in range(n+1):
		for w in range(W + 1):
			if i == 0 or w == 0:
				K[i][w] = 0
			elif wt[i-1] <= w:
				max_opt = (val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w])
				max_opt2 = ([coor[i-1],coor_k[i-1][w-wt[i-1]]], coor_k[i-1][w])
				K[i][w] = max(max_opt)
				coor_k[i][w]= max_opt2[max_opt.index(max(max_opt))]
			else:
				K[i][w] = K[i-1][w]
				coor_k[i][w]= (coor_k[i-1][w])
		n_point, values = update(val, wt, coor,coor_k[n][W],point_num)
	return (n_point,values) 

# part 3: calculate final cost and energy of selected pathfor MURMEL
def f_mm_route(points,capacity):
	# calculate the final cost energy, time and distance 
	final_route = []
	f_dist_cost = 0
	d_energy_cost_loc = 0
	d_time_cost_loc = 0
	t_nodes_coor = nodes_coor.tolist()
	cap_coor_2 = []
	# if SD did not exists
	t_cap = []
	temp_cap = 0
	temp_cap_2 = 0
	for i in points:
		final_route.append(t_nodes_coor[i])
		#if SD did not exists
		t_cap.append(capacity[i])
	for i in range (0,len(points)-1):
		f_dist_cost +=  (dist[points[i], points[i+1]])
		a = dist[points[i], points[i+1]]
		b,c = cost_murmel_distance(a)
		d_energy_cost_loc += b
		d_time_cost_loc += c
		#print (t_cap)
		#print (points[i], points[i+1])
		#if SD did not exists
		#print (t_cap[i], t_cap[i+1])
		temp_cap += t_cap[i]  + t_cap[i+1]
		if temp_cap>=100:
			num_visited_cap.append(cap_coor_2)
			cap_coor_2 =[]
			temp_cap_2 += 1
			temp_cap = 0
		cap_coor_2.append(points[i])
	print (cap_coor_2)
	#if SD did not exists
	#given that the at needs to be emptied 
	temp_cap_2 = temp_cap_2+1
	energy_cost_bin, time_cost_bin = cost_murmel_compressing(temp_cap_2)
	print (d_energy_cost_loc,energy_cost_bin,d_time_cost_loc,time_cost_bin)
	f_energy_cost_bins = d_energy_cost_loc+energy_cost_bin
	f_time_cost_bins = d_time_cost_loc+time_cost_bin
	battery_changes, f_energy_cost_battery, f_time_cost_battery = cost_battery(f_energy_cost_bins,f_time_cost_bins)
	return (temp_cap_2,f_energy_cost_bins,f_time_cost_bins,f_dist_cost,final_route, battery_changes,f_energy_cost_battery,f_time_cost_battery)

# part 4: calculate final cost and energy of selected pathfor Mothership
def f_ms_route(mm_f_rout_cap):
	f_dist_cost = 0
	d_energy_cost_loc = 0
	d_time_cost_loc = 0
	coor_ms = []
	dis_temp = 0
	num_visited_ms = []
	t_nodes_coor = nodes_coor.tolist()
	for point in mm_f_rout_cap:
		print (len(point))
		num_visited_ms.append(point[-1])
		coor_ms.append(t_nodes_coor[point[-1]])
	# calculate the final cost energy distance
	for i in range (0,len(num_visited_ms)-1):
		f_dist_cost +=  (dist[num_visited_ms[i], num_visited_ms[i+1]])
		a = dist[num_visited_ms[i], num_visited_ms[i+1]]
		b,c = cost_mothership(a)
		dis_temp += a
		d_energy_cost_loc += b
		d_time_cost_loc += c
	return (num_visited_ms,coor_ms, d_energy_cost_loc,d_time_cost_loc,dis_temp)

# part 5: draw path planning
def gui(f_route,f_route_ms):
	f_route = np.array(f_route)
	f_route_ms = np.array(f_route_ms)
	x1 = f_route[:,1] #lon
	y1 = f_route[:,0] #lat
	x2 = f_route_ms[:,1] #lon
	y2 = f_route_ms[:,0] #lat
	plt.xlabel("Longitud")
	plt.ylabel("Latitud")
	plt.scatter(x1, y1,color='blue')
	plt.plot(x1,y1,color='blue')
	plt.scatter(x2, y2,color='red')
	plt.plot(x2,y2,color='red')
	plt.show()

if __name__ == "__main__":
	file_oppening(filename)
	print('#Debug info: input file nodes part:')
	for node in nodes:
		print ('#   ' + str(node))
	# get nodes specifications in different arrays
	nodes_num = np.array(nodes)[:,[0]]
	nodes_coor = np.array(nodes)[:,[1,2]]
	t_nodes_bins_cap = np.array(nodes)[:,[3]].astype(int)
	nodes_bins_cap = np.full((len(t_nodes_bins_cap),1), 50, dtype=int)
	# generate cost and value functions for murmel
	dist = np.zeros((len(nodes_coor), len(nodes_coor)))
	value = np.zeros((len(nodes_coor), len(nodes_coor)))
	for i in range(len(nodes_coor)):
		for j in range(i):
			dist[i,j] = geographic_2d_distance(i,j,nodes_coor)
			dist[j,i] = dist[i,j]
			#value[i,j] = (100000/((100-nodes_bins_cap[i])+0.0001+dist[i,j]))						# Value function 1
			#value[j,i] = (100000/((100-nodes_bins_cap[j])+0.0001+dist[j,i]))						# Value function 1
			value[i,j] = ((nodes_bins_cap[i]+nodes_bins_cap[j])/sum(nodes_bins_cap))/dist[i,j]		# Value function 2
			value[j,i] = ((nodes_bins_cap[i]+nodes_bins_cap[j])/sum(nodes_bins_cap))/dist[j,i]		# Value function 2
			#value[i,j] = ((nodes_bins_cap[i]+nodes_bins_cap[j])/sum(nodes_bins_cap))/dist[i,j]**2	# Value function 3
			#value[j,i] = ((nodes_bins_cap[i]+nodes_bins_cap[j])/sum(nodes_bins_cap))/dist[j,i]**2 	# Value function 3
	# change from np.array to lists 
	nodes_num_2 = nodes_num.tolist()
	nodes_coor_2 = nodes_coor.tolist()
	t_nodes_bins_cap = flatten(t_nodes_bins_cap)
	nodes_bins_cap_2= flatten(nodes_bins_cap)
	# eliminate origin point of MURMEL
	nodes_coor_2 = nodes_coor_2[1:]
	nodes_num_2 = nodes_num_2[1:]
	nodes_bins_cap_2 = nodes_bins_cap_2[1:]
	# get the value function from initial point
	value_current = value[0].tolist()
	value_current = value_current[1:]
	'''
	print (nodes_coor_2)
	print (nodes_num_2)
	print (nodes_bins_cap_2)
	print (value)
	print (nodes_coor)
	input()
	'''
	# part 1: calculate the optimal path
	while nodes_coor_2: 
		n = len(nodes_bins_cap_2)
		#print ( murmel_capacity, nodes_bins_cap_2,value_current,n,nodes_coor_2, nodes_num_2)
		#print (murmel_capacity,len(nodes_bins_cap_2),len(value_current),n,len(nodes_coor_2),len(nodes_num_2))
		num_visited,c_values = knapSack(murmel_capacity, nodes_bins_cap_2,value_current,n,nodes_coor_2, nodes_num_2)
		#change value list to last point
		value_current = c_values[num_visited[-1]].tolist()
	#calculate final path on energy time and distance
	num_visited = [0, 1, 2, 9, 10, 11, 12, 13, 8, 7, 4, 3, 6, 5, 14, 15, 16, 17, 18, 20, 19, 21, 22, 23, 24, 27, 26, 29, 30, 28, 31, 32, 36, 37, 38, 39, 40, 41, 35, 33, 34, 42, 43, 46, 48, 49, 47, 50, 45, 44, 25]
	t_nodes_bins_cap = [0,35,68,2,94,46,97,0,43,58,51,64,82,39,94,31,62,37,54,30,30,15,69,65,55,21,90,73,89,79,70,96,46,11,31,30,18,46,16,11,28,91,71,16,44,100,83,99,43,16,8]
	#num_visited = [0, 86, 3, 1, 2, 4, 5, 18, 19, 20, 21, 17, 16, 10, 11, 13, 12, 14, 28, 50, 49, 27, 25, 26, 24, 23, 22, 36, 37, 38, 39, 40, 41, 47, 46, 45, 58, 57, 56, 55, 54, 53, 61, 52, 51, 60, 59, 48, 9, 8, 7, 6, 35, 29, 30, 31, 42, 43, 32, 33, 34, 44, 15, 76, 77, 79, 78, 80, 82, 81, 83, 85, 84, 74, 75, 69, 68, 67, 66, 73, 72, 71, 70, 64, 63, 65, 62]
	#t_nodes_bins_cap = [0,93,58,13,55,69,2,79,12,65,45,30,97,92,25,80,58,80,44,78,21,54,16,13,45,79,34,60,99,15,9,90,11,73,76,57,60,34,17,89,65,8,1,18,11,77,85,0,71,93,89,0,99,5,65,81,27,2,2,17,54,53,9,62,35,91,50,0,44,91,55,36,0,75,93,36,49,81,41,6,36,88,57,13,7,70,42]
	#num_visited = [0, 3, 2, 4, 6, 5, 8, 7, 10, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 22, 24, 123, 124, 25, 125, 27, 26, 28, 29, 30, 34, 35, 36, 37, 38, 39, 33, 32, 31, 47, 48, 44, 42, 40, 41, 43, 45, 46, 120, 122, 121, 50, 49, 53, 51, 52, 54, 55, 59, 57, 58, 56, 60, 63, 64, 65, 61, 66, 67, 62, 68, 71, 72, 73, 75, 76, 77, 78, 80, 79, 81, 74, 70, 69, 86, 85, 87, 97, 88, 96, 84, 89, 94, 90, 93, 91, 92, 100, 98, 101, 99, 102, 103, 104, 106, 105, 107, 108, 110, 111, 109, 113, 112, 114, 116, 118, 115, 117, 119, 179, 181, 180, 182, 183, 184, 185, 186, 187, 189, 188, 190, 178, 177, 176, 175, 174, 173, 171, 172, 170, 169, 168, 167, 166, 165, 163, 164, 143, 142, 141, 144, 145, 146, 152, 153, 151, 148, 150, 147, 149, 161, 160, 159, 155, 158, 162, 157, 156, 154, 133, 139, 132, 131, 130, 129, 136, 135, 137, 134, 138, 140, 82, 83, 95, 128, 127, 126, 1, 191, 192, 210, 211, 212, 213, 209, 208, 207, 204, 202, 206, 201, 200, 199, 198, 203, 196, 197, 195, 194, 193, 205]
	#t_nodes_bins_cap =[0,14,18,70,72,78,81,95,96,67,69,84,88,65,79,43,30,22,14,53,52,19,57,6,25,28,2,33,75,29,74,45,46,92,14,14,58,55,96,90,8,39,37,20,11,41,93,52,5,84,18,3,38,29,56,7,15,100,14,48,12,7,23,61,100,60,94,36,33,61,49,5,46,2,43,78,72,91,78,77,93,64,50,6,51,18,70,5,70,6,55,65,69,83,51,28,74,78,25,72,48,3,11,21,59,64,71,54,12,4,65,4,6,20,58,83,62,4,87,100,31,8,24,42,92,6,31,39,5,69,46,88,38,11,84,76,27,32,96,97,19,70,100,47,56,64,5,56,5,52,10,87,56,64,98,27,5,95,50,34,66,21,5,93,90,40,63,66,84,57,72,19,18,23,6,69,31,25,6,71,15,65,32,24,59,58,79,89,31,47,64,59,97,95,11,63,82,64,53,84,56,4,47,40,99,98,19,99,1,53,9,41,77,8]
	print ('c:',num_visited)
	print ('g:',t_nodes_bins_cap)
	print ('ms_route: ',num_visited_cap)
	f_cap, f_energy,f_time,f_distance, f_route, b_changes, b_energy, b_time = f_mm_route(num_visited,t_nodes_bins_cap)
	num_visited_ms, f_route_ms, f_energy_ms, f_time_ms, f_dist = f_ms_route(num_visited_cap)
	print (num_visited)
	cap_dist = []
	cap_dist_ms = []
	num_visited_cap_t_t = flatten(num_visited_cap)
	for i in range(0,len(num_visited)-1):
		#print (num_visited_cap_t_t[i], num_visited_cap_t_t[i+1])
		cap_dist.append(dist[num_visited[i], num_visited[i+1]])
	for i in range(0,len(num_visited_ms)-1):
		cap_dist_ms.append(dist[num_visited_ms[i], num_visited_ms[i+1]])
			
	
	constant_energy = constant_energy*f_time		
	# print all the results from path planning
	print ('Scenario:',filename)
	print ('Value function:')
	print ('Number of dustbins:', len(num_visited))
	print ('Emptying times: ',f_cap)
	print ('MURMEL route: ',num_visited_cap)
	print ('Mothership route: ',num_visited_ms)
	print ('MURMEL Energy in KWhs: ',f_energy, constant_energy, constant_energy+f_energy)
	print ('MURMEL Time in hrs: ', f_time)
	print ('MURMEL Distance in km: ',f_distance) 
	print ('MS Energy in KWhs: ',f_energy_ms)
	print ('MS Time in hrs: ', f_time_ms)
	print ('MS Distance in km:', f_dist)
	print ('Battery changes: ', b_changes)
	print ('Swap time in hrs: ', b_time)
	print ('Total Energy in KWhs: ',f_energy+f_energy_ms+constant_energy)
	print ('Total Time in hrs: ', max((f_time+b_time),(f_time_ms+b_time)))
	print ('Total Distance in km: ', f_distance+f_dist)
	print('Distance MM traveled breakdown: ', cap_dist)
	print('Distance MS traveled breakdown: ', cap_dist_ms)
	print ('Finished')

	if show_gui:
		gui(f_route,f_route_ms)		

