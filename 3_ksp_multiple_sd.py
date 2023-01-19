import sys
import math	
import numpy as np
from  collections.abc import Iterable	#flatten lists
import os								# access operationg system dependent file paths
import matplotlib.pyplot as plt					# plotting solution process and graphs
from itertools import zip_longest

''' 
Dynamic Knapsack Problem, returns the maximum value that can be put in a knapsack of capacity W given weight and value
'''

# part 0: settings
filename = 'moabit87_2'						# select vrp file
#filename = 'moabit214_2'						# select vrp file
#filename = 'monbijou-james-simon_2'				# select vrp file
show_gui = True
num_mm = 3
# part 1: MURMEL energy and time
## part 1.1: murmel energy
speed_murmel= 1/3.24							# time per distance in h per km (MURMEL) based on 0.9 m/s maximum speed in Urbanek bachelor thesis
energy_murmel_loc = 0.122						# energy consumption mobility of MURMEL in KWh per kmilometer
energy_murmel_bin = 0.0185						# energy consumption per bin in KWh
murmel_capacity = 100							# on average, a mothership visit is necessary every 'murmel_capacity's waypoint
## part 1.2: murmel time compression
time_adjust_bin = 120/3600						# seconds-hr, whole process of opening and closing trash can 		
time_compress  = 300/3600 						# seconds-hr
#time_emptying = 42.5/3600						# time to empty a dustbin in hr (MURMEL) based on 42.5s in Merle Simulation

# part 2: MOTHERSHIP energy and time
speed_mothership = 18							# time per distance in h per km (mothership)
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
	time_cost = dist*speed_murmel
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
	time_cost = dist/speed_mothership
	return (energy_cost,time_cost) 

# part 1.5: battery energy and time cost
def cost_battery(energy_cost,time_cost):
	battery_changes = math.ceil(energy_cost/battery_capacity)
	energy_cost = battery_changes
	time_cost = (time_swapping_battery*battery_changes)
	return (battery_changes,energy_cost,time_cost)

def init_route(murmel_capacity, nodes_bins_cap_2,value_current,n,nodes_coor_2, nodes_num_2,num_visited,num_visited_cap):
	# initialize the route
	for mm in range (1,num_mm+1):
		num_visited,c_values = knapSack( murmel_capacity, nodes_bins_cap_2,value_current,n,nodes_coor_2, nodes_num_2)
		value_current = c_values[num_visited[0]].tolist()
		n = len(nodes_bins_cap_2)
	return (num_visited,num_visited_cap,value_current,c_values)

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
		num_visited_cap.append(a)
		values_t = np.delete(value, num_visited, 1) # last arg colum 1 / row 0
	#return the value list of the visited node
	return (num_visited,values_t)
	
# part 2.1: update listed after nodes are visited
def update_m(t_val, t_wt, t_coor,t_nod,t_point,t_num_visited_r,j):
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
			t_num_visited_r.append(int(num[1]))
		#print (a)
		#print (j)
		j.append(a)
		#print (t_num_visited_r)
		#print (j)
		#input()
		values_t = np.delete(value, t_num_visited_r, 1) # last arg colum 1 / row 0
	#return the value list of the visited node
	return (t_num_visited_r,values_t,j)

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

# part 2: coptimize path between value and cost
def knapSack_m(W, wt, val, n,coor, point_num,t_num_visited_r,t_num_visited_cap_r):
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
		n_point, values,t_num_visited_cap_r = update_m(val, wt, coor,coor_k[n][W],point_num,t_num_visited_r,t_num_visited_cap_r)
	return (n_point,values,t_num_visited_cap_r) 

# part 3: calculate final cost and energy of selected pathfor MURMEL
def f_mm_route(points_cap,points):
	mm_f_cap = []
	mm_f_energy_cost_bins = []
	mm_f_time_cost_bins = []
	mm_f_dist_cost =[]
	mm_final_route = []
	mm_battery_changes = []
	mm_f_energy_cost_battery = []
	mm_f_time_cost_battery = []

	for j in range (0,len(points_cap)):
		# calculate the final cost energy, time and distance 
		final_route = []
		f_dist_cost = 0
		d_energy_cost_loc = 0
		d_time_cost_loc = 0
		f_cap = len(points_cap[j])-1
		t_nodes_coor = nodes_coor.tolist()
		for i in points[j]:
			final_route.append(t_nodes_coor[i])
		for i in range (0,len(points[j])-1):
			f_dist_cost +=  (dist[points[j][i], points[j][i+1]])
			a = dist[points[j][i], points[j][i+1]]
			b,c = cost_murmel_distance(a)
			d_energy_cost_loc += b
			d_time_cost_loc += c
		energy_cost_bin, time_cost_bin = cost_murmel_compressing(f_cap)
		#print (d_energy_cost_loc,energy_cost_bin,d_time_cost_loc,time_cost_bin) #consumption of murmel divided into distance and bins 
		f_energy_cost_bins = d_energy_cost_loc+energy_cost_bin
		f_time_cost_bins = d_time_cost_loc+time_cost_bin
		battery_changes, f_energy_cost_battery, f_time_cost_battery = cost_battery(f_energy_cost_bins,f_time_cost_bins)
		#print (f_cap,f_energy_cost_bins,f_time_cost_bins,f_dist_cost,final_route, battery_changes,f_energy_cost_battery,f_time_cost_battery)
		mm_f_cap.append(f_cap)
		mm_f_energy_cost_bins.append(f_energy_cost_bins)
		mm_f_time_cost_bins.append(f_time_cost_bins)
		mm_f_dist_cost.append(f_dist_cost)
		mm_final_route.append(final_route)
		mm_battery_changes.append(battery_changes)
		mm_f_energy_cost_battery.append(f_energy_cost_battery)
		mm_f_time_cost_battery.append(f_time_cost_battery)
	return (mm_f_cap,mm_f_energy_cost_bins,mm_f_time_cost_bins,mm_f_dist_cost,mm_final_route, mm_battery_changes,mm_f_energy_cost_battery,mm_f_time_cost_battery)

# part 4: calculate final cost and energy of selected pathfor Mothership
def f_ms_route(mm_f_rout_cap):
	t_nodes_coor = nodes_coor.tolist()
	coor_ms = []
	num_visited_ms = []
	f_dist_cost = 0
	dis_temp = 0
	d_energy_cost_loc = 0
	d_time_cost_loc = 0
	for path in mm_f_rout_cap:
		t_num_visited_ms = []
		t_coor_ms = []
		for point in path:
			t_num_visited_ms.append(point[-1])
			t_coor_ms.append(t_nodes_coor[point[-1]])
			#num_visited_ms.append(point[-1])
			#coor_ms.append(t_nodes_coor[point[-1]])
		num_visited_ms.append(t_num_visited_ms)
		coor_ms.append(t_coor_ms)
	#print (num_visited_ms)
	#print (coor_ms)
	num_visited_ms = [list(filter(None,i)) for i in zip_longest(*num_visited_ms)]
	coor_ms = [list(filter(None,i)) for i in zip_longest(*coor_ms)]
	num_visited_ms = flatten(num_visited_ms)
	coor_ms = flatten(coor_ms)
	#print (num_visited_ms)
	#print (coor_ms)
	#input()
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
	col=['blue','darkgreen','gray','lightblue']
	x = []
	y = []
	print ('**************')
	#print (f_route)
	#input()
	
	for i in range (0,len(f_route)):
		s_route = np.array(f_route[i])
		#print (s_route)
		#print (s_route[:,1])
		#input()
		x = s_route[:,1] #lon
		y = s_route[:,0] #lat
		plt.scatter(x, y,color=col[i])
		plt.plot(x,y, color=col[i])
	plt.xlabel("Longitud")
	plt.ylabel("Latitud")
	#f_route_ms = [[el] for el in f_route_ms]
	f_route_ms = [f_route_ms[i: i+2] for i in range(0, len(f_route_ms), 2)]
	f_route_ms = np.array(f_route_ms)
	x2 = f_route_ms[:,1] #lon
	y2 = f_route_ms[:,0] #lat
	plt.scatter(x2, y2,color=col[-1])
	plt.plot(x2,y2,color=col[-1])
	plt.show()

if __name__ == "__main__":
	file_oppening(filename)
	print('#Debug info: input file nodes part:')
	for node in nodes:
		print ('#   ' + str(node))

	# get nodes specifications in different arrays
	nodes_num = np.array(nodes)[:,[0]]
	nodes_coor = np.array(nodes)[:,[1,2]]
	nodes_bins_cap = np.array(nodes)[:,[3]].astype(int)
	# generate cost and value functions for murmel
	dist = np.zeros((len(nodes_coor), len(nodes_coor)))
	#costs = np.zeros((len(nodes_coor), len(nodes_coor)))
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
	nodes_bins_cap_2= flatten(nodes_bins_cap)
	# eliminate origin point of MURMEL
	nodes_coor_2 = nodes_coor_2[1:]
	nodes_num_2 = nodes_num_2[1:]
	nodes_bins_cap_2 = nodes_bins_cap_2[1:]
	# get the value function from initial point
	value_current = value[1].tolist()
	value_current = value_current[1:]
	num_visited = [0]
	num_visited_cap = []
	# part 1: calculate the route of MURMEL
	num_visited, num_visited_cap,value_current, c_values = init_route( murmel_capacity, nodes_bins_cap_2,value_current,len(nodes_bins_cap_2),nodes_coor_2, nodes_num_2,num_visited,num_visited_cap)
	for mm in num_visited_cap:
		mm.insert(0,0)
	# copied route so it would not change th e inital path
	t_num_visited = num_visited[:]
	t_num_visited_cap_r = []
	n = len(nodes_bins_cap_2)
	#print (murmel_capacity, nodes_bins_cap_2,value_current,n,nodes_coor_2, nodes_num_2)
	#print (murmel_capacity,len(nodes_bins_cap_2),len(value_current),n,len(nodes_coor_2),len(nodes_num_2))
	#print (t_num_visited,num_visited_cap)
	initial_points = num_visited_cap[:]
	for mm_init in num_visited_cap:
		t_num_visited_r = mm_init[:]  # inital route! 
		t_num_visited_cap_r.append(flatten(mm_init[:])) # inital route! 
		value_current = c_values[t_num_visited_r[-1]].tolist()
		#print(t_num_visited_r,t_num_visited_cap_r)
		#print (mm_init)
		a = flatten(t_num_visited_r)
		#input()
		while  len(t_num_visited_r) <= math.ceil(len(nodes_num.tolist())/num_mm)-2:
			#print (t_num_visited_r) 
			#input()
			n = len(nodes_bins_cap_2)
			t_num_visited_r,c_values,t_num_visited_cap_r = knapSack_m(murmel_capacity, nodes_bins_cap_2,value_current,n,nodes_coor_2, nodes_num_2,t_num_visited_r,t_num_visited_cap_r)
			value_current = c_values[t_num_visited_r[-1]].tolist()
	#print ('¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡¡')

	## points we have the inital routes, 
	### the point is repeated one takes the place of a point so is not taken 
	#### when recalculating with game theory all thepoints should be taken.

	c = []
	d = []
	for i in range (0,len(t_num_visited_cap_r)):
		#print (i)
		d.append(t_num_visited_cap_r[i])
		if t_num_visited_cap_r[i] in initial_points:
			c.append(d[:-1])
			d = []
			d.append(t_num_visited_cap_r[i])
		if i == len(t_num_visited_cap_r)-1:
			c.append(d)
			c = c[1:]
	print ('-----------------------------------------------------')
	#print (c)
	g = []
	for yy in c:
		flat_list = []
		#print (yy)
		g.append(flatten(yy))
	#print (g)
	#print (t_num_visited_r,t_num_visited_cap_r)
	#print (num_visited,num_visited_cap)
	#print (t_num_visited)
	#print (len(t_num_visited_r),len(t_num_visited_cap_r),len(num_visited),len(num_visited_cap))
	
	#print (num_visited_cap,num_visited)
	# calculate final path on energy time and distance
	print ('c:',c)
	print ('g:',g)
	f_cap, f_energy,f_time,f_distance, f_route, b_changes, b_energy, b_time = f_mm_route(c,g)
	#fix the mothership  route 
	num_visited_ms, f_route_ms, f_energy_ms, f_time_ms, f_dist = f_ms_route(c)
	# print all the results from path planning
	f_constant_energy = []
	ff_energy = []
	# print all the results from path planning
	print ('Scenario:',filename)
	print ('Value function:')
	print ('Number of dustbins:', len(flatten(g)))
	print ('Emptying times: ',f_cap)
	for i in range (0,len(c)):
		print ('MURMEL route ', i+1 ,':',c[i])
		f_constant_energy.append(constant_energy*f_time[i])
		ff_energy.append(f_constant_energy[i]+f_energy[i])
	print ('Mothership route: ',num_visited_ms)
	print ('MURMEL Energy in KWhs: ',f_energy, f_constant_energy,f_energy,ff_energy)
	print ('MURMEL Time in hrs: ', f_time)
	print ('MURMEL Distance in km: ',f_distance) 
	print ('MS Energy in KWhs: ',f_energy_ms)
	print ('MS Time in hrs: ', f_time_ms)
	print ('MS Distance in km:', f_dist)
	print ('Battery changes: ', b_changes)
	print ('Swap time in hrs: ', b_time)
	print ('Total Energy in KWhs: ',[x + y for x, y in zip(f_energy, ff_energy)],f_energy_ms,sum(f_energy)+sum(ff_energy)+f_energy_ms)
	print ('Total Time in hrs: ', max(max([x + y for x, y in zip(f_time, b_time)]),(f_time_ms+max(b_time))))
	print ('Total Distance in km: ', sum(f_distance)+f_dist, f_distance,f_dist)
	print (sum(f_energy)+sum(ff_energy)+f_energy_ms, max(max([x + y for x, y in zip(f_time, b_time)]),(f_time_ms+max(b_time))),sum(f_distance)+f_dist)
	print ('Finished')
	if show_gui:
		gui(f_route,f_route_ms)
	




