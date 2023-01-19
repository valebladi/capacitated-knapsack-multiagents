#!/usr/bin/env python3
import sys
import numpy as np							# mathematic functions, easy to use arrays
import math								# basic mathematic functions
import random as rd							# random number generation
import csv								# read and write comma separated files
import argparse							# to create command line options
import matplotlib							# plotting solution process and graphs
import matplotlib.pyplot as plt					# plotting solution process and graphs
from matplotlib.lines import Line2D
import copy								# allows deep copy operations
import time								# measuring time
import os								# access operationg system dependent file paths
from pathlib import Path						# working with paths independent of the operationg system
from enum import Enum							# working with enumerations
from itertools import zip_longest

class Debug(Enum):
	OFF = 0
	MIN = 1
	FULL = 2

	# define str() to use in it printf
	def __str__(self):
		return self.name

class Initial(Enum):
	NEARESTNEIGHBOR = 1
	SWEEP = 0

	def __str__(self):
		return self.name

class Operator(Enum):
	TWOOPT = 0
	RELOCATE = 1
	GLOBAL_RELOCATE = 2
	GLOBAL_EXCHANGE = 3

	def __str__(self):
		return self.name

class CostFunction(Enum):
	ENERGY_CONSUMPTION = 1
	TIME = 0

	def __str__(self):
		return self.name

class EdgeWeight(Enum):
	EUCLIDEAN = 0
	GEOGRAPHIC = 1

	def __str__(self):
		return self.name

class ShowGui(Enum):
	OFF = 0
	SOLUTION = 1
	STEPS = 2

	def __str__(self):
		return self.name

### part 0: settings
filename = 'moabit87_1'						# select vrp file
#filename = 'moabit214_1'						# select vrp file
#filename = 'monbijou-james-simon_1'							# select vrp file
debug_output = Debug.FULL						# amount of debugging output
plot_pause = 0.0001							# visualization delay between plot steps
iterations = 10000							# number of improvement steps
initial_sln = Initial.NEARESTNEIGHBOR					# initial solution method
operator_weights = [4, 1, 1, 4]					# weights to change permutation operator rates [TWOOPT, RELOCATE, GLOBAL_RELOCATE, GLOBAL_EXCHANGE]
cost_function = CostFunction.ENERGY_CONSUMPTION			# the objective function for the optimization
show_node_numbers = False						# visualizes the node numbers, if true
weight_murmel = 0.17							# energy consumption kWh per km (MURMEL)
weight_mothership = 0.27						# energy consumption kWh per km (mothership)
energy_dropoff = 0							# energy consumption for dropping off load at mothership in kWh TODO
speed_murmel= 1/3.24							# time per distance in h per km (MURMEL) based on 0.9 m/s maximum speed in Urbanek bachelor thesis
speed_mothership = 1/50						# time per distance in h per km (mothership)
time_emptying = 42.5/3600						# time to empty a dustbin in h (MURMEL) based on 42.5s in Merle Simulation
time_dropoff = 1/60							# time to empty Murmels load into mothership in h
energy_emptying = 0.0513 * time_emptying				# energy consumption for emptying a dustbin in kWh (MURMEL)
murmel_capacity = 4							# on average, a mothership visit is necessary every 'murmel_capacity's waypoint
n_murmels = 3
cm = 1/2.54								# centimeters in inches
column_width = 8.4*cm							# width of single paper column
dpi = 600								# resolution of saved .png figures
show_gui = ShowGui.OFF							# displays solution, temperature and cost at the end (SOLUTION) or over time (STEPS)
draw_changes = False							# displays the last changes with purple dashed lines
Tmax = 100.0								# initial temperature of annealing schedule
Tmin = 0.1								# final temperature of annealing schedule
edge_weight = EdgeWeight.GEOGRAPHIC				# which distance between two nodes
replicate_paper = False					# Fix result to Tours shwon in the paper, only use for visualization
draw_legend = False						# Dislay the legend of the route graph


# textsizes for plots
fsize=10
params = {'legend.fontsize': fsize*0.8,
          'axes.labelsize': fsize*0.9,
          'axes.titlesize': fsize,
          'xtick.labelsize': fsize*0.8,
          'ytick.labelsize': fsize*0.8,
          'axes.titlepad': fsize*1.5,
          'font.family': 'serif',
          'font.serif': ['cmr10']}
plt.rcParams.update(params)

# check font; returns the default font name, if font is not installed
#from matplotlib.font_manager import findfont, FontProperties
#print(findfont(FontProperties(family=matplotlib.rcParams['font.family'])))

# help function to accept only positive values at the argument parser
def check_positive(value):
	ivalue = int(value)
	if ivalue <= 0:
		raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
	return ivalue
	
# define and parse command line options
arg_parser = argparse.ArgumentParser(allow_abbrev=False)
arg_parser.add_argument('--file', action='store', type=str, help='Filename of the TSP file without ending.')
arg_parser.add_argument('--debug', type=lambda debug: Debug[debug], choices=list(Debug), help='Defines the amount of command line debugging messages. Default: ' + str(debug_output))
arg_parser.add_argument('--initial', type=lambda initial: Initial[initial], choices=list(Initial), help='Defines the initial solution strategy. Default: ' + str(initial_sln))
arg_parser.add_argument('--operator_weights', metavar=('weight_two_opt', 'weight_relocate', 'weight_global_relocate', 'weight_global_exchange'), action='store', nargs=4, type=float, help='Defines the weights for the permutation operators. It will be normalized to the sum 1, to be a valid probability distribution. Default: ' + ' '.join(map(str,operator_weights)))
arg_parser.add_argument('--iterations', action='store', type=check_positive, help='Defines the number of iterations per simulated annealing run. Attention: runtime increases approximately linearly with the number of iterations. Default: ' + str(iterations))
arg_parser.add_argument('--n_murmels', action='store', type=check_positive, help='Defines the number MURMEL robots. Default: ' + str(n_murmels))
arg_parser.add_argument('--murmel_capacity', action='store', type=check_positive, help='Defines the number of visited dustbins, until the next mothership stop is required. Default: ' + str(murmel_capacity))
arg_parser.add_argument('--show_gui', type=lambda show: ShowGui[show], choices=list(ShowGui), help="Creates a GUI which shows route, cost and temperature at the end (SOLUTION) or over time (STEPS). If the GUI is enabled, shown results are saved as .png and .pdf images at the folder 'figures'. Default: " + str(show_gui))
arg_parser.add_argument('--draw_changes', choices=[True, False], type=lambda x: (str(x).lower() == 'true'), help='Visualizes the last changed edges as dashed purple lines, if true and GUI is enabled. Default: ' + str(draw_changes))
arg_parser.add_argument('--temperature_maximum', type=float, help='Sets the maximum annealing temperature. Default: ' + str(Tmax))
arg_parser.add_argument('--temperature_minimum', type=float, help='Sets the minimum annealing temperature. Should be lower than temperature_maximum. Default: ' + str(Tmin))
args = arg_parser.parse_args()
if debug_output != Debug.OFF and args.debug != Debug.OFF: print('Argument: debug: ' + str(args.debug))

# apply parsed command line arguments
if args.debug != None: debug_output=args.debug
if args.file != None: filename=args.file
if args.initial != None: initial_sln=args.initial
if args.iterations != None: iterations=args.iterations
if args.operator_weights != None: operator_weights=args.operator_weights
if args.show_gui != None: show_gui=args.show_gui
if args.temperature_maximum != None: Tmax=args.temperature_maximum
if args.temperature_minimum != None: Tmin=args.temperature_minimum
if args.n_murmels != None: n_murmels=args.n_murmels
if args.draw_changes != None: draw_changes=args.draw_changes
if args.murmel_capacity != None: murmel_capacity=args.murmel_capacity

if Tmax < Tmin: print('WARNING: Initial temperature is lower than final temperature. This normally causes poor results.')

if debug_output != Debug.OFF:
	print('Argument: file: ' + str(args.file))
	print('Argument: initial solution heuristic: ' + str(args.initial))
	print('Argument: iterations: ' + str(args.iterations))
	print('Argument: operator_weights: ' + str(args.operator_weights))
	print('Argument: show_gui: ' + str(args.show_gui))
	print('Argument: draw_changes: ' + str(args.draw_changes))
	print('Argument: Tmax: ' + str(args.temperature_maximum))
	print('Argument: Tmin: ' + str(args.temperature_minimum))
	print('Argument: n_murmels: ' + str(args.n_murmels))
	print('Argument: murmel_capacity: ' + str(args.murmel_capacity))

# only intra route operators, if just 1 MURMEL robot
if n_murmels == 1:
	operator_weights[2] = operator_weights[3] = 0
elif n_murmels < 1: # less than 1 MURMEL makes no sense
	print('ERROR: Number of MURMELs is to small. Increase it to 1 or more!\nExiting..')
	sys.exit()

### part 1: get waypoints and draw them
file = os.path.join(os.path.dirname(__file__), 'example_problems', filename)
if debug_output != Debug.OFF: print('Try to open ' + filename + '.tsp: ', end='')
try:
	with open(file + '.tsp', 'r') as tsp_file:
		tsp_file_data = tsp_file.readlines()
except Exception as e:
	if debug_output != Debug.OFF: print('error!\nExiting..') # more exception details: str(e)
	sys.exit()
if debug_output != Debug.OFF: print('successful!')

#print(tsp_file_data) # print input file; may be useful to debug tsp parser

# possible entries in specification part:
specification_list = ['NAME', 'TYPE', 'COMMENT', 'DIMENSION', 'CAPACITY', 'GRAPH_TYPE', 'EDGE_TYPE', 'EDGE_WEIGHT_TYPE', 'EDGE_WEIGHT_FORMAT', 'EDGE_DATA_FORMAT', 'NODE_TYPE', 'NODE_COORD_TYPE', 'COORD1_OFFSET', 'COORD1_SCALE', 'COORD2_OFFSET', 'COORD2_SCALE', 'COORD3_OFFSET', 'COORD3_SCALE', 'DISPLAY_DATA_TYPE']

specification = [None] * len(specification_list)
node_data = False
nodes = []
for data_line in tsp_file_data:
	data_line = data_line.replace('\n', '')
	if node_data:
		node = data_line.split()
		if len(node) == 3:
			try:
				node[0], node[1], node[2] = int(node[0]), float(node[1]), float(node[2])
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
		
	if (data_line.find('EDGE_WEIGHT_TYPE') == 0 and data_line.find('GEO') > 0): # keep the standard if not 'GEO'
		edge_weight = EdgeWeight.GEOGRAPHIC

### swap x and y for better visualization: x = latitude, y = longitude
if edge_weight == EdgeWeight.GEOGRAPHIC:
	for node in nodes:
		node[1], node[2] = node[2], node[1]

if debug_output == Debug.FULL:
	print('#Debug info: input file specification part:')
	for counter, value in enumerate(specification_list):
		print ('#   ' + value + ': ' + str(specification[counter]))

	print('#Debug info: input file nodes part:')
	for node in nodes:
		print ('#   ' + str(node))

nodes = np.array(nodes)[:,[1,2]]

### visualization helper functions
initial_color='cyan'
def draw_nodes(nodes):
	ax1.plot(nodes[:1,0], nodes[:1,1], color=initial_color, marker='o', linestyle='', label='initial waypoint')
	if show_gui==ShowGui.STEPS: ax1.plot(nodes[1:,0], nodes[1:,1], color='orange', marker='o', linestyle='', label='waypoint')

	for i in range(len(nodes)):
		if show_node_numbers: ax1.text(nodes[i,0], nodes[i,1], str(i), fontsize=fsize*0.75)

def draw_node(index, color,marker, label):
	return ax1.plot(nodes[index, 0], nodes[index, 1], color=color, marker=marker, linestyle='', label=label)

def draw_line(start, end, color, style, legendlabel):
	return ax1.plot([nodes[start,0], nodes[end,0]],[nodes[start,1], nodes[end,1]], color=color, linestyle=style, label=legendlabel)

def move_figure(f, x, y):
	backend = matplotlib.get_backend()
	if backend == 'TkAgg':
		f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
	elif backend == 'WXAgg':
		f.canvas.manager.window.SetPosition((x, y))
	else:
		# This works for QT and GTK
		f.canvas.manager.window.move(x, y)

if show_gui != ShowGui.OFF:
	fig1, ax1 = plt.subplots(1, 1, constrained_layout=True)
	fig2, ax2 = plt.subplots(1, 1, constrained_layout=True)
	fig3, ax3 = plt.subplots(1, 1, constrained_layout=True)
	fig1.set_size_inches(column_width, column_width*1.06)
	fig2.set_size_inches(column_width, column_width)
	fig3.set_size_inches(column_width, column_width)
	move_figure(fig1, 100 + 0*120*column_width, 100)
	move_figure(fig2, 100 + 1*120*column_width, 100)
	move_figure(fig3, 100 + 2*120*column_width, 100)
	draw_nodes(nodes)
	ax1.yaxis.set_major_locator(plt.MaxNLocator(5)) # avoid yticklabel overlap
	# ax1.set_title('Current solution')
	ax2.set_title('Tour cost')
	ax3.set_title('Simulated annealing temperature')
	dst_unit = '$(^\circ)$' if edge_weight == EdgeWeight.GEOGRAPHIC else '(km)'
	x_label = 'longitude' if edge_weight == EdgeWeight.GEOGRAPHIC else 'x position'
	y_label = 'latitude' if edge_weight == EdgeWeight.GEOGRAPHIC else 'y position'
	ax1.set_xlabel(x_label + ' ' +  dst_unit)
	ax2.set_xlabel('iteration i')
	ax3.set_xlabel('iteration i')
	ax1.set_ylabel(y_label + ' ' + dst_unit)
	ax2.set_ylabel('f(i) (kWh)')
	ax3.set_ylabel('T(i)')
	ax3.set_xlim([0,iterations])
	ax3.set_ylim([Tmin,Tmax])
	ax1.tick_params(axis='y', labelrotation=90)
	ax2.tick_params(axis='y', labelrotation=90)
	ax3.tick_params(axis='y', labelrotation=90)
	ax1.ticklabel_format(useOffset=False)

### part 2: create cost matrix
def euclidean_2d_distance(i, j):
	xd = nodes[i,0] - nodes[j,0]
	yd = nodes[i,1] - nodes[j,1]
	return int(math.sqrt(xd*xd+yd*yd)+0.5)

def geographic_2d_distance(i, j):
	R = 6371  # Earth radius in kilometers

	dLat = math.radians(nodes[j,0] - nodes[i,0])
	dLon = math.radians(nodes[j,1] - nodes[i,1])
	lat1 = math.radians(nodes[i,0])
	lat2 = math.radians(nodes[j,0])

	return 2 * R * math.asin(math.sqrt(math.sin(dLat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dLon / 2)**2))

use_linear_distance = False
try:
	# try to load Phillips cost matrix
	costs = np.loadtxt(open(file + '_distances.csv', "rb"), delimiter=",", skiprows=2, max_rows=len(nodes), usecols=range(2,len(nodes)+2))*km

	if not(len(nodes) == costs.shape[0] and len(nodes) == costs.shape[1]):
		use_linear_distance = True

except Exception as e:
	if debug_output != Debug.OFF: print('No distances file.') # more exception details: str(e)
	use_linear_distance = True

if use_linear_distance:
	costs = np.zeros((len(nodes), len(nodes)))
	for i in range(len(nodes)):
		for j in range(i):
			if edge_weight == EdgeWeight.EUCLIDEAN:
				costs[i,j] = euclidean_2d_distance(i,j)
			if edge_weight == EdgeWeight.GEOGRAPHIC:
				costs[i,j] = geographic_2d_distance(i,j)
			costs[j,i] = costs[i,j]

def murmel_cost(tour: "list[int]"):
	'''calculate the cost of the route a murmel takes'''
	cost = 0
	distance = 0
	for i in range(len(tour)):
		distance += costs[tour[i-1], tour[i]]
	dropoffs = math.ceil(len(tour)/murmel_capacity)
	bins = len(tour)
	if cost_function == CostFunction.ENERGY_CONSUMPTION:
		cost += bins * energy_emptying
		cost += dropoffs * energy_dropoff
		cost += weight_murmel* distance
	elif cost_function == CostFunction.TIME:
		cost += bins * time_emptying
		cost += dropoffs * time_dropoff
		cost += speed_murmel * distance
	return cost

def ms_cost(ms_tour: "list[int]"):
	'''calculate the cost of the route the ms takes'''
	cost = 0
	distance = 0
	for i in range(len(ms_tour)):
		distance += costs[ms_tour[i-1], ms_tour[i]]
	if cost_function == CostFunction.ENERGY_CONSUMPTION:
		cost += distance * weight_mothership
	elif cost_function == CostFunction.TIME:
		cost += time_dropoff * len(ms_tour)
		cost += speed_mothership*distance
	return cost

costs_per_MURMEL = [] 
def tour_cost(tour_global: "list[list[int]]", ms_tour: "list[int]"):
	'''calculate the total cost of all routes combined'''
	cost = 0
	costs_per_MURMEL = [] 
	if cost_function == CostFunction.ENERGY_CONSUMPTION:
		for i in range(len(tour_global)):
			new_cost = murmel_cost(tour_global[i])
			costs_per_MURMEL.append(new_cost)
			cost+= new_cost
		cost += ms_cost(ms_tour)
	elif cost_function == CostFunction.TIME:
		times = []
		for i in range(len(tour_global)):
			new_cost = murmel_cost(tour_global[i])
			costs_per_MURMEL.append(new_cost)
			times.append(new_cost)
		times.append(ms_cost(ms_tour))
		cost = max(times)
	return cost, costs_per_MURMEL

### part 3: create initial solution 

### part 3 a: nearest neigbhbor heuristic for n murmels
if initial_sln == Initial.NEARESTNEIGHBOR:
	inital_points = [0] * n_murmels
	tour_global = []
	visited_nodes = []
	visited_nodes.extend(inital_points)
	murmel_capacity_MA = math.ceil(len(nodes)/n_murmels)+1
	current = 0
	for point in inital_points:
		tour_local = []
		current = point
		tour_local.append(point)
		while len(tour_local) < murmel_capacity_MA and len(visited_nodes)-(len(inital_points)-1) < len(nodes): 
			last = current
			for i in range(0,len(nodes)):
				if (not i in tour_local) and (not i in visited_nodes):
					#print (i,visited_nodes)
					if (last == current):
						current = i
					elif costs[last, i] < costs[last, current]:
						current = i
			visited_nodes.append(current)
			tour_local.append(current)
		tour_global.append(tour_local)

### part 3 b: sweep heuristic for 3 murmels
if initial_sln == Initial.SWEEP:
	murmel_capacity_MA = math.ceil(len(nodes)/n_murmels)+1
	list_max_min = nodes.transpose()
	min_x_sweep = int(math.floor(min(list_max_min[0])))
	max_x_sweep = int(math.ceil(max(list_max_min[0])))
	min_y_sweep = int(math.floor(min(list_max_min[1])))
	max_y_sweep = int(math.ceil(max(list_max_min[1])))
	lower_bound = []
	upper_bound = []
	sweep_boarder_left = []
	sweep_boarder_right = []
	sweep_boarder_bottom = []
	sweep_boarder_top = []
	sweep_boarder = []
	visited_nodes = []
	tour_local = []
	tour_global = []

	def area_sweep(x1, y1, x2, y2, x3, y3):
		return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

	def zone_sweep(x1, y1, x2, y2, x3, y3, x, y):
		a = area_sweep(x1, y1, x2, y2, x3, y3)
		a1 = area_sweep(x, y, x2, y2, x3, y3)
		a2 = area_sweep(x1, y1, x, y, x3, y3)
		a3 = area_sweep(x1, y1, x2, y2, x, y)
		if(a == a1 + a2 + a3):
			return True
		else:
			return False
	# create boarder matrix in order fro sweeping
	for i in range(min_y_sweep, max_y_sweep+1) :
		sweep_boarder_left.append([min_x_sweep,i])
		sweep_boarder_right.append([max_x_sweep,i])
	for j in range(min_x_sweep, max_x_sweep-1) :
		sweep_boarder_bottom.append([j+1,min_y_sweep])
		sweep_boarder_top.append([j+1,max_y_sweep])
	sweep_boarder.extend(sweep_boarder_left)
	sweep_boarder.extend(sweep_boarder_top)
	sweep_boarder.extend(sweep_boarder_right[::-1])
	sweep_boarder.extend(sweep_boarder_bottom[::-1])
	sweep_boarder.append([0,0])

	# sweep heuristic
	lower_bound = sweep_boarder[0]
	tour_local.append(0) # append the first node
	for i in range (1,len(sweep_boarder)):
		upper_bound = sweep_boarder[i]
		for j in range (1,len(nodes)):
			if len(tour_local) == murmel_capacity_MA: 
					tour_global.append(tour_local)
					tour_local = [0] # check if they a all start with the same, so point 0 add in [0].
			elif (zone_sweep(round(nodes[0,0]),round(nodes[0,1]),round(lower_bound[0]),round(lower_bound[1]),round(upper_bound[0]),round(upper_bound[1]),round(nodes[j,0]),round(nodes[j,1])) 
				and j not in visited_nodes):
				if len(tour_local) < murmel_capacity_MA:
					tour_local.append(j)
					visited_nodes.append(j)
		lower_bound = upper_bound
	if tour_local:
		tour_global.append(tour_local)
		tour_local = []


def cost_route(ms_routes):
	route_ms_cost_all =[0] 
	current = 0
	for ms_route in ms_routes:
		route_ms_cost = []
		route_ms_cost.append(current)
		while len(route_ms_cost) < len(ms_route)+1:
			last = current
			for i in ms_route:
				if not i in route_ms_cost:
					if last == current:
						current = i
					elif costs[last, i] < costs[last, current]:
						current = i
			route_ms_cost.append(current)
		route_ms_cost_all.extend(route_ms_cost[1:])
	return route_ms_cost_all

### mother ship path
def calc_ms_tour(tour_global):
	ms_tour_MA = []
	for tour in tour_global:
		ms_tour_part = tour[::murmel_capacity]
		ms_tour_end = tour[-1]
		if ms_tour_end not in ms_tour_part:
			ms_tour_part.append(ms_tour_end)
		ms_tour_MA.append(ms_tour_part)
	ms_tour_MA = [list(filter(None,i)) for i in zip_longest(*ms_tour_MA)]
	ms_tour_MA = cost_route(ms_tour_MA[1:])
	return ms_tour_MA

tour_global_best = copy.deepcopy(tour_global)
ms_tour_best = ms_tour = calc_ms_tour(tour_global)
cost_best = cost_current = initial_cost = tour_cost(tour_global, ms_tour)
if debug_output != Debug.OFF: print('Init tour:' + str(tour_global) + '\nInit MS tour:' + str(ms_tour) + '\nCost of init tour: ' + str(initial_cost))

### draw tour and MS path
colors_path = [(228/255,26/255,28/255), (55/255, 126/255, 184/255), (152/255, 78/255, 163/255), 'royalblue', 'goldenrod', 'darkturquoise','limegreen', 'brown',  'yellow', 'purple',  'pink', 'turquoise', 'gray', 'darkred', 'teal', 'coral', (50/255, 50/255, 50/255)]
murmel_lines = []
ms_lines = []
murmel_nodes=[]
ms_nodes=[]
def draw_tour_MA(tour_global, ms_tour, touched):
	# remove old tours (TODO: only touch changed stuff to increase speed)
	for line in murmel_lines:
		line[0].remove()
	murmel_lines.clear()
	for line in ms_lines:
		line[0].remove()
	ms_lines.clear()
	for node in murmel_nodes:
		node[0].remove()
	murmel_nodes.clear()
	for node in ms_nodes:
		node[0].remove()
	ms_nodes.clear()
	# draw MURMEL tour
	for t in range(0,len(tour_global)):
		murmel_nodes.append(draw_node(tour_global[t][len(tour_global[t])-1], colors_path[t],'o', 'waypoint'))
		for i in range(len(tour_global[t])-1):
			label = 'robot ' + str(t+1) if i == 0 else ''
			if i>0 : murmel_nodes.append(draw_node(tour_global[t][i], colors_path[t],'o', ''))
			murmel_lines.append(draw_line(tour_global[t][i], tour_global[t][i+1], colors_path[t], 'solid', label))
			if (i, t) in touched:
				if draw_changes: murmel_lines.append(draw_line(tour_global[t][i], tour_global[t][i+1], 'purple', 'dashed', 'last change'))
	# draw mothership tour
	ms_label='.'
	for i in range(len(ms_tour)-1):
		label = 'mothership' if i == 0 else ''
		ms_lines.append(draw_line(ms_tour[i],ms_tour[i+1], colors_path[-1], 'solid', label))
		ms_nodes.append(draw_node(ms_tour[i], 'black', ms_label, ''))
	ms_nodes.append(draw_node(ms_tour[-1], 'black', ms_label, 'mothership waypoint'))
# removes edges (a0, a1) and (b0, b1) and adds edges (a0, b0) and (a1, b1). Returns new tour.
def two_opt(a0, b0, tour, selected_tour):
	# get a1 and b1
	a1 = a0+1
	b1 = (b0+1) % len(tour)
	
	if debug_output != Debug.OFF: print('(Local) 2-Opt nodes: a0: ' + str((selected_tour, a0)) + ' a1: ' + str((selected_tour, a1)) + ' b0: ' + str((selected_tour, b0)) + ' b1: ' + str((selected_tour, b1)))
	
	new_tour = list(tour[0:a1]) + list(reversed(tour[a1:b0+1])) + list(tour[b0+1:len(tour)])

	return new_tour, [(a0, selected_tour), (b0, selected_tour)]
	
def relocate(a0, b0, tour, selected_tour):
	a1 = a0+1
	a2 = a0+2
	b1 = (b0+1) % len(tour)
	
	if debug_output != Debug.OFF: print('(Local) relocate nodes: a0: ' + str((selected_tour, a0)) + ' a1: ' + str((selected_tour, a1)) + ' a2: ' + str((selected_tour, a2)) + ' b0: ' + str((selected_tour, b0)) + ' b1: ' + str((selected_tour, b1)))

	new_tour = list(tour[0:a1]) + list(tour[a2:b0+1]) + list(tour[a1:a1+1]) + list(tour[b0+1:len(tour)])

	return new_tour, [(a0, selected_tour), (a1, selected_tour), (b0, selected_tour)]

def global_relocate(a0, b0, tour_1, tour_2, selected_tour_1, selected_tour_2):
	a1 = a0+1
	a2 = a0+2
	b1 = b0+1

	if debug_output != Debug.OFF: print('(Global) relocate nodes: a0: ' + str((selected_tour_1, a0)) + ' a1: ' + str((selected_tour_1, a1)) + ' a2: ' + str((selected_tour_1, a2)) + ' b0: ' + str((selected_tour_2, b0)) + ' b1: ' + str((selected_tour_2, b1)))

	new_tour_1 = list(tour_1[0:a1]) + list(tour_1[a2:len(tour_1)])
	new_tour_2 = list(tour_2[0:b0+1]) + list(tour_1[a1:a1+1]) + list(tour_2[b1:len(tour_2)])

	return new_tour_1, new_tour_2, [(a0, selected_tour_1), (b0, selected_tour_2), (b1, selected_tour_2)]

def global_exchange(a0, b0, tour_1, tour_2, selected_tour_1, selected_tour_2):
	a1 = a0+1
	a2 = a0+2
	b1 = b0+1
	b2 = b0+2

	if debug_output != Debug.OFF: print('(Global) exchange nodes: a0: ' + str((selected_tour_1, a0)) + ' a1: ' + str((selected_tour_1, a1)) + ' a2: ' + str((selected_tour_1, a2)) + ' b0: ' + str((selected_tour_2, b0)) + ' b1: ' + str((selected_tour_2, b1)) + ' b2: ' + str((selected_tour_2, b2)))

	new_tour_1 = list(tour_1[0:a1]) + list(tour_2[b1:b1+1]) + list(tour_1[a2:len(tour_1)])
	new_tour_2 = list(tour_2[0:b0+1]) + list(tour_1[a1:a1+1]) + list(tour_2[b2:len(tour_2)])

	return new_tour_1, new_tour_2, [(a0, selected_tour_1), (a1, selected_tour_1), (b0, selected_tour_2), (b1, selected_tour_2)]

def do_permutation(tour_global, operator):
	### intra route operators
	if operator == Operator.RELOCATE or operator == Operator.TWOOPT:
		l = 0
		while (l <= 3 and operator == Operator.TWOOPT) or (l <= 4 and operator == Operator.RELOCATE): # tour need to be long enough (Relocate: 5 elements; 2-Opt: 4 elements)
			selected_tour = rd.randint(0, len(tour_global)-1)
			tour = tour_global[selected_tour]
			l = len(tour)-1
		b0 = a0 = 0
		if operator == Operator.RELOCATE:
			thr = 2
		else:
			thr = 1	
		while (abs(a0 - b0) % l) <= thr:
			a0 = rd.randint(0, l)
			b0 = rd.randint(0, l)
			
		if operator == Operator.RELOCATE:
			new_tour, touched = relocate(min(a0, b0), max(a0, b0), tour, selected_tour)
		else: # = elif operator == Operator.TWOOPT:
			new_tour, touched = two_opt(min(a0, b0), max(a0, b0), tour, selected_tour)
		tour_global[selected_tour] = new_tour
	### inter route operators
	else: # GLOBAL_RELOCATE or GLOBAL_EXCHANGE
		# get two distinct tours..
		selected_tour_1 = selected_tour_2 = rd.randint(0, len(tour_global)-1)
		while selected_tour_1 == selected_tour_2:
			selected_tour_2 = rd.randint(0, len(tour_global)-1)
			if len(tour_global[selected_tour_1]) <= 2: selected_tour_1 = rd.randint(0, len(tour_global)-1)
			if len(tour_global[selected_tour_2]) <= 2: selected_tour_2 = rd.randint(0, len(tour_global)-1)
		tour_1 = tour_global[selected_tour_1]
		tour_2 = tour_global[selected_tour_2]
		l_1 = len(tour_1)-2
		l_2 = len(tour_2)-2
		a0 = b0 = 0
		if l_1 > 0: a0 = rd.randint(0, l_1)
		if l_2 > 0: b0 = rd.randint(0, l_2)
		
		if operator == Operator.GLOBAL_RELOCATE:
			if l_1 == 0:
				if debug_output != Debug.OFF: print('This GLOBAL_RELOCATE is not possible, since tour ' + str(selected_tour_1) + ' cannot provide a node.')
				return tour_global, []
			new_tour_1, new_tour_2, touched = global_relocate(a0, b0, tour_1, tour_2, selected_tour_1, selected_tour_2)
		else: # = elif operator == Operator.GLOBAL_EXCHANGE:
			new_tour_1, new_tour_2, touched = global_exchange(a0, b0, tour_1, tour_2, selected_tour_1, selected_tour_2)

		tour_global[selected_tour_1] = new_tour_1
		tour_global[selected_tour_2] = new_tour_2

	return tour_global, touched

t = np.arange(0, 0, 1)
s = []
probs = []
selected_operator = []
applied_changes = []
improved_cost = []

## simulated annealing parameters
T = Tmax
lambd = math.log(Tmax/Tmin)

# draw initial tours / graphs
if show_gui != ShowGui.OFF:
	draw_tour_MA(tour_global,ms_tour,[])
	if(draw_legend):
		i_wp = Line2D([0],[0],marker='o', color=initial_color, linestyle='none')
		r1= Line2D([0],[0],marker='o', color=colors_path[0], linestyle='solid')
		r2= Line2D([0],[0],marker='o', color=colors_path[1], linestyle='solid')
		r3= Line2D([0],[0],marker='o', color=colors_path[2], linestyle='solid')
		ms= Line2D([0],[0],marker='.', color=colors_path[-1], linestyle='solid')
		ax1.legend([i_wp, r1, r2, r3, ms], ['initial waypoint', 'route robot 1', 'route robot 2', 'route robot 3', 'route mothership'], loc='upper center', bbox_to_anchor=(0.5, -0.23), ncol=2)
	graph = ax2.plot(t, s)
	graphProb = ax3.plot(probs, s)
	plt.pause(plot_pause*100)

counter_twoopt = counter_relocate = counter_global_relocate = counter_global_exchange = 0
counter_twoopt_improvement = counter_relocate_improvement = counter_global_relocate_improvement = counter_global_exchange_improvement = 0

operators = list(map(lambda x: x.value, Operator._member_map_.values()))
start_time = time.time()
for i in range(iterations):
	did_twoopt = did_relocate = did_global_relocate = did_global_exchange = False
	# permutate MURMELs route
	tour_global_copy = copy.deepcopy(tour_global)
	operator = Operator(rd.choices(operators, weights = operator_weights)[0])
	if debug_output == Debug.FULL: print('Randomly drawn operator: ' + str(operator))
	
	if operator == Operator.TWOOPT:
		new_tour_global, touched = do_permutation(tour_global_copy, Operator.TWOOPT)
		did_twoopt = True
	elif operator == Operator.RELOCATE:
		new_tour_global, touched = do_permutation(tour_global_copy, Operator.RELOCATE)
		if touched != []: did_relocate = True
	elif operator == Operator.GLOBAL_RELOCATE:
		new_tour_global, touched = do_permutation(tour_global_copy, Operator.GLOBAL_RELOCATE)
		did_global_relocate = True
	elif operator == Operator.GLOBAL_EXCHANGE:
		new_tour_global, touched = do_permutation(tour_global_copy, Operator.GLOBAL_EXCHANGE)
		did_global_exchange = True

	if debug_output == Debug.FULL: print('Before permutation: ' + str(tour_global))
	if debug_output == Debug.FULL: print('After permutation: ' + str(new_tour_global))

	# derive mothership route
	new_ms_tour = calc_ms_tour(new_tour_global)

	# calculate costs
	cost_new = tour_cost(new_tour_global, new_ms_tour)
	cost_change = cost_new[0] - cost_current[0]

	# cool down
	T = Tmax * math.exp(-lambd*i/iterations)

	if cost_change >= 0:
		prob = math.exp(-cost_change/T)
	else:
		prob = 1.0 # -> 100% chance to accept, if cost improved
	if debug_output == Debug.FULL: 
		print('SA: T: ' + str(round(T,2)) + ', delta cost: ' + str(round(cost_change,2)) + ', acceptance probability: ' + str(round(prob,2)) + ' [iteration: ' + str(i) + ']')
	if prob > rd.random() and (operator != Operator.GLOBAL_RELOCATE or did_global_relocate == True):
		cost_current = cost_new
		if debug_output != Debug.OFF: print('SA: Accepted new solution: Cost: ' + str(cost_new))
		tour_global = new_tour_global
		ms_tour = new_ms_tour
		if did_twoopt: counter_twoopt += 1
		if did_relocate: counter_relocate += 1
		if did_global_relocate: counter_global_relocate += 1
		if did_global_exchange: counter_global_exchange += 1
		if did_twoopt and cost_change < 0.0: counter_twoopt_improvement += 1
		if did_relocate and cost_change < 0.0: counter_relocate_improvement += 1
		if did_global_relocate and cost_change < 0.0: counter_global_relocate_improvement += 1
		if did_global_exchange and cost_change < 0.0: counter_global_exchange_improvement += 1
		applied_changes.append(True)

		if cost_current < cost_best: # keep the best solution; it might get worse afterwards
			tour_global_best = copy.deepcopy(tour_global)
			ms_tour_best = ms_tour
			cost_best = cost_current
	else:
		applied_changes.append(False)
	s.append(cost_current[0])
	selected_operator.append(operator)
	improved_cost.append(cost_change < 0)
	probs.append(T)
	if (show_gui == ShowGui.STEPS and applied_changes[-1]) or (show_gui != ShowGui.OFF and i == iterations-1):
		draw_tour_MA(tour_global, ms_tour, touched)
		t = np.arange(0, i+1, 1) # Numpy operations decrease speed, since for every iteration a C subroutine is called. Only do it if necessary for GUI or at the last iteration.
		graph_plot = graph.pop(0)
		graph_plot.remove()
		graph = ax2.plot(t, s, color='black')

		graph_plot = graphProb.pop(0)
		graph_plot.remove()
		graphProb = ax3.plot(t, probs, color='black')

		plt.pause(plot_pause) # important to draw changes
	
	if debug_output != Debug.OFF: print('')

end_time = time.time()
runtime = str(round((end_time - start_time), 2))
if replicate_paper:
	if filename=='monbijou-james-simon':
		tour_global_best=[[0,42,43,2,3,33,34,35,30,29,23], [0,11,10,6,5,8,7,27,26,25,22,21,18,15,16,17,40,50,49,46,44,45,47,48,41], [0,12,13,4,9,36,39,38,32,31,37,28,24,19,20,14,1]]
		ms_tour_best=calc_ms_tour(tour_global_best)
	if filename=='ch130':
		tour_global_best=[[0,44,10,4,111,124,89,58,29,84,65,77,22,9,101,5,73,51,8,55,64,102,80,88,93,109,97,42,106,126,33,99,115,23,94,78,3,41,43,36,70], [0,123,68,63,15,40,129,49,125,120,12,13,121,54,59,52,48,71,90,98,74,56,81,118,112,47,6,25,69,96,62,67,14,37,105,1,34,117,79,45,19,92,113,82,38], [0,60,108,75,127,27,114,116,32,21,46,39,50,119,57,91,72,100,122,110,76,35,83,31,24,85,87,128,103,28,11,86,18,26,30,16,53,95,66,20,17,105,7,2,61,104]]
		ms_tour_best=calc_ms_tour(tour_global_best)
if show_gui != ShowGui.OFF: draw_tour_MA(tour_global_best, ms_tour_best, [])

tmp_cost_function = cost_function
if cost_function == CostFunction.ENERGY_CONSUMPTION:
	other_cost_function = cost_function = CostFunction.TIME
else:
	other_cost_function = cost_function = CostFunction.ENERGY_CONSUMPTION
other_cost = tour_cost(tour_global_best, ms_tour_best)
cost_function = tmp_cost_function

print('Finished! Found tour with cost (' + str(cost_function) + '): ' + str(round(cost_best[0], 2)) + (' kWh' if cost_function == CostFunction.ENERGY_CONSUMPTION else ' s') + ' [Per MURMEL: ' + str([round(num, 2) for num in cost_best[1]]) + '] in ' + runtime + ' seconds.')
if debug_output != Debug.OFF:
	print('Tours (MURMEL): ' + str(tour_global_best))
	print('Tour (Mothership): ' + str(ms_tour_best))
	print('Number of two opt operations: ' + str(counter_twoopt))
	print('Number of relocate operations: ' + str(counter_relocate))
	print('Number of global relocate operations: ' + str(counter_global_relocate))
	print('Number of global exchange operations: ' + str(counter_global_exchange))
	print('Finished! Found tour with length ' + str(cost_current))

if show_gui != ShowGui.OFF:
	figures_dir = os.path.join(os.path.dirname(__file__), 'figures')
	for fileformat in ['.png','.pdf']:
		fig1.savefig(os.path.join(figures_dir, 'fig_' + filename + '_path' + fileformat), dpi=dpi, pad_inches=0.0, transparent=True)
		fig2.savefig(os.path.join(figures_dir, 'fig_' + filename + '_costs' + fileformat), dpi=dpi, pad_inches=0.0, transparent=True)
		fig3.savefig(os.path.join(figures_dir, 'fig_' + filename + '_temps' + fileformat), dpi=dpi, pad_inches=0.0, transparent=True)
	plt.show(block=True)
	plt.pause(plot_pause*100)
	plt.close()


## write results to file
if debug_output != Debug.OFF: print('Try to write results.csv and results_timeline.csv: ', end='')


header = ['file', 'initial', str(Operator(0)) + ' weight',  str(Operator(1)) + ' weight',  str(Operator(2)) + ' weight',  str(Operator(3)) + ' weight', 'iterations', 'target costfunction', 'cost (best solution)', 'cost (final solution)', 'costs per MURMEL (final solution)', 'cost (initial solution)', 'costs per MURMEL (initial solution)', 'costfunction 2 (no target)', 'cost 2 (final solution)', 'Tmin', 'Tmax', 'runtime (s)', 'two opts', 'two opts with improvement', 'relocates', 'relocates with improvement', 'global relocates', 'global relocates with improvement', 'global exchanges', 'global exchanges with improvement']
my_file = Path('results.csv')
header_needed = False
if not my_file.is_file():
	if debug_output != Debug.OFF: print('Need to add header to results.csv.', end='')
	header_needed = True
data = [filename + '.tsp', str(initial_sln), str(operator_weights[0]), str(operator_weights[1]), str(operator_weights[2]), str(operator_weights[3]), iterations, str(cost_function), str(cost_best[0]), str(cost_current[0]), str(cost_current[1]), str(initial_cost[0]), str(initial_cost[1]), str(other_cost_function), str(other_cost[0]), str(Tmin), str(Tmax), str(runtime), str(counter_twoopt), str(counter_twoopt_improvement), str(counter_relocate), str(counter_relocate_improvement), str(counter_global_relocate), str(counter_global_relocate_improvement), str(counter_global_exchange), str(counter_global_exchange_improvement)]
try:
	with open('results.csv', 'a', newline='\n') as results_file:
		writer = csv.writer(results_file)
		if header_needed:
			writer.writerow(header)
		writer.writerow(data)
	with open('results_timeline.csv', 'w', newline='\n') as results_file:
		writer = csv.writer(results_file)
		writer.writerow(['iteration', 'cost', 'selected operator', 'applied changes', 'improved cost'])
		for i in range(0, len(s)):
			writer.writerow([i, s[i], selected_operator[i], applied_changes[i], improved_cost[i]])
	if debug_output != Debug.OFF: print('successful!')
except Exception as e:
	if debug_output != Debug.OFF: print('error!\nExiting..') # str(e)
	sys.exit()
