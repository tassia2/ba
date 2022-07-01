#!/usr/bin/python

import os
from collections import defaultdict
#import msvcrt as m

def simple_cycles(G):
    # Yield every elementary cycle in python graph G exactly once
    # Expects a dictionary mapping from vertices to iterables of vertices
    def _unblock(thisnode, blocked, B):
        stack = set([thisnode])
        while stack:
            node = stack.pop()
            if node in blocked:
                blocked.remove(node)
                stack.update(B[node])
                B[node].clear()
    G = {v: set(nbrs) for (v,nbrs) in G.items()} # make a copy of the graph
    sccs = strongly_connected_components(G)
    while sccs:
        scc = sccs.pop()
        startnode = scc.pop()
        path=[startnode]
        blocked = set()
        closed = set()
        blocked.add(startnode)
        B = defaultdict(set)
        start_connections = G.get(startnode)
        if start_connections != None:
            stack = [ (startnode,list(start_connections)) ]
        while stack:
            thisnode, nbrs = stack[-1]
            if nbrs:
                nextnode = nbrs.pop()
                if nextnode == startnode:
                    yield path[:]
                    closed.update(path)
                elif nextnode not in blocked:
                    path.append(nextnode)
                    cur_connections = G.get(nextnode)
                    if cur_connections != None:
                        stack.append( (nextnode,list(cur_connections)) )
                    closed.discard(nextnode)
                    blocked.add(nextnode)
                    continue
            if not nbrs:
                if thisnode in closed:
                    _unblock(thisnode,blocked,B)
                else:
                    for nbr in G[thisnode]:
                        if thisnode not in B[nbr]:
                            B[nbr].add(thisnode)
                stack.pop()
                path.pop()
        remove_node(G, startnode)
        H = subgraph(G, set(scc))
        sccs.extend(strongly_connected_components(H))

def strongly_connected_components(graph):
    # Tarjan's algorithm for finding SCC's
    # Robert Tarjan. "Depth-first search and linear graph algorithms." SIAM journal on computing. 1972.
    # Code by Dries Verdegem, November 2012
    # Downloaded from http://www.logarithmic.net/pfh/blog/01208083168

    index_counter = [0]
    stack = []
    lowlink = {}
    index = {}
    result = []
    
    def _strong_connect(node):
        index[node] = index_counter[0]
        lowlink[node] = index_counter[0]
        index_counter[0] += 1
        stack.append(node)
    
        #print (node)
        successors = graph.get(node)
        #successors = graph[node]
        #print (successors)
        
        if successors != None:
            for successor in successors:
                if successor not in index:
                    _strong_connect(successor)
                    lowlink[node] = min(lowlink[node],lowlink[successor])
                elif successor in stack:
                    lowlink[node] = min(lowlink[node],index[successor])

        if lowlink[node] == index[node]:
            connected_component = []

            while True:
                successor = stack.pop()
                connected_component.append(successor)
                if successor == node: break
            result.append(connected_component[:])
    
    for node in graph:
        if node not in index:
            _strong_connect(node)
    
    return result

def remove_node(G, target):
    # Completely remove a node from the graph
    # Expects values of G to be sets
    tmp = G.get(target)
    if tmp != None:
        del G[target]
    for nbrs in G.values():
        nbrs.discard(target)

def subgraph(G, vertices):
    # Get the subgraph of G induced by set vertices
    # Expects values of G to be sets
    return {v: G[v] & vertices for v in vertices}

##example:
#graph = {0: [7, 3, 5], 1: [2], 2: [7, 1], 3: [0, 5], 4: [6, 8], 5: [0, 3, 7], 6: [4, 8], 7: [0, 2, 5, 8], 8: [4, 6, 7]}
#print(tuple(simple_cycles(graph)))

graph = {}
line_off = 10

# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk("../src/"):
    path = root.split(os.sep)
    print((len(path) - 1) * '---', os.path.basename(root))
    if root == '../src/linear_algebra/lmp':
        continue

    for file in files:
        f_name, f_ext = os.path.splitext(file)
        filename = root + '/' + file
        cur_header = filename[7:]
        if f_ext == '.h':
            #print(filename)
            #print(cur_header)
            includes = []
            
            with open(filename) as f:
                for line in f:
                    if line[0:line_off] == '#include "':
                        included_header = "void"
                        counter = line_off
                        for c in line[line_off:]:
                            #print (c)
                            counter = counter + 1
                            if c == '"':
                                included_header = line[line_off:counter-1]
                                break
                        #print(line)
                        #print(included_header)
                        includes.append(included_header)
            
            graph[cur_header] = includes
            #print cur_header
            #print includes
            #text = raw_input("")

print ('---')

print(tuple(simple_cycles(graph)))
