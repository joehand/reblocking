import numpy as np
import networkx as nx
import math
import random
from scipy.cluster.hierarchy import linkage, dendrogram


import my_graph as mg


def distance(MyPoint0, MyPoint1):
    return np.sqrt((MyPoint0.x-MyPoint1.x)**2+(MyPoint0.y-MyPoint1.y)**2)

def distance_squared(MyPoint0, MyPoint1):
    return (MyPoint0.x-MyPoint1.x)**2+(MyPoint0.y-MyPoint1.y)**2

def distance_point_to_segment(target, myedge):
    n1 = myedge.nodes[0]
    n2 = myedge.nodes[1]

    if myedge.length == 0:
        dist = distance(target, n1)
    elif  target == n1 or target == n2:
        dist = 0
    else:
        px = float(n2.x - n1.x)
        py = float(n2.y - n1.y)
        u = float((target.x - n1.x)*px + (target.y - n1.y)*py)/(px*px + py*py)
        if u > 1:
            u = 1
        elif u < 0:
            u = 0
        x = n1.x + u*px
        y = n1.y + u*py

        dx = x - target.x
        dy = y - target.y

        dist = math.sqrt(dx * dx + dy * dy)
    return dist

def area(face):
    return 0.5*abs(sum(e.nodes[0].x*e.nodes[1].y - e.nodes[1].x*e.nodes[0].y for e in face.edges))


def centroid(face):

    a = 0.5*(sum(e.nodes[0].x*e.nodes[1].y - e.nodes[1].x*e.nodes[0].y for e in face.edges))
    if abs(a) < 0.01:
        cx = np.mean([n.x for n in face.nodes])
        cy = np.mean([n.y for n in face.nodes])
    else:
        cx = (1/(6*a))*sum(  [(e.nodes[0].x+e.nodes[1].x )*(e.nodes[0].x*e.nodes[1].y - e.nodes[1].x*e.nodes[0].y) for e in face.edges])
        cy = (1/(6*a))*sum(  [(e.nodes[0].y+e.nodes[1].y )*(e.nodes[0].x*e.nodes[1].y - e.nodes[1].x*e.nodes[0].y) for e in face.edges])

    centroid = mg.MyNode((cx,cy))
    return centroid


def myRoll(mylist):
    mylist.insert(0,mylist[-1])
    del mylist[-1]
    return(mylist)

def bisect_angle(a,b,c, epsilon = 0.2, radius = 1):
    ax = a.x - b.x
    ay = a.y - b.y

    cx = c.x - b.x
    cy = c.y - b.y

    a1 = mg.MyNode(((ax,ay))/np.linalg.norm((ax,ay)))
    c1 = mg.MyNode(((cx,cy))/np.linalg.norm((cx,cy)))

    #if vectors are close to parallel, find vector that is perpendicular to ab
    #if they arenot, then find the vector that bisects a and c
    if abs(np.cross(a1.loc,c1.loc)) < 0 + epsilon:
        #print "vectors {0} {1}  and {1} {2} are close to parallel".format(a,b,c)
        dx = -ay
        dy = ax
    else:
        dx = (a1.x+c1.x)/2
        dy = (a1.y+c1.y)/2

    #convert d values into a vector of length radius
    dscale = ((dx,dy)/np.linalg.norm((dx,dy)))*radius
    myd = mg.MyNode(dscale)

    #make d a node in space, not vector around b
    d = mg.MyNode((myd.x + b.x, myd.y + b.y))

    return d

def find_negative(d,b):
    """finds the vector -d when b is origen """
    negx = -1*(d.x - b.x)+ b.x
    negy = -1*(d.y- b.y)+ b.y
    dneg = mg.MyNode((negx, negy))
    return dneg


def WeightedPick(d):
    """picks an item out of the dictionary, with probability proportional to the
    value of that item.  e.g. in {a:1, b:0.6, c:0.4} selects and returns "a" 5/10
    times, "b" 3/10 times and "c" 2/10 times. """

    r = random.uniform(0,sum(d.itervalues()))
    s = 0.0
    for k,w in d.iteritems():
        s += w
        if r < s: return k
    return k


def mat_reorder(matrix, order):
    """sorts a square matrix so both rows and columns are ordered by "order" """

    Drow = [matrix[i] for i in order]
    Dcol = [[r[i] for i in order] for r in Drow]


    return Dcol


def path_length(path):
    """finds the geometric path length for a path that consists of a list of
    MyNodes. """
    length = 0
    for i in range(1,len(path)):
        length += distance(path[i-1],path[i])
    return length

def shorten_path(ptup):
    """ all the paths found in my pathfinding algorithm start at the fake road
    side, and go towards the interior of the parcel.  This method drops nodes
    beginning at the fake road node, until the first and only the first node
    is on a road.  This gets rid of paths that travel along a curb before ending."""

    while ptup[1].road == True:
        ptup = ptup[1:]
    return ptup


def add_fake_edges(myG,p, roads_only = False):

    if roads_only == True:
        for n in p.nodes:
            if n.road:
                newedge = mg.MyEdge((p.centroid, n))
                newedge.length = 0
                myG.add_edge(newedge)
    else:
        for n in p.nodes:
            newedge = mg.MyEdge((p.centroid, n))
            newedge.length = 0
            myG.add_edge(newedge)


def shortest_path_setup(myA, parcel):
    """ sets up graph to be ready to find the shortest path from a parcel to the road. """

    fake_interior = parcel.centroid
    add_fake_edges(myA,parcel)

    fake_road = mg.MyNode((305620,8022470))

    road_fences =  [i for i in myA.road_nodes if len(myA.G.neighbors(i)) > 2]

    for i in road_fences:
        newedge = mg.MyEdge((fake_road,i))
        newedge.length = 0
        myA.add_edge(newedge)
    return fake_interior, fake_road

def shortest_path_p2r(myA, parcel):
    """finds the shortest path along fenclines from a given interior parcel to the
    broader road system """

    fake_interior, fake_road = shortest_path_setup(myA,parcel)

    path = nx.shortest_path(myA.G,fake_road,fake_interior,"weight")
    length = nx.shortest_path_length(myA.G,fake_road,fake_interior,"weight")

    myA.G.remove_node(fake_interior)
    myA.G.remove_node(fake_road)

    return path[1:-1], length


def shortest_path_p2p(myA, p1, p2):
    """finds the shortest path along fenclines from a given interior parcel to the
    broader road system """

    add_fake_edges(myA,p1, roads_only = True)
    add_fake_edges(myA,p2, roads_only = True)

    path = nx.shortest_path(myA.G,p1.centroid,p2.centroid,"weight")
    length = nx.shortest_path_length(myA.G,p1.centroid,p2.centroid,"weight")

    myA.G.remove_node(p1.centroid)
    myA.G.remove_node(p2.centroid)

    return path[1:-1], length


def find_short_paths(myA, parcel):
    interior, road = shortest_path_setup(myA,parcel)

    shortest_path = nx.shortest_path(myA.G,interior,road,"weight")
    shortest_path_segments = len(shortest_path)
    shortest_path_distance = path_length(shortest_path[1:-1])
    all_simple = [shorten_path(p[1:-1]) for p in nx.all_simple_paths(myA.G,road,interior,cutoff = shortest_path_segments + 2)]
    paths ={tuple(p): path_length(p) for p in all_simple if path_length(p) < shortest_path_distance*2}

    myA.G.remove_node(road)
    myA.G.remove_node(interior)

    return paths

def find_short_paths_all_parcels(myA):
    all_paths = {}

    for p in myA.interior_parcels:
        paths = find_short_paths(myA,p)
        all_paths.update(paths)

    return all_paths


############################
### connectviity optimization
##############################


def shortest_path_p2p_matrix(myG, full = False):

    etup_drop = myG.find_interior_edges()
    if full == False:
        myG.G.remove_edges_from(etup_drop)
        #print "dropping {} edges".format(len(etup_drop))

    path_mat = []
    path_len_mat = []

    for p0 in myG.inner_facelist:
        path_vec = []
        path_len_vec = []
        add_fake_edges(myG,p0)

        for p1 in myG.inner_facelist:
            if p0.centroid == p1.centroid:
                length = 0
                path = p0.centroid
            else:
                add_fake_edges(myG,p1)
                try:
                    path = nx.shortest_path(myG.G,p0.centroid,p1.centroid, "weight")
                    length =path_length(path[1:-1])
                except:
                    path = []
                    length = np.nan
                myG.G.remove_node(p1.centroid)
            path_vec.append(path)
            path_len_vec.append(length)


        myG.G.remove_node(p0.centroid)
        path_mat.append(path_vec)
        path_len_mat.append(path_len_vec)

    return path_mat, path_len_mat

def difference_roads_to_fences(myG):
    fullpaths, fullpath_len = shortest_path_p2p_matrix(myG,full = True)
    S1 = myG.copy()
    paths, path_len = shortest_path_p2p_matrix(S1, full = False)

    diff = [[path_len[j][i] - fullpath_len[j][i] for i in range(0,len(fullpath_len[j]))] for j in range(0,len(fullpath_len))]

    dmax = max([max(i) for i in diff])
    #print dmax

    for j in range(0,len(path_len)):
        for i in range(0,len(path_len[j])):
            if np.isnan(path_len[j][i]):
                diff[j][i] = dmax + 150
                #            diff[j][i] = np.nan


    totaltravel = sum([max(i) for i in path_len])

    return diff, fullpath_len, path_len, totaltravel


def __temp_select_roads(myG, diff, parcelmax, maxdiff):
    p1index = parcelmax.index(maxdiff)
    p2index = diff[p1index].index(maxdiff)

    p1 = myG.inner_facelist[p1index]
    p2 = myG.inner_facelist[p2index]


    path, length = shortest_path_p2p(myG,p1,p2)
    #print path
    edge = myG.G[path[0]][path[1]]["myedge"]

    return edge

def road_nearest_centroid(myG):

    c = myG.outerface.centroid
    distsq = {n:distance_squared(c,n) for n in myG.road_nodes}
    closest = min(distsq,key = distsq.get)

    return closest

def shortest_distances(myG,start,nodes = None, offroad = True):
    mycopy = myG.copy()

    if not nodes:
        nodes = [n for n in mycopy.road_nodes if len(mycopy.G[n]) > 2]

    if offroad == False:
        etup_drop = mycopy.find_interior_edges()
        mycopy.G.remove_edges_from(etup_drop)

    dist = {n:nx.shortest_path_length(mycopy.G, start, n, weight = 'weight')  for n in nodes}

    return dist


def bisecting_path_endpoints(myG):
    start = road_nearest_centroid(myG)
    offroad_distances = shortest_distances(myG,start, offroad = True)
    onroad_distances = shortest_distances(myG,start, offroad = False)
    distdiff = {n:(onroad_distances[n]-offroad_distances[n]) for n in onroad_distances}
    tol2 = 1.1
    offroad_filter = {k:v for (k,v) in offroad_distances.items() if distdiff[k]*tol2 >=  max(distdiff.values())}
    finish = min(offroad_filter, key = offroad_filter.get)

    return start, finish

def build_path(myG,start,finish):
    ptup = nx.shortest_path(myG.G,start,finish, weight = "weight")

    ptup = shorten_path(ptup)
    ptup.reverse()
    ptup = shorten_path(ptup)


    myedges = [ myG.G[ptup[i-1]][ptup[i]]["myedge"]  for i in range(1,len(ptup))]

    for e in myedges:
        myG.add_road_segment(e)


    return ptup,myedges



################
## GRAPH INSTANTIATION
###################

def graphFromMyFaces(flist,name=None):
    myG = mg.MyGraph(name=name)
    for f in flist:
        for e in f.edges:
            myG.add_edge(e)
    return myG

def graphFromShapes(shapes, name):
    nodedict = dict()
    plist = []
    for s in shapes:
        nodes = []
        for k in s.points:
            myN = mg.MyNode(k)
            if not myN in nodedict:
                nodes.append(myN)
                nodedict[myN] = myN
            else:
                nodes.append(nodedict[myN])
            plist.append(mg.MyParcel(nodes))

    myG = mg.MyGraph(name = name)

    for p in plist:
        for e in p.edges:
            myG.add_edge(mg.MyEdge(e.nodes))

    print "data loaded"

    return myG


####################
## PLOTTING FUNCTIONS
####################

def plot_cluster_mat(distance_difference,distance,title,dmax):
    """from http://nbviewer.ipython.org/github/OxanaSachenkova/hclust-python/blob/master/hclust.ipynb    """

    fig = plt.figure(figsize=(8,8))
    # x ywidth height
    ax1 = fig.add_axes([0.05,0.1,0.2,0.6])
    Y = linkage(distance, method='single')
    Z1 = dendrogram(Y, orientation='right') # adding/removing the axes
    ax1.set_xticks([])
                                            #ax1.set_yticks([])

    # Compute and plot second dendrogram.
    ax2 = fig.add_axes([0.3,0.75,0.6,0.1])
    Z2 = dendrogram(Y)
    #ax2.set_xticks([])
    ax2.set_yticks([])

    #set up custom color map
    c = mcolors.ColorConverter().to_rgb
    seq = [c('navy'),c('mediumblue'),.1,c('mediumblue'), c('darkcyan'),.2,c('darkcyan'),c('darkgreen'),.3, c('darkgreen'),c('lawngreen'),.4,c('lawngreen'), c('yellow'),.5,c('yellow'),c('orange'),.7, c('orange'),c('red'), 0.99, c('black')]
    custommap = make_colormap(seq)


    #Compute and plot the heatmap
    axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
    idx1 = Z1['leaves']
    D = mat_reorder(distance_difference,idx1)
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap = custommap,vmin=0, vmax=dmax)
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])

    # Plot colorbar.
    axcolor = fig.add_axes([0.91,0.1,0.02,0.6])
    plt.colorbar(im, cax=axcolor)
    ax2.set_title(title)


def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


####################
## Testing functions
###################


def test_edges_equality():
    testG = testGraph()
    testG.trace_faces()
    outerE = list(testG.outerface.edges)[0]
    return outerE is testG.G[outerE.nodes[0]][outerE.nodes[1]]['myedge']


def test_weak_duals():
    S0 = testGraph()
    S1 = S0.weak_dual()
    S2 = S1.weak_dual()
    S3 = S2.weak_dual()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    S0.plot(ax=ax, node_color='b', edge_color='k', node_size=300)
    S1.plot(ax=ax, node_color='g', edge_color='b', node_size=200)
    S2.plot(ax=ax, node_color='r', edge_color='g', node_size=100)
    S3.plot(ax=ax, node_color='c', edge_color='r', node_size=50)
    ax.legend()
    ax.set_title("Test Graph")
    plt.show()


def test_nodes(n1,n2):
    eq_num = len(set(n1).intersection(set(n2)))
    is_num = len(set([id(n) for n in n1]).intersection(set([id(n) for n in n2])))
    print "is eq? ", eq_num, "is is? ", is_num


def testGraph():
    n = {}
    n[1]=mg.MyNode((0,0))
    n[2]=mg.MyNode((0,1))
    n[3]=mg.MyNode((0,2))
    n[4]=mg.MyNode((0,3))
    n[5]=mg.MyNode((1,2))
    n[6]=mg.MyNode((1,3))
    n[7]=mg.MyNode((0,4))
    n[8]=mg.MyNode((-1,4))
    n[9]=mg.MyNode((-1,3))
    n[10]=mg.MyNode((-1,2))
    n[11]=mg.MyNode((1,4))
    n[12]=mg.MyNode((-2,3))

    lat = mg.MyGraph(name="S0")
    lat.add_edge(mg.MyEdge((n[1],n[2])))
    lat.add_edge(mg.MyEdge((n[2],n[3])))
    lat.add_edge(mg.MyEdge((n[2],n[5])))
    lat.add_edge(mg.MyEdge((n[3],n[4])))
    lat.add_edge(mg.MyEdge((n[3],n[5])))
    lat.add_edge(mg.MyEdge((n[3],n[9])))
    lat.add_edge(mg.MyEdge((n[4],n[5])))
    lat.add_edge(mg.MyEdge((n[4],n[6])))
    lat.add_edge(mg.MyEdge((n[4],n[7])))
    lat.add_edge(mg.MyEdge((n[4],n[8])))
    lat.add_edge(mg.MyEdge((n[4],n[9])))
    lat.add_edge(mg.MyEdge((n[5],n[6])))
    lat.add_edge(mg.MyEdge((n[6],n[7])))
    lat.add_edge(mg.MyEdge((n[7],n[8])))
    lat.add_edge(mg.MyEdge((n[8],n[9])))
    lat.add_edge(mg.MyEdge((n[9],n[10])))
    lat.add_edge(mg.MyEdge((n[3],n[10])))
    lat.add_edge(mg.MyEdge((n[2],n[10])))
    lat.add_edge(mg.MyEdge((n[7],n[11])))
    lat.add_edge(mg.MyEdge((n[6],n[11])))
    lat.add_edge(mg.MyEdge((n[10],n[12])))
    lat.add_edge(mg.MyEdge((n[8],n[12])))

    return lat


def testGraphLattice():
    n = {}
    n[1]=mg.MyNode((0,0))
    n[2]=mg.MyNode((0,10))
    n[3]=mg.MyNode((0,25))
    n[4]=mg.MyNode((0,30))
    n[5]=mg.MyNode((0,40))
    n[6]=mg.MyNode((13,0))
    n[7]=mg.MyNode((13,10))
    n[8]=mg.MyNode((13,25))
    n[9]=mg.MyNode((13,30))
    n[10]=mg.MyNode((13,40))
    n[11]=mg.MyNode((22,0))
    n[12]=mg.MyNode((22,10))
    n[13]=mg.MyNode((22,25))
    n[14]=mg.MyNode((22,30))
    n[15]=mg.MyNode((22,40))
    n[16]=mg.MyNode((31,0))
    n[17]=mg.MyNode((31,10))
    n[18]=mg.MyNode((31,25))
    n[19]=mg.MyNode((31,30))
    n[20]=mg.MyNode((31,40))
    n[21]=mg.MyNode((40,0))
    n[22]=mg.MyNode((40,10))
    n[23]=mg.MyNode((40,25))
    n[24]=mg.MyNode((40,30))
    n[25]=mg.MyNode((40,40))

    lat = mg.MyGraph(name="S0")
    lat.add_edge(mg.MyEdge((n[1],n[2])))
    lat.add_edge(mg.MyEdge((n[1],n[6])))
    lat.add_edge(mg.MyEdge((n[2],n[3])))
    lat.add_edge(mg.MyEdge((n[2],n[7])))
    lat.add_edge(mg.MyEdge((n[3],n[4])))
    lat.add_edge(mg.MyEdge((n[3],n[8])))
    lat.add_edge(mg.MyEdge((n[4],n[5])))
    lat.add_edge(mg.MyEdge((n[4],n[9])))
    lat.add_edge(mg.MyEdge((n[5],n[10])))
    lat.add_edge(mg.MyEdge((n[6],n[7])))
    lat.add_edge(mg.MyEdge((n[6],n[11])))
    lat.add_edge(mg.MyEdge((n[7],n[8])))
    lat.add_edge(mg.MyEdge((n[7],n[12])))
    lat.add_edge(mg.MyEdge((n[8],n[9])))
    lat.add_edge(mg.MyEdge((n[8],n[13])))
    lat.add_edge(mg.MyEdge((n[9],n[10])))
    lat.add_edge(mg.MyEdge((n[9],n[14])))
    lat.add_edge(mg.MyEdge((n[10],n[15])))
    lat.add_edge(mg.MyEdge((n[11],n[12])))
    lat.add_edge(mg.MyEdge((n[11],n[16])))
    lat.add_edge(mg.MyEdge((n[12],n[13])))
    lat.add_edge(mg.MyEdge((n[12],n[17])))
    lat.add_edge(mg.MyEdge((n[13],n[14])))
    lat.add_edge(mg.MyEdge((n[13],n[18])))
    lat.add_edge(mg.MyEdge((n[14],n[15])))
    lat.add_edge(mg.MyEdge((n[14],n[19])))
    lat.add_edge(mg.MyEdge((n[15],n[20])))
    lat.add_edge(mg.MyEdge((n[15],n[20])))
    lat.add_edge(mg.MyEdge((n[16],n[17])))
    lat.add_edge(mg.MyEdge((n[16],n[21])))
    lat.add_edge(mg.MyEdge((n[17],n[18])))
    lat.add_edge(mg.MyEdge((n[17],n[22])))
    lat.add_edge(mg.MyEdge((n[18],n[19])))
    lat.add_edge(mg.MyEdge((n[18],n[23])))
    lat.add_edge(mg.MyEdge((n[19],n[20])))
    lat.add_edge(mg.MyEdge((n[19],n[24])))
    lat.add_edge(mg.MyEdge((n[20],n[25])))
    lat.add_edge(mg.MyEdge((n[21],n[22])))
    lat.add_edge(mg.MyEdge((n[22],n[23])))
    lat.add_edge(mg.MyEdge((n[23],n[24])))
    lat.add_edge(mg.MyEdge((n[24],n[25])))

    return lat,n

def testGraphEquality():
    n = {}
    n[1]=mg.MyNode((0,0))
    n[2]=mg.MyNode((0,1))
    n[3]=mg.MyNode((1,1))
    n[4]=mg.MyNode((1,0))
    n[5]=mg.MyNode((0,0)) # actually equal
    n[6]=mg.MyNode((0.0001,0.0001)) #within rounding
    n[7]=mg.MyNode((0.1,0.1)) #within threshold
    n[8]=mg.MyNode((0.3,0.3)) #actually different

    G = mg.MyGraph(name="S0")
    G.add_edge(mg.MyEdge((n[1],n[2])))
    G.add_edge(mg.MyEdge((n[2],n[3])))
    G.add_edge(mg.MyEdge((n[3],n[4])))
    G.add_edge(mg.MyEdge((n[4],n[5])))

    return G,n


def __centroid_test():
    n = {}
    n[1]=mg.MyNode((0,0))
    n[2]=mg.MyNode((0,1))
    n[3]=mg.MyNode((1,1))
    n[4]=mg.MyNode((1,0))
    n[5]=mg.MyNode((0.55,0))
    n[6]=mg.MyNode((0.5,0.9))
    n[7]=mg.MyNode((0.45,0))
    n[8]=mg.MyNode((0.4,0))
    n[9]=mg.MyNode((0.35,0))
    n[10]=mg.MyNode((0.3,0))
    n[11]=mg.MyNode((0.25,0))
    nodeorder = [1,2,3,4,5,6,7,8,9,10,11,1]
    nodetups = [(n[nodeorder[i]],n[nodeorder[i+1]]) for i in range(0,len(nodeorder)-1)]
    edgelist = [mg.MyEdge(i) for i in nodetups]

    f1  = mg.MyFace(nodetups)
    S0 = graphFromMyFaces([f1])

    S0.define_roads()
    S0.define_interior_parcels()

    S0.plot_roads(parcel_labels = True)


    return S0, f1, n,edgelist


def testmat():
    testmat = []
    dim = 4
    for i in range(0,dim):
        k = []
        for j in range(0,dim):
            k.append((i-j)*(i-j))
        testmat.append(k)
    return testmat
