import shapefile
from operator import attrgetter
import networkx as nx


import my_graph_helpers as mgh

def import_and_setup(component, shp_file):
    """Imports the data from a shapefile, and returns a component.
    Component 0 is the largest connected component (block) in
    the neighborhood's graph.    """

    sf = shapefile.Reader(shp_file)
    myG = mgh.graphFromShapes(sf.shapes(), "Before")


    #nodes within a distance threshold of each other are combined.
    threshold = 0.5
    myG = myG.clean_up_geometry(threshold)

    connected = myG.connected_components()

    S0 = connected[component]

    return S0

def best_of_n(myG,alpha,n,s):
    """Uses the probablistic greedy algorithm to connect all parcels to roads;
    returns the n best outcomes found in s attempts."""

    keep_list= []

    min_length = 1000
    tot_steps = 0

    while tot_steps < s:
        S0 = myG.copy()
        new_roads_i = S0.build_all_roads(alpha)
        keep_list.append(S0)
        if new_roads_i < min_length:
            min_length = new_roads_i
        tot_steps += 1

    keep_list.sort(key = attrgetter('added_roads'))

    plotting = False
    if plotting == True:
        for g in keep_list[0:n]:
            title = "Length of new roads is %.1f"%(g.added_roads)
            g.plot_roads(title = title)
            fig_name = "CC{}_{}best".format(component,keep_list.index(g))
            plt.savefig(fig_name, bbox_inches = 'tight')

    print "total interations is {}, best outcome is {}".format(tot_steps,keep_list[0].added_roads)
    return keep_list[0:n]

    plt.show()

def reblocked_JSON(shp_file):

    component = 0

    master = import_and_setup(component, shp_file)

    S0 = master.copy()

    ##build bisecting path, then probablistic roads

    S0.define_roads()
    S0.define_interior_parcels()

    start, finish = mgh.bisecting_path_endpoints(S0)
    ptup,myedges = mgh.build_path(S0,start,finish)
    new_path = mgh.path_length(ptup)

    new_roads_i = S0.build_all_roads(alpha = 2)

    built_road_len = new_path + new_roads_i

    return S0.myedges_geoJSON()
