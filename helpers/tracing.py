import numpy as np
import cv2

def track_edges(img):
    edges = list()
    edges_cells = list()
    edges_nodes = list()
    endpoints = list()
    
    (ih, iw) = img.shape
    
    ce = None
    cp = np.array([-1,-1], np.int)
    Q0 = np.zeros(4, np.int)
    P0 = np.array([[0,-1],[ 0,0],[-1, 0],[-1,-1]], np.int)
    R0 = np.array([[0, 1],[-1,0],[ 0,-1],[ 1, 0]], np.int)
    
    nodes_coords = list()
    nodes_cells  = list()
    nodes_edges  = list()
    
    def create_end_node(cp, ce_n, cells):
#         print(tuple(cp))
        nodes_coords.append( tuple(cp) )
        nodes_cells.append(cells)
        nodes_edges.append( [ce_n] )
        return len(nodes_coords)-1
    
    def track_edge(cp, d, start_node):
        P = np.roll(P0, -d, axis=0)
        R = np.roll(R0, -d, axis=0)
        Q = np.zeros(4, np.int)
        
        ce_n = len(edges)
        nodes_edges[start_node].append(ce_n)
        
        ce = [ cp.copy() ]
        edges.append( ce )
        cp += R[0]
        
        Q[0] = img[ tuple(cp + P[0]) ]
        Q[3] = img[ tuple(cp + P[3]) ] 
        edges_cells.append( (Q[0], Q[3]) )
        
        while True:
            # bumped into image edge
            if cp[0] == 0 or cp[0] == ih or cp[1] == 0 or cp[1] == iw:
                ce.append( cp.copy() )
                end_node = create_end_node(cp, len(edges)-1, [Q[0],Q[3]] )
                edges_nodes.append( (start_node, end_node) )
                return
            else:
                Q[1] = img[ tuple(cp + P[1]) ]
                Q[2] = img[ tuple(cp + P[2]) ]
                if   Q[1] == Q[0] and Q[2] == Q[3]:
                    cp += R[0]
                    pass
                elif Q[1] == Q[0] and Q[2] == Q[0]:
                    ce.append( cp.copy() )
                    d += 1
                    R = np.roll(R, -1, axis=0)
                    P = np.roll(P, -1, axis=0)
                    cp += R[0]
                elif Q[1] == Q[3] and Q[2] == Q[3]:
                    ce.append( cp.copy() )
                    d -= 1
                    R = np.roll(R, 1, axis=0)
                    P = np.roll(P, 1, axis=0)
                    cp += R[0]
                else:
                    ce.append( cp.copy() )
                    node_c = tuple(cp)
#                     print( node_c )
                    if node_c not in nodes_coords:
                        end_node = len(nodes_coords)
                        
                        nodes_coords.append( node_c )
                        ncells = []
                        nodes_cells.append( ncells )
                        nedges = [len(edges_cells)-1]
                        nodes_edges.append( nedges )
                        
                        edges_nodes.append( (start_node, end_node) )
                        if Q[1] == Q[2]:
                            ncells += [Q[1],Q[3],Q[0]]
                            track_edge( cp.copy(), (d+1) % 4 , end_node )
                            if len(nodes_edges[end_node])<3:
                                track_edge( cp.copy(), (d-1) % 4 , end_node )
                        elif Q[0] == Q[1]:
                            ncells += [Q[2],Q[3],Q[0]]
                            track_edge( cp.copy(), (d+1) % 4 , end_node )
                            if len(nodes_edges[end_node])<3:
                                track_edge( cp.copy(), ( d ) % 4 , end_node ) 
                        elif Q[2] == Q[3]:
                            ncells += [Q[1],Q[3],Q[0]]
                            track_edge( cp.copy(), ( d ) % 4 , end_node )
                            if len(nodes_edges[end_node])<3:
                                track_edge( cp.copy(), (d-1) % 4 , end_node )                        
                        else:
                            ncells += [Q[1],Q[2],Q[3],Q[0]]
                            track_edge( cp.copy(), (d+1) % 4 , end_node )
                            if len(nodes_edges[end_node])<3:
                                track_edge( cp.copy(), ( d ) % 4 , end_node ) 
                            if len(nodes_edges[end_node])<4:
                                track_edge( cp.copy(), (d-1) % 4 , end_node )                        
                            # raise Exception( "Bad node", cp )
                    else:
                        edges_nodes.append( (start_node, nodes_coords.index(node_c) ) )
                    return
#                 print( cp )
#             time.sleep(0.01)
    #find starting point
    
    cp = np.array([1,0], np.int)
    corners = set( {(ih, 0), (ih, iw), (0, iw) } )
    P = np.roll( P0, 1, axis=0)
    R = np.roll( R0, 1, axis=0)
    # Q = img[0, 0]
    d = -1
    while tuple( cp ) != (0,0):
        tcp = tuple( cp )
        if tcp in corners:
            d += 1
            P = np.roll(P, -1, axis=0)
            R = np.roll(R, -1, axis=0)
        elif img[ tuple(cp+P[2]) ]  != img[ tuple(cp+P[3]) ]:
            if tcp not in nodes_coords:
                nodes_coords.append( tcp )
                nodes_edges.append( [] )
                nodes_cells.append( [ img[ tuple(cp+P[2]) ] , img[ tuple(cp+P[3])] ] )
                track_edge( np.array(cp, np.int), (d+1) % 4, len(nodes_coords)-1 )
        cp += R[0]
        
    return {
        'edges_coords': [ np.array(e, np.int) for e in edges ],
        'edges_cells' : edges_cells,
        'edges_nodes' : edges_nodes,
        'nodes_coords': nodes_coords,
        'nodes_cells' : nodes_cells,
        'nodes_edges' : nodes_edges
    }

import mahotas as mh


def get_masks(img, ext_shape):
    masks = np.zeros( ext_shape, np.uint8 )
    for i in range(ext_shape[1]):
        for j in range(ext_shape[2]):
            masks[img[i,j],i,j] = 1
    return masks

def create_recolor_map(mpd1, mpd2, ump1, ump2):
    n2 = max( max(mpd2,default=-1),max(ump2,default=-1) ) + 1
    m1 = np.array( mpd1 + ump1, np.int )
    m2 = np.array( mpd2 + list( range(n2, n2+len(ump1)) ), np.int )
    return m2[m1.argsort()]

def get_torn_colors(torn_watershed):
    return np.sort(
           np.unique( 
               torn_watershed.flatten()
           ) )

def squeeze_colors(torn_watershed, colors):
    cmap = dict( (c,i) for i,c in enumerate(colors))
    return np.vectorize(lambda v: cmap[v])(torn_watershed)
    

flash_area_threshold = 0.9

def get_map(dst, src):
    n1 = np.max(dst)+1
    n2 = np.max(src)+1
    M = np.zeros( (n1, n2) )
    U = np.zeros( (n1, n2), np.uint32 )
    D = np.zeros( (n1, n2), np.uint32 )


    H, W = src.shape

    for i in range(H):
        for j in range(W):
            M[ dst[i,j], src[i,j] ] = 1
    # dst[:] = -1-dst[:]

    ms1 = get_masks( dst, (n1, H, W) )
    ms2 = get_masks( src, (n2, H, W) )
    for i in range(n1):
        for j in range(n2):
            if M[i,j]>1e-7:
                U[i,j] = np.sum( ms1[i]*ms2[j] )
                D[i,j] = np.sum((ms1[i]-ms2[j])**2)
                M[i,j] = U[i,j]/D[i,j] if D[i,j]>0 else np.inf

    (mpd1, mpd2) = [list(e) for e in np.where(M>3)]
    ump1 =       set(range(n1)) - set(mpd1) 
    ump2 = list( set(range(n2)) - set(mpd2) )

    A1 = np.zeros( n1 )
    A2 = np.zeros( n2 )

    ump1_flashes = set()
    # for i in ump1:
    #     jxs, = M[i,:].nonzero()
    #     if len(jxs) > 0:
    #         area = ms1[i].sum()
    #         fracs = np.array( [U[i,j]/area for j in jxs] )
    #         k = np.array( fracs ).argmax()
    #         if fracs[k] > 0.9:   #detected flash
    #             ump1_flashes.add( i )
    #             mpd1.append( i )
    #             mpd2.append( jxs[k] )

    ump1 = list(ump1 - ump1_flashes)

    M1 = M[ump1][:,ump2]

    # print( np.array([mpd1,mpd2]) )
    # print( np.array([ump1,ump2]) )
    while len(ump1)*len(ump2)>0:
        arg = M1.argmax()
        if(M1.max() < 1e-5):
            break
        (i, j) = np.unravel_index(arg, M1.shape)
        
        # border missed
        # print( np.sum(ms1[i]), np.sum(ms2[j]), U[i,j]  )
        
        
        mpd1.append( ump1.pop(i) )
        mpd2.append( ump2.pop(j) )
        M1 = np.delete(np.delete(M1, [i], axis=0), [j], axis=1)

    return ((mpd1,mpd2),(ump1,ump2))