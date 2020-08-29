import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

'''
3 NODE TRIANGULAR ELEMENT
x0 = array [x, y] pairs of points
xi = xi gauss point
eta = eta gauss point

returns: element shape matrix, element shape derivative matrix, jacobian (det)
'''
'''3 NODE TRIANGULAR ELEMENT'''
def T3(x0, xi, eta, dim='2D'):

    # Canonical shape functions (xi's and eta's)
    Nh = np.array([xi, eta, 1-xi-eta])
    Bh = np.array([[1, 0, -1], [0, 1, -1]])

    # Jacobian matrix
    J = Bh @ x0

    # Element matrices
    Je = la.det(J)

    if dim=='1D':
        return Nh, Bh, Je
    if dim=='0D':
        return np.array([1]), np.array([0]), Je


    # Derivative of shape functions
    dN = la.inv(J) @ Bh

    Ne = np.zeros((2, 6))
    Be = np.zeros((3, 6))
    for i in range(3):
        Ne[0, 2*i] = Nh[i]
        Ne[1, 2*i+1] = Nh[i]

        Be[0, 2*i] = dN[0, i]
        Be[2, 2*i] = dN[1, i]
        Be[1, 2*i+1] = dN[1, i]
        Be[2, 2*i+1] = dN[0, i]


    if dim=='2D':
        return Ne, Be, Je



'''4 NODE QUADRILATERAL ELEMENT'''
def Q4(x0, xi, eta, dim='2D'):

    # Canonical shape functions (xi's and eta's)
    Nh = 1/4*np.array([(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)])
    Bh = 1/4*np.array([[(eta-1), (-eta+1), (eta+1), (-1-eta)], [(xi-1), (-xi-1), (xi+1), (-xi+1)]])

    # Jacobian matrix
    J = Bh @ x0

    # Element matrices
    Je = la.det(J)

    if dim=='1D':
        return Nh, Bh, Je
    if dim=='0D':
        return np.array([1]), np.array([0]), Je


    # Derivative of shape functions
    dN = la.inv(J) @ Bh

    Ne = np.zeros((2, 8))
    Be = np.zeros((3, 8))
    for i in range(4):
        Ne[0, 2*i] = Nh[i]
        Ne[1, 2*i+1] = Nh[i]

        Be[0, 2*i] = dN[0, i]
        Be[2, 2*i] = dN[1, i]
        Be[1, 2*i+1] = dN[1, i]
        Be[2, 2*i+1] = dN[0, i]

    if dim=='2D':
        return Ne, Be, Je


'''GET CANONICAL POINTS IN ELEMENT'''
def canonical_points(elem):

    if elem==T3:
        return np.array([[1, 0], [0, 1], [0, 0]])

    #elif elem==T6:
        #return np.array([[1, 0], [0, 1], [0, 0], [.5, .5], [0, .5], [.5, 0]])

    elif elem==Q4:
        return np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])


    else:
        print('Element Type not Implemented')
        return np.array([[]])


'''
GET NODE OF AT (X,Y) COORDINATE
'''
def get_node(nodes, x, y):

    if y=='all':
        return np.where(nodes[:, 0]==x)[0]
    if x=='all':
        return np.where(nodes[:, 1]==y)[0]

    ix = np.where(nodes[:, 0]==x)[0]
    iy = np.where(nodes[:, 1]==y)[0]
    i = np.intersect1d(ix, iy)

    if len(i)==0:
        print('Point not in nodes')

    return i[0]


'''
ASSIGN A FIXITY AT A NODE

node = node # that is being fixed
fixity = string containing which directions to constrain ('x', 'y', or 'xy')
val = array values to constrain each: ex) [2] for 'x', or [0, 5] for 'xy'

returns:
u0 = array of known displacements
i0 = degrees of freedom of known displacements (based on node number)

'''
def assign_fixity(node, fixity, val=[]):
    if type(node) == np.ndarray:
        i0 = np.zeros((node.shape[0]*2), dtype=np.uint64)
        i0[0::2] = node*2
        i0[1::2] = node*2+1

        if len(val)==0:
            u0 = np.zeros(i0.shape)
        else:
            u0 = np.array(val)

        return u0, i0


    u0 = []
    i0 = []

    if 'x' in fixity:
        i0.append(int(node*2))

        if len(val) == 0:
            u0.append(0)

    if 'y' in fixity:
        i0.append(int(node*2+1))

        if len(val) == 0:
            u0.append(0)

    return u0, i0


'''
STIFFNESS K MATRIX FOR ELEMENT

x0 = coordinates of each node
elem = element type (T3, Q4, etc.), function
xi, eta, wi = gauss points and weight
'''
def stiffness(x0, elem, xi, eta, wi, E=0, v=0, mu=0, kap=0, k=0):

    # Element Shape Functions
    Ne, Be, Je = elem(x0, xi, eta)


    # Material Properties

    # predetermined stiffness
    if k!=0:
        De = k*np.eye(3)
    # input mu and kappa instead of E and v
    elif E==0:
        E = (9*kap*mu)/(3*kap+mu)
        v = (3*kap-2*mu)/(3*(2*kap+mu))
    De = E/((1+v)*(1-2*v)) * np.array([[1-v, v, 0], [v, 1-v, 0], [0, 0, (1-2*v)/2]])

    # Stiffness Matrix
    Ke = Be.T @ De @ Be * Je * wi

    return Ke


'''
POINT LOAD ON ELEMENT

x0 = coordinates of each node
nodes = global node indices
elem = element type (T3, Q4, etc.), function
P = [Px, Py] force vector
Pn = global node number vector is applied on
'''
def point_load(x0, nodes, elem, P, Pn):

    # Get canonical points of element
    can_points = canonical_points(elem)

    # Find corresponding (xi, eta) pair for node force is applied on
    [xi, eta] = can_points[np.where(nodes==Pn)[0][0]]

    # Get Shape Functions
    Ne, Be, Je = elem(x0, xi, eta)

    fT = Ne.T @ P

    return fT


'''
FINITE ELEMENT ANALYSIS ASSEMBLY (HYBRID FORMULATION)

nodes = [x, y] coordinates of each node
elements = array containing the nodes per each element element[i] = [nodes of i]
mu =
lam =
utype = element type function (T3, Q4, etc) of displacement discretization
qtype = element type function (T3, Q4, etc) of pressure (automatically set to constant pressure)
GP = array of gauss points to iterate through [(xi, eta, weight), ...]
P = array of force vectors [[Px, Py], ...]
Pn = array of nodes per each force []
Pe = array of elements associated with each force []


returns
K= global stiffness matrix
F= global force matrix

'''
def assemble_hybrid(nodes, elements, utype, qtype, GP, P, Pn, Pe, mu=1, lam=1):

    # Degrees of freedom
    nf = 2*nodes.shape[0]
    nq = elements.shape[0]
    dofG = np.arange(0, nf+nq)

    # Initialize Matrices
    Ku = np.zeros((nf, nf))
    G = np.zeros((nq, nf))
    M = np.zeros((nq, nq))
    f = np.zeros(nf)

    # Constant Matrices
    De = mu * np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    Y = np.array([1, 1, 0])


    for i, e in enumerate(elements):

        xe = nodes[e]

        # degrees of freedom for element
        dof_u = np.zeros((elements.shape[1]*2), dtype=np.uint64)
        dof_q = np.zeros((1), dtype=np.uint64)

        dof_u[0::2] = 2*e
        dof_u[1::2] = 2*e+1
        k_in = np.ix_(dof_u, dof_u)

        dof_q[0] = i
        m_in = np.ix_(dof_q, dof_q)
        g_in = np.ix_(dof_q, dof_u)

        # stiffness matrices
        for gp in GP:
            Ne, Be, Je = utype(xe, gp[0], gp[1])
            Nq, Bq, Jq = qtype(xe, gp[0], gp[1], dim='0D')

            Ge = Nq * Y @ Be *Je * gp[2]
            Me = 1/lam * Nq.T @ Nq * gp[2]
            Ke = Be.T @ De @ Be *Je* gp[2]

            # Assemble Global Matrices
            Ku[k_in] += Ke
            G[g_in] += Ge
            M[m_in] += Me

        # Point Load Matrix
        if i in Pe:
            ip = np.where(i==Pe)[0][0]
            fTe = point_load(xe, e, utype, P[ip], Pn[ip])
        else:
            fTe = np.zeros(dof_u.shape)

        f[dof_u] += fTe


    # Assemble Full K matrix
    K1 = np.concatenate((Ku, G.T), 1)
    K2 = np.concatenate((G, M), 1)
    K = np.concatenate((K1, K2), 0)

    fq = np.zeros(nq)
    F = np.concatenate((f, fq))

    return K, F


'''

'''
def assemble(nodes, elements, etype, GP, P, Pn, Pe, E=0, v=0, mu=0, kap=0):
    # Global matrices
    nf = nodes.shape[0]*2
    K = np.zeros((nf, nf))
    F = np.zeros(nf)
    U = np.zeros(nf)

    for i, e in enumerate(elements):

        # Get degrees of freedom for the element
        dof = np.zeros((e.shape[0]*2), dtype=np.uint64)
        dof[0::2] = e*2
        dof[1::2] = e*2+1
        e_in = np.ix_(dof, dof)

        # Geometry of element
        xe = nodes[e]

        # Stiffness Matrix
        for gp in GP:
            if E==0:
                Ke = stiffness(xe, etype, gp[0], gp[1], gp[2], mu=mu, kap=kap)
            else:
                Ke = stiffness(xe, etype, gp[0], gp[1], gp[2], E=E, v = v)
            K[e_in] += Ke


        # Point Load Matrix
        if i in Pe:
            ip = np.where(i==Pe)[0][0]
            fTe = point_load(xe, e, etype, P[ip], Pn[ip])
        else:
            fTe = np.zeros(dof.shape)

        F[dof] += fTe

    return K, F

'''
SOLVE SYSTEM FOR UNKNOWN DISPLACEMENTS AND UNKNOWN REACTIONS

K = global stiffness matrix
F = global force matrix
u0 = known displacement array
i0 = array of corresponding known degrees of freedom
'''
def solve_system(K, F, u0, i0):
    nf = F.shape[0]

    # Displacement Vector
    U = np.zeros(nf)
    U[i0] = u0

    # Known (A)/unknown (B) indices
    indices = np.arange(0, nf, 1)
    ia = np.array(i0)
    ib = np.setdiff1d(indices, ia)

    # Known degrees of freedom (u0)
    AA = np.ix_(ia, ia)

    # Unknown degrees of freedom
    BB = np.ix_(ib, ib)

    # Other quadrants
    AB = np.ix_(ia, ia)
    BA = np.ix_(ib, ia)

    # Divide the stiffness matrix
    Kaa = K[AA] # known u
    Kab = K[AB] #
    Kba = K[BA] #
    Kbb = K[BB] # unknown u

    # Divide applied force vector
    Fa = F[ia]
    Fb = F[ib]

    # Solve for unknown displacements

    ub = la.solve(Kbb, Fb - Kba @ u0)

    U[ia] = u0
    U[ib] = ub

    # Solve for unknown forces/reactions
    R = K @ U - F

    return U, R

'''
GENERATE A BAR SHAPE
L = length (x)
W = width (y)
dx = discretization (make them perectly divisible)
'''
def gen_bar(L, W, dx, etype='Q4'):
    if L/dx%1 != 0 or W/dx%1 !=0:
        print('dx not perfectly divisible')
        return 0, 0

    # Create mesh
    x = np.arange(0, L+dx, dx)
    y = np.arange(0, W+dx, dx)

    xv, yv = np.meshgrid(x, y)
    xv = xv.reshape((xv.shape[0]*xv.shape[1], 1))
    yv = yv.reshape((yv.shape[0]*yv.shape[1], 1))
    nodes = np.concatenate((xv, yv), axis=1)

    # Dimensions in terms of number of nodes
    nX = int(L/dx)
    nY = int(W/dx)

    # Create element connectivity
    if etype == 'Q4':
        elements = []
        for i in range(nY):
            for j in range(nX):
                elems = [j + i*(nX+1), j + i*(nX+1) + 1, j + (i+1)*(nX+1) +1, j + (i+1)*(nX+1) ]
                elements.append(elems)


    if etype=='T3':
        elements = []
        for i in range(nY):
            for j in range(nX):
                elem1 = [j + i*(nX+1), j + i*(nX+1) + 1, j + (i+1)*(nX+1)+1]
                elem2 = [j + (i+1)*(nX+1)+1, j + (i+1)*(nX+1), j + i*(nX+1) ]
                elements.append(elem1)
                elements.append(elem2)

    elements = np.array(elements)

    return nodes, elements
