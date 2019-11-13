from dh import DH
import numpy as np
import sympy as sp

q1, q2, q3 = sp.symbols('q1, q2, q3')
#assumption d1 and a
d1=2
a2=2
# theta_rad, d, alpha, a
DH_table = [
[q1*sp.pi/180, d1, -90*sp.pi/180, 0],
[q2*sp.pi/180, 0,  -90*sp.pi/180,  0],
[0, a2+q3, 0, 0]
]
joint_types=["R","R","linear"]

# TODO jacobians
fk = DH() #object from DH class
F_M = fk.Robot_matrix(DH_table) #forward matrix

#jacobian matrix size 6x3
jacobian = np.ones((6,1))
#print(jacobian.shape)

# jacobian column
def jacobian_col (type,joint):
    H_t = fk.homogenous(DH_table, joint-1) #homogenous transform
    if type == "linear":
        if joint == 0:
            R_M = np.identity(3) #rotation matrix
        else:
            R_M = H_t[0:3,0:3]
        j_v=np.dot(R_M,np.array([[0],[0],[1]]))
        j_w = np.array([0,0,0]).reshape(3,1)
    else: # revolute
        if joint ==0:
            R_M = np.identity(3) #rotation matrix
            d_i = np.array([0,0,0]).reshape(3,1) # d(0,0)
        else:
            R_M = H_t[0:3,0:3]
            d_i = H_t[:3,3] # d(0,i)
            d_e = F_M[:3,3] #d(0,n) end_effector
        R = np.dot(R_M,np.array([[0],[0],[1]]))
        #print("shape_R ", R.shape)
        d = (d_e - d_i).reshape(3,1)
        #print("shape_d ", d.shape)
        j_v = np.cross(R,d,axis=0)
        j_w = np.dot(R_M,np.array([[0],[0],[1]]))
    j = np.concatenate((j_v, j_w), axis=0)
    return j

for i in range (len(DH_table)):
    jacobian=np.concatenate((jacobian,jacobian_col(joint_types[i],i+1)),1)
    #print(jacobian.shape)
    #jacobian.hstack(jacobian_col(joint_types[i],i+1))
def jac ():
    return np.delete(jacobian,0, axis=1)

# partial deravative approach
j_partial = [
[F_M[0,3].diff(q1), F_M[0,3].diff(q2), F_M[0,3].diff(q3)],
[F_M[1,3].diff(q1), F_M[1,3].diff(q2), F_M[1,3].diff(q3)],
[F_M[2,3].diff(q1), F_M[2,3].diff(q2), F_M[2,3].diff(q3)],
# rotational part
[F_M[0,2].diff(q1), F_M[0,2].diff(q2), F_M[0,2].diff(q3)],
[F_M[1,2].diff(q1), F_M[1,2].diff(q2), F_M[1,2].diff(q3)],
[F_M[2,2].diff(q1), F_M[2,2].diff(q2), F_M[2,2].diff(q3)],
]
