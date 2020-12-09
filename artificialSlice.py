# from Utils import *
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
import ants
from scipy import ndimage
import vtk

#Load original image
reader = vtk.vtkNIFTIImageReader()
reader.SetFileName("/media/sf_VB_Folder/Training dataset/training_axial_full_pat0.nii.gz")
reader.Update()
t = reader.GetOutput()

x = np.arange(0,t.GetDimensions()[0])
y = np.arange(0,t.GetDimensions()[1])
# z = np.arange(0,t.GetDimensions()[2])
z = np.linspace(0,t.GetDimensions()[2],16)

xy,yx,zx = np.meshgrid(x,y,z)
cenX = np.mean(xy)
cenY = np.mean(yx)
cenZ = np.mean(zx)
xy = xy - cenX
yx = yx - cenY
zx = zx - cenZ
ori = np.array([259,5,219])
#normal = np.array([0.75,0,0.75])
# new_zx = np.zeros(zx.shape)
# new_zx = new_zx == 1
# ci = 0
# for i in np.arange(0,384,10):
#         cj = 0
#         for j in np.arange(0,384,10):
#             ck = 0
#             for k in np.arange(0,160,10):
#                 if  normal[0]*i + normal[1]*j + normal[2]*k - (normal[0]*ori[0] + normal[1]*ori[1] + normal[2]*ori[2]) == 0:
#                     new_zx[ci,cj,ck] = True
#                 else:
#                     new_zx[ci,cj,ck] = False
#                 ck = ck +1
#             cj = cj + 1
#         ci = ci + 1
# fig = plt.figure()
# ax = fig.add_subplot(111,projection = '3d')
# ax.scatter(xy,yx,zx,alpha = 0.1)
# ax.set_zlim([0,384])
# plt.xlabel("X")
# plt.ylabel("Y")
# ax.set_zlabel("Z")

normal = np.array([0.75,0,-0.75])

z_ax = np.array([0,0,1])
#Angle btw vectors 22 
v = np.cross(z_ax,normal)
s = np.linalg.norm(v)
c = np.dot(z_ax, normal)
I = np.identity (3)
vx = np.array([[0, -v[2],v[1]],[v[2],0, -v[0]],[-v[1],v[0],0]])
R = I + vx + np.matmul(vx,vx) * ((1-c)/(s**2))
angle = np.arccos(c)*180/np.pi
print(angle)
# scale = [int(xy.shape[0]*1.5),int(xy.shape[1]*1.5),int(xy.shape[2]*3)]
scale = xy.shape
xy_new = np.zeros(scale)
yx_new = np.zeros(scale)
zx_new = np.zeros(scale)
Rot = np.array([])
for i in range(zx_new.shape[0]):
    for j in range(zx_new.shape[1]):
        for k in range(zx_new.shape[2]):
            Rot = np.dot(R,np.array([xy[i,j,k],yx[i,j,k],zx[i,j,k]]))
            xy_new[i,j,k] = Rot[0] + cenX
            yx_new[i,j,k] = Rot[1] + cenY
            zx_new[i,j,k] = Rot[2] + cenZ
# ax.scatter(xy_new,yx_new,zx_new,c = 'gray',alpha = 0.1)
# plt.show()


#VTK 
print("Starting VTK process")

pts = vtk.vtkPoints()
nPoints = zx_new.shape[0]*zx_new.shape[1]*zx_new.shape[2]
pts.SetNumberOfPoints(nPoints)
count = 0
#Define the points with reoriente coordinates
for i in range(zx_new.shape[0]):
    for j in range(zx_new.shape[1]):
        for k in range(zx_new.shape[2]):
            pts.SetPoint(count,xy_new[i,j,k],yx_new[i,j,k],zx_new[i,j,k])
            count = count +1
print("Endding of Point creation")
#Create UNstructured grid and add the points
ug = vtk.vtkUnstructuredGrid()
ug.SetPoints(pts)
#Probe filter to extract image pixel intensities in specific coordinates
probe = vtk.vtkProbeFilter()
probe.SetInputData(ug)
probe.SetSourceData(t)
probe.Update()
samples = probe.GetOutput()
samples = samples.GetPointData()
samples = samples.GetScalars()
#Create array with all pixel intensities extracted
w = vtk.vtkDoubleArray()
w.SetNumberOfTuples(samples.GetNumberOfTuples())
for i in range(samples.GetNumberOfTuples()):
    w.SetTuple1(i,samples.GetTuple1(i))
#Check paraview
ug.GetPointData().AddArray(w)
wr = vtk.vtkXMLUnstructuredGridWriter()
wr.SetInputData(ug)
wr.SetFileName("test.vtu")
wr.EncodeAppendedDataOff()
wr.Write()
#Create Image
output = vtk.vtkImageData()
output.SetDimensions(xy_new.shape)
output.AllocateScalars(vtk.VTK_DOUBLE,0)
# output.SetDirectionMatrix(R[0,0],R[0,1],R[0,2],R[1,0],R[1,1],R[1,2],R[2,0],R[2,1],R[2,2])
output.SetOrigin(78.085,-164.968,163.991)

output.SetSpacing(t.GetSpacing()[0],t.GetSpacing()[0],8)
count = 0
# Axis changed and i,k goes form max to min
for j in range(zx_new.shape[1]):
    for i in range(zx_new.shape[0]):
        for k in range(zx_new.shape[2]):
            output.SetScalarComponentFromFloat(zx_new.shape[0]-i-1,j,zx_new.shape[2]-k-1,0,w.GetTuple1(count))
            count = count + 1
    #     print(count)
    # print(count)
P000 = pts.GetPoint(0)
P100 = np.asarray(pts.GetPoint((zx_new.shape[1])*(zx_new.shape[2])))
P010 = np.asarray(pts.GetPoint(zx_new.shape[2]))
P001 = np.asarray(pts.GetPoint(1))
#Point 2
# P000 = pts.GetPoint((zx_new.shape[1])*(zx_new.shape[2])*(zx_new.shape[0]-1))
# P100 = np.asarray(pts.GetPoint((zx_new.shape[1])*(zx_new.shape[2])*(zx_new.shape[0]-2)))
# P010 = np.asarray(pts.GetPoint(zx_new.shape[2]+(zx_new.shape[1])*(zx_new.shape[2])*(zx_new.shape[0]-1)))
# P001 = np.asarray(pts.GetPoint(1+(zx_new.shape[1])*(zx_new.shape[2])*(zx_new.shape[0]-1)))
#Point 3
# P000 = pts.GetPoint((zx_new.shape[1])*(zx_new.shape[2])*(zx_new.shape[0]-1)+zx_new.shape[1]-1)
# P100 = np.asarray(pts.GetPoint((zx_new.shape[1])*(zx_new.shape[2])*(zx_new.shape[0]-2)+zx_new.shape[1]-1))
# P010 = np.asarray(pts.GetPoint((zx_new.shape[1])*(zx_new.shape[2])*(zx_new.shape[0]-1)+ 2*zx_new.shape[1]-1))
# P001 = np.asarray(pts.GetPoint((zx_new.shape[1])*(zx_new.shape[2])*(zx_new.shape[0]-1)+zx_new.shape[1]-2))
#Point 4
# P000 = pts.GetPoint(zx_new.shape[1]-1)
# P100 = np.asarray(pts.GetPoint((zx_new.shape[1])*(zx_new.shape[2]) + zx_new.shape[1]-1))
# P010 = np.asarray(pts.GetPoint(2*zx_new.shape[1]-1))
# P001 = np.asarray(pts.GetPoint(zx_new.shape[1]-2))
#Point 5
# P000 = pts.GetPoint(zx_new.shape[1]*zx_new.shape[2]-1)
# P100 = np.asarray(pts.GetPoint(zx_new.shape[1]*zx_new.shape[2]*2-1))
# P010 = np.asarray(pts.GetPoint((zx_new.shape[1]-1)*zx_new.shape[2]-1))
# P001 = np.asarray(pts.GetPoint(zx_new.shape[1]*zx_new.shape[2]-2))
#Point 6
# P000 = pts.GetPoint((zx_new.shape[1]-1)*zx_new.shape[2])
# P100 = np.asarray(pts.GetPoint((zx_new.shape[1])*zx_new.shape[2] + (zx_new.shape[1]-1)*zx_new.shape[2]))
# P010 = np.asarray(pts.GetPoint((zx_new.shape[1]-2)*zx_new.shape[2]))
# P001 = np.asarray(pts.GetPoint((zx_new.shape[1]-1)*zx_new.shape[2]+1))
#point 7 
# P000 = pts.GetPoint((zx_new.shape[1])*zx_new.shape[2]*zx_new.shape[0]-1)
# P100 = np.asarray(pts.GetPoint((zx_new.shape[1])*zx_new.shape[2]*(zx_new.shape[0]-1)-1))
# P010 = np.asarray(pts.GetPoint((zx_new.shape[1])*zx_new.shape[2]*zx_new.shape[0] - zx_new.shape[2]-1 ))
# P001 = np.asarray(pts.GetPoint((zx_new.shape[1])*zx_new.shape[2]*zx_new.shape[0]  -2))
#Point 8
# P000 = pts.GetPoint((zx_new.shape[1])*zx_new.shape[2]*zx_new.shape[0] - zx_new.shape[2])
# P100 = np.asarray(pts.GetPoint((zx_new.shape[1])*zx_new.shape[2]*(zx_new.shape[0]-1) - zx_new.shape[2]))
# P010 = np.asarray(pts.GetPoint((zx_new.shape[1])*zx_new.shape[2]*zx_new.shape[0] - 2*zx_new.shape[2] ))
# P001 = np.asarray(pts.GetPoint((zx_new.shape[1])*zx_new.shape[2]*zx_new.shape[0] - zx_new.shape[2] + 1))
v3 = (P001 - np.asarray(P000))/np.linalg.norm((P001-P000))
v2 = (P010 - np.asarray(P000))/np.linalg.norm((P010-P000))
v1 = (P100 - np.asarray(P000))/np.linalg.norm((P100-P000))
print(v1.dot(v2))
print(v2.dot(v3))
print(v1.dot(v3))
print(v1)
print(v2)
print(v3)
print(P000)
# output.SetDirectionMatrix(v1[0],v1[1],v1[2],v2[0],v2[1],v2[2],v3[0],v3[1],v3[2])

# output.SetOrigin(np.dot(R,np.array([78.085,-164.968,163.991])))
output.SetOrigin(P000)
output.SetDirectionMatrix(v1[0],v2[0],v3[0],v1[1],v2[1],v3[1],v1[2],v2[2],v3[2])
# output.SetDirectionMatrix(0.5,0.5,0,0,1,0,0,0,1)

ImgWr =vtk.vtkMetaImageWriter()
ImgWr.SetFileName('FinalVol1.vti')
ImgWr.SetInputDataObject(output)
ImgWr.Write()
print("END")







# new_zx = np.zeros(zx.shape)
# new_zx = new_zx == 1
# normal = np.array([0.75,0,-0.75])
# ori0 = np.array([259,5,219])
# # spacings = np.array([0.8,0.8,10])
# # line = np.zeros([len(np.arange(-250,250)),3])
# # for i in np.arange(-250,250): 
# #     line[i,:] = ori0 + np.multiply(normal,(i*spacings))
# # new_zx = - (normal[0]*xy + normal[1]*yx - (normal[0]*normlaPoint[0] + normal[1]*normlaPoint[1] + normal[2]*normlaPoint[2]))/ normal[2]

# ci = 0
# for n in np.arange(-100,100,10):
#     ori = ori0 + np.round(normal*n)
#     print(ori)
#     ci = 0
#     for i in np.arange(0,384,10):
#         cj = 0
#         for j in np.arange(0,384,10):
#             ck = 0
#             for k in np.arange(0,160,10):
#                 if  normal[0]*i + normal[1]*j + normal[2]*k - (normal[0]*ori[0] + normal[1]*ori[1] + normal[2]*ori[2]) == 0:
#                     new_zx[ci,cj,ck] = True
#                 else:
#                     new_zx[ci,cj,ck] = False
#                 ck = ck +1
#             cj = cj + 1
#         ci = ci + 1
# #     fig = plt.figure()
# #     ax = fig.add_subplot(111,projection = '3d')
# #     ax.scatter(xy,yx,zx,c = new_zx)
# #     ax.set_zlim([0,384])
# #     plt.xlabel("X")
# #     plt.ylabel("Y")
# #     ax.set_zlabel("Z")
# # plt.show()

# zx = ndimage.map_coordinates(zx, [[1,0,0],[0,1,0],[0.75,0,-0.75]])
# fig = plt.figure()
# ax = fig.add_subplot(111,projection = '3d')
# ax.scatter(xy,yx,zx)
# ax.set_zlim([0,384])
# plt.xlabel("X")
# plt.ylabel("Y")
# ax.set_zlabel("Z")
# plt.show()
# # #Load original image
# # initialVol = ants.image_read("/media/sf_VB_Folder/Training dataset/training_axial_full_pat0.nii.gz")
# # #Extract numpy array and transform to pyvista
# # t = initialVol.view()
# # tt = pv.wrap(t)
# # #Normal of 2CH orientation and define some origins 
# # CH2Dir = np.array([0.752577,0,-0.75])
# # a = tt.center + CH2Dir * tt.length / 3
# # b = tt.center - CH2Dir * tt.length / 3
# # line = pv.Line(a,b,20)

# # slices = pv.MultiBlock()
# # #Iterate over the different origines and slice the original image
# # for point in line.points:
# #     print(tt.slice(normal = CH2Dir, origin = point))
# #     slices.append(tt.slice(normal = CH2Dir, origin = point))
# # # p = pv.Plotter()
# # # p.add_mesh(slices,cmap = 'gray')
# # # p.show()

# # print(initialVol) 