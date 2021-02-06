import pyvista as pv
import numpy as np

class Preprocessor:

	@staticmethod
	def array_reshape(meshfile, data_shape,channel_firtst=False):

		mesh_pv = pv.read(meshfile)

		disps = mesh_pv.point_arrays['computedDispl']
		forces = mesh_pv.point_arrays['externalForce']
		pts = mesh_pv.points

		pts_x = np.unique(pts[:,0])
		pts_y = np.unique(pts[:,1])
		pts_z = np.unique(pts[:,2])

		forces_reshape = np.zeros(data_shape)
		disps_reshape = np.zeros(data_shape)

		for i in range(forces.shape[0]):
			x = pts[i, 0]
			y = pts[i, 1]
			z = pts[i, 2]

			x_index = np.where(pts_x == x)[0][0]
			y_index = np.where(pts_y == y)[0][0]
			z_index = np.where(pts_z == z)[0][0]

			if channel_firtst:
				forces_reshape[:, x_index, y_index, z_index] = forces[i, :]
				disps_reshape[:, x_index, y_index, z_index] = disps[i, :]

			else:
				forces_reshape[x_index, y_index, z_index, :] = forces[i, :]
				disps_reshape[x_index, y_index, z_index, :] = disps[i, :]

		force_mean= np.mean(np.abs(forces), axis=0)
		disp_mean = np.mean(np.abs(disps), axis=0)
		force_std = np.std(np.abs(forces), axis=0)
		disp_std = np.std(np.abs(disps), axis=0)

		return forces_reshape, disps_reshape, force_mean, disp_mean, force_std, disp_std

	# def generate_time_step(meshfiles,time_step, data_shape,channel_first=False):

	# 	'''
	# 	This function prepared dataset for LSTM model
	# 	input: meshfiles: a list which includes pathes to simulation files
	# 	data_shape: shape of data for each simulation file, in format (nx, ny,nz,3)
	# 	nx: number of elements in x direction
	# 	ny: number of elements in y direction
	# 	nz: number of elements in z direction
	# 	'''

	# 	forces_time_steps = np.zeros((time_step, data_shape[0], data_shape[1], data_shape[2],data_shape[3]))
	# 	disps_time_steps = np.zeros((time_step, data_shape[0], data_shape[1], data_shape[2],data_shape[3]))

	# 	for (i,meshfile) in enumerate(meshfiles):

	# 		forces, disps = Preprocessor.array_reshape(meshfile, data_shape,channel_first)
	# 		forces_time_steps[i, :, :, :, :] = forces
	# 		disps_time_steps[i, :, :, :, :] = disps
			
	# 		# print("test,i={}".format(i))
	# 		# if i==(time_step-1): #just consider the last time step
	# 		# 	disps_time_steps[:, :, :, :] = disps

	# 	return forces_time_steps, disps_time_steps
