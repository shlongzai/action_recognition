import cv2
import numpy


class HofFeature:
	
	def __init__(self):
		self.PATCH_SIZE = 32
		self.XY_CELL_NUM = 2
		self.T_CELL_NUM = 3
		self.MIN_FLOW_THRESH = 0.4
		self.BIN_NUM = 9
		self.DIM = self.BIN_NUM*self.XY_CELL_NUM*self.XY_CELL_NUM
		self.EPSILON = 0.05
	

	def Compute(self, flow):
		# Split xy flow
		x_flow, y_flow = cv2.split(flow)
		
		# Split xy flow to magnitude and angle 
		flow_mag = numpy.sqrt(x_flow*x_flow + y_flow*y_flow)
		flow_ang_rad = numpy.arctan2(y_flow, x_flowdtype=np.float32)
		
		# Calc flow histogram bins
		bin_origin = flow_ang_rad * (self.BIN_NUM - 1)/(2*numpy.pi)

		# Extract min flow and replace magnitude and bin
##		min_flow_mask = (flow_mag <= self.MIN_FLOW_THRESH)
##		flow_mag[min_flow_mask] = 1.0
##		bin_origin[min_flow_mask] = self.BIN_NUM - 1
##
##		# Split two adjacent bins
##		bin_floor = numpy.floor(bin_origin).astype(numpy.int64)
##		bin_ceil = numpy.ceil(bin_origin).astype(numpy.int64) % (self.BIN_NUM - 1)
##
##		# Calc adjacent bins mask
##		bin_masks_floor = [(bin_floor == i).astype(numpy.int64) for i in range(self.BIN_NUM)]
##		bin_masks_ceil = [(bin_ceil == i).astype(numpy.int64) for i in range(self.BIN_NUM)]
##		
##		# Split the magnitude to two adjacent bins
##		mag_floor = (bin_origin - bin_floor)*flow_mag
##		mag_ceil = (flow_mag - mag_floor)
##	
##		# Add adjacent magnitudes and convert integral features
##		features = [(mag_floor*mask_floor) + (mag_ceil*mask_ceil) for(mask_floor, mask_ceil) in zip(bin_masks_floor, bin_masks_ceil)]
##		integral_features = [cv2.integral(feature) for feature in features]
                integral_features=numpy.zeros((flow.shape[0],flow.shape[1],self.BIN_NUM),dtype=numpy.float32)
                bin_floor = numpy.floor(bin_origin).astype(numpy.int8)
                sub=bin_origin-bin_floor
                double_flow_mag=numpy.stack((sub*flow_mag,(1-sub)*flow_mag),axis=2)
                min_flow_mask = (flow_mag > self.MIN_FLOW_THRESH)
                for i in range(self.BIN_NUM-1):
                        mask=numpy.logical_and(bin_floor==i,min_flow_mask)
                        integral_features[:,:,i]+=numpy.where(mask,double_flow_mag[:,:,0],0)
                        integral_features[:,:,(i+1)%(self.BIN_NUM-1)]+=numpy.where(mask,double_flow_mag[:,:,1],0)
                integral_features[:,:,self.BIN_NUM-1]=np.where(min_flow_mask,0,flow_mag)
                integral_features=numppy.cumsum(numpy.cumsum(integral_features,axis=0),axis=1)
		return integral_features
	

	def Extract(self, integral_features, point, img_size):

		block_size = int(self.PATCH_SIZE/2)
		top_right_x = int(min(max(point[0] - block_size, 0), img_size[1] - self.PATCH_SIZE))
		top_right_y = int(min(max(point[1] - block_size, 0), img_size[0] - self.PATCH_SIZE))
		
##		rect_points = [[(top_right_y + block_size*idx_y,           top_right_x + block_size*idx_x),
##						(top_right_y + block_size*idx_y,           top_right_x + block_size*(idx_x + 1) - 1),
##						(top_right_y + block_size*(idx_y + 1) - 1, top_right_x + block_size*idx_x),
##						(top_right_y + block_size*(idx_y + 1) - 1, top_right_x + block_size*(idx_x + 1) - 1)] for idx_y in range(self.XY_CELL_NUM) for idx_x in range(self.XY_CELL_NUM)]
##                id_x=self.PATCH_SIZE/numpy.arange(self.XY_CELL_NUM)
##		feature = numpy.array([integral[points[3]] + integral[points[0]] - integral[points[1]] - integral[points[2]] for points in rect_points for integral in integral_features])
                coor=numpy.array([[i,j,k] for i in range(self.XY_CELL_NUM) for j in range(self.XY_CELL_NUM) for k in range(self.BIN_NUM)],dtype=numpy.int16)
                block_size2=np.array([[self.PATCH_SIZE//self.XY_CELL_NUM,self.PATCH_SIZE//self.XY_CELL_NUM,1]])
                coor2=coor[:,0:2]+(self.PATCH_SIZE//self.XY_CELL_NUM-1)
                coor[:,:]=coor*block_size2
                a=(2*self.XY_CELL_NUM+1)*self.BIN_NUM
                feature=integral_features[coor2[:,0],coor2[:,1],coor[:,2]]\
                        +integral_features[coor[:,0],coor[:,1],coor[:,2]]\
                        -integral_features[coor2[:,0],coor[:,1],coor[:,2]]\
                        -integral_features[coor[:,0],coor2[:,1],coor[:,2]]
                            
		feature[feature < 0] = 0
		feature += self.EPSILON

		feature_normalize = numpy.sqrt(feature/numpy.sum(feature))
		return feature_normalize

	
	def Normalize(self, feature):
		split_features = numpy.split(feature, self.T_CELL_NUM)

		feature = [numpy.sum(split_feature, axis=0) for split_feature in split_features]
		feature = numpy.array(feature).reshape(-1)
		
		normalize = numpy.floor(feature.shape[0]/self.T_CELL_NUM)
		feature_normalize = feature/normalize

		return feature_normalize
