import os
import sys
import cv2
import numpy
import copy
import itertools
#from tqdm import tqdm
from .pyramid import PyramidImageCreator
from .track import TrackList
from .flow import OpticalflowWrapper
from .Feature.hog import HogFeature
from .Feature.hof import HofFeature
from .Feature.mbh import MbhFeature
from .Feature.trajectory import TrajectoryFeature
from . import param


class DenseTrajectory:

	def __init__(self):
		# Create parameter object
		self.DENSE_SAMPLE_PARAM = param.DenseSampleParameter()
		self.TRJ_PARAM = param.TrajectoryParameter()
		self.PYRAMID_PARAM = param.PyramidImageParameter()
		self.SURF_PARAM = param.SurfParameter()
		self.FLOW_KYPT_PARAM = param.FlowKeypointParameter()
		self.HOMO_PARAM = param.HomographyParameter()

		# Create frature object
		self.surf_create = cv2.xfeatures2d.SURF_create(self.SURF_PARAM.HESSIAN_THRESH)
		self.flow_create = OpticalflowWrapper()
		self.hog_create = HogFeature()
		self.hof_create = HofFeature()
		self.mbh_create = MbhFeature()
		self.trj_create = TrajectoryFeature()


	def __GetCaptureFrames(self, vieo_path):
		capture = cv2.VideoCapture(vieo_path)
		if not capture.isOpened():
			error_message = '{} is not exist.'.format(vieo_path)
			raise Exception('{}:{}():{}'.format(os.path.basename(__file__), sys._getframe().f_code.co_name, error_message))
		
		capture_frames = []
		is_capture = True
		while is_capture:
			is_capture, frame = capture.read()
			if is_capture:
				capture_frames.append(frame)

		frame_rate = int(capture.get(cv2.CAP_PROP_FPS))
		frame_size = (capture_frames[0].shape[1], capture_frames[0].shape[0])
		frame_count = len(capture_frames)
		return capture_frames, frame_rate, frame_size, frame_count
	

	def __DrawTrack(self, frame, track_list, image_scale):
		def GetLineData(track, image_scale):
			track_pts = (track.points[:track.track_num + 1]*image_scale).astype(numpy.int64)

			if track_pts.shape[0] > 1:
				line_begin_idxs = numpy.array(range(track_pts.shape[0] - 1)).tolist()
				line_end_idxs = (numpy.array(range(track_pts.shape[0] - 1)) + 1).tolist()
				line_begin_pts = [(track_pts[i][1], track_pts[i][0]) for i in line_begin_idxs]
				line_end_pts = [(track_pts[j][1], track_pts[j][0]) for j in line_end_idxs]
				line_colors = [(0, numpy.floor(255.0*(i + 1.0)/(track.track_num + 1.0)), 0) for i in line_end_idxs]
			elif track_pts.shape[0] == 1:
				line_end_pts = [(track_pts[0][1], track_pts[0][0])]
				line_begin_pts = []
				line_colors = []
			else:
				line_begin_pts = []
				line_end_pts = []
				line_colors = []
			
			return line_begin_pts, line_end_pts, line_colors
		
		def DrawTrackLine(frame, line_begin_pts, line_end_pts, line_colors):
			[cv2.line(frame, begin_pt, end_pt, line_color, 2, 8) for (begin_pt, end_pt, line_color) in zip(line_begin_pts, line_end_pts, line_colors)]
			if line_end_pts:
				cv2.circle(frame, line_end_pts[-1], 2, (0, 0, 255), -1, 8)

		line_datas = [GetLineData(track, image_scale) for track in track_list.tracks]
		[DrawTrackLine(frame, data[0], data[1], data[2]) for data in line_datas]
		
	
	def __DenseSample(self, gray_frame, prev_points,i):
		# Prepare usage parameters
		width = int(gray_frame.shape[0]/self.DENSE_SAMPLE_PARAM.MIN_DIST)
		height = int(gray_frame.shape[1]/self.DENSE_SAMPLE_PARAM.MIN_DIST)
		x_max = int(self.DENSE_SAMPLE_PARAM.MIN_DIST*width)
		y_max = int(self.DENSE_SAMPLE_PARAM.MIN_DIST*height)
		offset = int(self.DENSE_SAMPLE_PARAM.MIN_DIST/2.0)

		# Prepare sampling points
		all_points = numpy.array([[w, h] for w in range(width) for h in range(height)])	
		if prev_points.size>0:
			enable_prev_flg2 = ((prev_points[:,0] < self.x_max[i]) & (prev_points[:,1] < self.y_max[i]))
			prev_points_1 = (prev_points[enable_prev_flg2]/self.DENSE_SAMPLE_PARAM.MIN_DIST).astype(numpy.int16)
			enable_point_flg2=numpy.full((self.width_list[i],self.height_list[i]),True)
			enable_point_flg2[prev_points_1[:,0],prev_points_1[:,1]]=numpy.full((prev_points_1.shape[0]),False)
			enable_points2 = self.all_points[i][enable_point_flg2]*self.DENSE_SAMPLE_PARAM.MIN_DIST + self.offset			
		else:
			enable_points2=numpy.reshape(self.all_points[i],(-1,2))*self.DENSE_SAMPLE_PARAM.MIN_DIST + self.offset

		eigen_mat = cv2.cornerMinEigenVal(gray_frame, self.DENSE_SAMPLE_PARAM.EIGEN_BLICK_SIZE, self.DENSE_SAMPLE_PARAM.EIGEN_APERTURE_SIZE)
		# Calculate the eigenvalue threshold for corner detection
		max_value = cv2.minMaxLoc(eigen_mat)[1]
		eigen_thresh = max_value*self.DENSE_SAMPLE_PARAM.QUALITY
		
		# Extract corner points
		enable_point_eigen = eigen_mat[enable_points2[:,0], enable_points2[:,1]]
		corner_eigen_flg = (enable_point_eigen > eigen_thresh)
		corner_points = enable_points2[corner_eigen_flg]
		return corner_points



	def __ExtractFeatureDescs(self, prev_gray_frame, flow_warp, track_list, image_scale):
		# Compute feature description
		hog_integral = self.hog_create.Compute(prev_gray_frame)
		hof_integral = self.hof_create.Compute(flow_warp)
		mbhx_integral, mbhy_integral = self.mbh_create.Compute(flow_warp)

		# Extract features
		hog_descs  = [self.hog_create.Extract(hog_integral,  track.points[track.track_num], prev_gray_frame.shape) for track in track_list.tracks]
		hof_descs  = [self.hof_create.Extract(hof_integral,  track.points[track.track_num], prev_gray_frame.shape) for track in track_list.tracks]
		mbhx_descs = [self.mbh_create.Extract(mbhx_integral, track.points[track.track_num], prev_gray_frame.shape) for track in track_list.tracks]
		mbhy_descs = [self.mbh_create.Extract(mbhy_integral, track.points[track.track_num], prev_gray_frame.shape) for track in track_list.tracks]
		trj_descs  = [self.trj_create.Extract(flow_warp, track.points[track.track_num], image_scale) for track in track_list.tracks]
		[track.ResistDescriptor(hog, hof, mbhx, mbhy, trj)
			for (track, hog, hof, mbhx, mbhy, trj) in zip(track_list.tracks, hog_descs, hof_descs, mbhx_descs, mbhy_descs, trj_descs)]

	
	def __AddTrackPoints(self, flow, track_list, image_size):
		if track_list.tracks:
			prev_pts = numpy.array([track.points[track.track_num,:] for track in track_list.tracks])

			# Calcurate track points
			index = numpy.round(numpy.copy(prev_pts)).astype(numpy.int64)
			index[:,0] = numpy.clip(index[:,0], 0, None)
			index[:,0] = numpy.clip(index[:,0], None, image_size[1] - 1)
			index[:,1] = numpy.clip(index[:,1], 0, None)
			index[:,1] = numpy.clip(index[:,1], None, image_size[0] - 1)
			flow_pts = flow[index[:,0], index[:,1]]
			flow_pts = numpy.vstack((flow_pts[:,1], flow_pts[:,0])).transpose()
			track_pts = prev_pts + flow_pts
			
			
			# Remove points outside the range of frame
			enable_track_flg = ((track_pts[:,0] > 0) & (track_pts[:,0] < image_size[1] - 1) & (track_pts[:,1] > 0) & (track_pts[:,1] < image_size[0] - 1))
			enable_track_pts = track_pts[enable_track_flg]
			track_list.RemoveTrack([not flg for flg in enable_track_flg])

			# Tracking points store
			[track.AddPoint(enable_track_pts[idx,:]) for (idx, track) in enumerate(track_list.tracks)]
	

	def __ResistTracks(self, prev_gray_frame, track_list,i):
		track_pts = numpy.array([track.points[track.track_num] for track in track_list.tracks])
		dense_pts = self.__DenseSample(prev_gray_frame, track_pts,i)
		[track_list.ResistTrack(dense_pts[idx]) for idx in range(dense_pts.shape[0])]


	def removetracks_extractfeatures(self,tracks_lists,scales):
                for tracks_list,scale in zip(tracks_lists,scales):
                        tracks=tracks_list.tracks
                        for i in range(len(tracks))[::-1]:
                                if tracks[i].CheckRemove():
                                        track=tracks.pop(i)
                                        if track.CheckValidTrajectory(scale) and track.CheckNotCameraMotion():
                                                self.hog_feature_store2.append(self.hog_create.Normalize(track.hog_descs))
                                                self.hof_feature_store2.append(self.hof_create.Normalize(track.hof_descs))
                                                self.mbhx_feature_store2.append(self.mbh_create.Normalize(track.mbhx_descs))
                                                self.mbhy_feature_store2.append(self.mbh_create.Normalize(track.mbhy_descs))
                                                self.trj_feature_store2.append(self.trj_create.Normalize(track.trj_descs))


	def compute(self, vieo_path, draw_path=None):
		capture_frames, frame_rate, frame_size, self.frame_num = self.__GetCaptureFrames(vieo_path)
		
		self.hog_feature_store2  = []
		self.hof_feature_store2  = []
		self.mbhx_feature_store2 = []
		self.mbhy_feature_store2 = []
		self.trj_feature_store2  = []
		
		# Preparation Video Writer
		if not draw_path is None:
			fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
			writer = cv2.VideoWriter(draw_path, fourcc, frame_rate, frame_size)
		
		# Create Pyramid Image Generator
		pyr_img_creator = PyramidImageCreator((frame_size[1], frame_size[0]), self.PYRAMID_PARAM.MIN_SIZE,
										self.PYRAMID_PARAM.PYRAMID_SCALE_STRIDE,
										self.PYRAMID_PARAM.PYRAMID_SCALE_NUM)
		
		# Create track list
		pyr_track_list = [TrackList(self.hog_create.DIM,
					self.hof_create.DIM,
					self.mbh_create.DIM,
					self.mbh_create.DIM,
					self.trj_create.DIM) for idx in range(pyr_img_creator.image_num)]

		# ----------------------------------------------------------------------------------------------------
		# Init Process Begin
		# ----------------------------------------------------------------------------------------------------
		#progress = tqdm(total=len(capture_frames))
		# Grayscale Conversion
		gray_frame = cv2.cvtColor(capture_frames[0], cv2.COLOR_BGR2GRAY)
		# Pyramid Image Generation
		prev_pyr_gray_frame = pyr_img_creator.Create(gray_frame)
##		# Find keypoints and descriptors
		
		self.width_list = [int(image_size[1]/self.DENSE_SAMPLE_PARAM.MIN_DIST) for image_size in pyr_img_creator.image_sizes]
		self.height_list =[int(image_size[0]/self.DENSE_SAMPLE_PARAM.MIN_DIST) for image_size in pyr_img_creator.image_sizes]
		self.x_max =[int(self.DENSE_SAMPLE_PARAM.MIN_DIST*width) for width in self.width_list]
		self.y_max =[int(self.DENSE_SAMPLE_PARAM.MIN_DIST*height) for height in self.height_list]
		self.offset = int(self.DENSE_SAMPLE_PARAM.MIN_DIST/2.0)

##		self.flow_create.get_para(pry_image_creator.image_sizes,pry_image_creator.image_scales)

		# Prepare sampling points
		self.all_points=[numpy.stack((numpy.tile(numpy.reshape(numpy.arange(width,dtype=numpy.int16),(width,1)),(1,height)),\
                                                      numpy.tile(numpy.arange(height,dtype=numpy.int16),(width,1))),axis=2)
                            for width,height in zip(self.width_list,self.height_list)]
		
		# Dense Sampling
		self.frame_count=0
		pyr_dense_pts = [self.__DenseSample(a,numpy.array([]),i) for i,a in enumerate(prev_pyr_gray_frame)]
		# Tracking points store
		[track_list.ResistTrack(pts[idx]) for (track_list, pts) in zip(pyr_track_list, pyr_dense_pts) for idx in range(pts.shape[0])]
		#progress.update(1)

		# ----------------------------------------------------------------------------------------------------
		# Init Process End
		# ----------------------------------------------------------------------------------------------------

		# ----------------------------------------------------------------------------------------------------
		# Compute Process Begin
		# ----------------------------------------------------------------------------------------------------
		for capture_frame in capture_frames[1:]:
			# Grayscale Conversion
			self.frame_count+=1
			gray_frame = cv2.cvtColor(capture_frame, cv2.COLOR_BGR2GRAY)
			# Pyramid Image Generation
			curr_pyr_gray_frame = pyr_img_creator.Create(gray_frame)
			# Compute Optical Flow
			pyr_flow = [self.flow_create.ExtractFlow(prev_gray_frame, curr_gray_frame) for (prev_gray_frame, curr_gray_frame) in zip(prev_pyr_gray_frame, curr_pyr_gray_frame)]
			
			# Extract feature descriptor
			[self.__ExtractFeatureDescs(prev_gray_frame, flow_warp, track_list, image_scale)
				for (prev_gray_frame, flow_warp, track_list, image_scale) in zip(curr_pyr_gray_frame, pyr_flow, pyr_track_list, pyr_img_creator.image_scales)]			
			
			# Add track points
			[self.__AddTrackPoints(flow, track_list, image_size) for (flow, track_list, image_size) in zip(pyr_flow, pyr_track_list, pyr_img_creator.image_sizes)]
			
			# Draw Tracking points
			if not draw_path is None:
				self.__DrawTrack(capture_frame, pyr_track_list[0], pyr_img_creator.image_scales[0])
				writer.write(capture_frame)


			self.removetracks_extractfeatures(pyr_track_list,pyr_img_creator.image_scales)
			if self.frame_num-self.frame_count>=14:
                                # Regist new points in track data
				[self.__ResistTracks(prev_pyr_gray_frame[i], pyr_track_list[i],i) for i in range(pyr_img_creator.image_num)]
			
			# Update current to previous
			prev_pyr_gray_frame =curr_pyr_gray_frame

			#progress.update(1)
		# ----------------------------------------------------------------------------------------------------
		# Compute Process End
		# ----------------------------------------------------------------------------------------------------

		hog_feature_store  = numpy.array(self.hog_feature_store2)
		hof_feature_store  = numpy.array(self.hof_feature_store2)
		mbhx_feature_store = numpy.array(self.mbhx_feature_store2)
		mbhy_feature_store = numpy.array(self.mbhy_feature_store2)
		trj_feature_store  = numpy.array(self.trj_feature_store2)
		print('VideoPath:{}'.format(vieo_path))
		print('size:{}, fps:{}, frame:{}'.format(frame_size, frame_rate, self.frame_num))

		return hog_feature_store, hof_feature_store, mbhx_feature_store, mbhy_feature_store, trj_feature_store
