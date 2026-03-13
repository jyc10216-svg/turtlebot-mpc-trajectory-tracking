#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import block_diag
from turtlebot3_mpc.src.Mpc_Final.Middle_Line import is_point_middle_line
from turtlebot3_mpc.src.Mpc_Final.Start_Point import findStartPoint
from turtlebot3_mpc.src.Mpc_Final.Target_Heading import targetHeading
from turtlebot3_mpc.src.Mpc_Final.mpc_controller_self2026 import mpc_controller
from turtlebot3_mpc.src.Mpc_Final.Line_Equation_Coefficient import line_eq_coef
from turtlebot3_mpc.src.Mpc_Final.wrap_angle import wrap_to_pi
import pandas

#Write the file path
file_path = '/home/jyc/catkin_ws/src/turtlebot3_mpc/src/Mpc_Final/reference_path.csv'

data = pandas.read_csv(file_path)

x_values = data['x'].tolist()
y_values = data['y'].tolist()

path = np.column_stack((x_values, y_values))
rows, _ = path.shape


class TurtlebotModel():
	def __init__(self):
		self.pose_subsrciber = rospy.Subscriber('/odom', Odometry, self.pose_callback) #Odometry 토픽을 구독 -> 속도 정보를 해당 토픽으로부터 받음
		self.control_publisher = rospy.Publisher('/cmd_vel', Twist, queue_size = 10) #Twist: linear vel, angular vel을 저장하는 메세지의 형태
				                                                             #cmd_vel은 토픽의 이름으로, 해당 토픽을 터틀봇 노드에 발행
				                                                             #twist 벡터에 적힌 제어 속도값으로 터틀봇을 운용
				                                                             
		self.x = 0.0
		self.y = 0.0
		self.theta = 0.0
		self.pose_received = False

	def pose_callback(self, msg):

		#Position determination

		#rospy.loginfo("Pose callback called")
		
		self.x = msg.pose.pose.position.x
		self.y = msg.pose.pose.position.y

		orientation_quat = (
			msg.pose.pose.orientation.x,
			msg.pose.pose.orientation.y,
			msg.pose.pose.orientation.z,
			msg.pose.pose.orientation.w
		)

		_, _, self.theta = euler_from_quaternion(orientation_quat)

		self.pose_received = True
		
	def control_step(self):
		if not self.pose_received:
			return
			
		x0 = np.array([self.x,self.y,self.theta])
		#Basic parameter specification

		Ts = 0.1

		Np = 10 # Number of prediction horizon
		N = rows # Length of the reference path coordinates 
		m = 3  # Number of state variables => x, y, theta
		n = 2 # Number of control input => v, w

		U_ref = np.zeros((N,n))
		X_ref = np.zeros((N,m))

		steering = 0
		v_ref = 0.1 #reference velocity

		# Constraints Definition

		max_steer = 0.3
		min_steer = -0.3
		max_vel = 0.15
		min_vel = 0
		U_max = np.array([max_vel, max_steer])
		U_min = np.array([min_vel, min_steer])
		goalradius = 0.5

		# Weight Matrix Definition:

		Q1 = np.array([[10,0,0], [0,10,0], [0,0,10]])
		Q0 = np.array([[10,0,0], [0,10,0], [0,0,10]])
		R1 = np.diag([100,10])
		R0 = np.diag([100,10])

		Q = Q0
		R = R0

		for i in range(1,Np):
			Q = block_diag(Q,Q1)
			R = block_diag(R,R1)
			
		# Find the nearest reference coordinate:

		lookforward = 2

		i = findStartPoint(x0, path)
		i = min(i + lookforward, N - Np - 1)

		if np.linalg.norm(x0[0:2]-path[i,:]) <= goalradius:
			i = min(i + lookforward, N - Np - 1)


	
		currentGoal = path[i,:]
		
		# Check whether the turtlebot model is too close to the reference point or not
		distance_to_Goal = np.linalg.norm(x0[0:2]-currentGoal)
		
		# Increase the index if it is too close
		if distance_to_Goal <= goalradius:
		    i = min(i + 1, N - Np - 1)
		
		# Reference Path & Control input assignment
		theta0 = x0[2]

		X_ref[i,0:2] = path[i,:]

		heading_ref = np.zeros((N, 1))

		heading_ref[i,:] = targetHeading(x0[0:2], path[i,:])

		X_ref[i,2] = heading_ref[i]
		U_ref[i, 0] = v_ref
		U_ref[i, 1] = (wrap_to_pi(heading_ref[i, 0] - theta0)) / Ts

		for j in range(2, Np+1):
		    X_ref[i+j-1, 0:2] = path[i+j-1,:]
		    heading_ref[i+j-1] = targetHeading(path[i+j-2,:], path[i+j-1,:])
		    X_ref[i+j-1, 2] = heading_ref[i+j-1,0]
		    U_ref[i+j-1, 0] = v_ref
		    U_ref[i+j-1, 1] = (wrap_to_pi(heading_ref[i+j-1, 0] - heading_ref[i+j-2, 0])) / Ts
		
		# Linear State Equation Parameter matrix definition
		
		A = lambda i: np.array([[1, 0, -U_ref[i,0]*np.sin(X_ref[i,2])*Ts], \
			    [0, 1, U_ref[i,0]*np.cos(X_ref[i,2])*Ts], \
			    [0, 0, 1]])

		B = lambda i: np.array([[np.cos(X_ref[i,2])*Ts, 0], 
			    [np.sin(X_ref[i,2])*Ts, 0], 
			    [0, Ts]])

		pos_diff = np.array(x0[0:2] - X_ref[i, 0:2])
		angle_diff = np.array([wrap_to_pi(heading_ref[i, 0] - theta0)])

		init = np.concatenate((pos_diff, angle_diff))
		
		# Send the parameters to the controller and calculate the optimal control input
		
		vel, steering = mpc_controller(i, A, B, init, Np, U_ref[i:i+Np,:], Q, R, U_min, U_max)
		
		# Monitor the position and the control input of turtlebot
		print('Input velocity: linear vel: {0}, \nangular vel: {1}	'.format(vel,steering))
		print('Current position - x = {0}, y = {1}, theta = {2}'.format(self.x, self.y, self.theta))
		print("start index:", i)
		print("target point:", path[i, :])
		print("distance to target:", np.linalg.norm(x0[0:2] - path[i, :]))
		print("theta0:", theta0)
		print("heading_ref:", heading_ref[i, 0])
		print("heading error:", wrap_to_pi(theta0 - heading_ref[i, 0]))

		# Publish the control input to the turtlebot
		
		twist = Twist()
		twist.linear.x = vel
		twist.angular.z = steering
		
		self.control_publisher.publish(twist)
		
	def spin(self):
		rate = rospy.Rate(10)
		while not rospy.is_shutdown():
			self.control_step()
			rate.sleep()

	def shutdown(self):
		rospy.loginfo('Node has been shut down')
        

        
        
def main(args=None):
	rospy.init_node('turtlebot_listener', anonymous = True)
	turtlebot3=TurtlebotModel()

	try:
		turtlebot3.spin()
	except rospy.ROSInterruptException:
		pass
	finally:
		turtlebot3.shutdown()


if __name__ == '__main__':
	main()



