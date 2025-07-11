import pybullet as p
import pybullet_data
import time

def setup_simulation():
    physicsClient = p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.setGravity(0, 0, -9.81)

    p.loadURDF("plane.urdf")
    
    return physicsClient

def load_robot_and_create_controls(start_pos=[0, 0, 1], start_orientation=[0, 0, 0, 1]):
    start_orientation_quat = p.getQuaternionFromEuler([0, 0, 0])
    
    robot_id = p.loadURDF(r"/home/robot/dev/dyn/3DOF/urdf/3DOF Yomi Assembly_Anush.urdf", start_pos, useFixedBase = True)

    joint_sliders = []
    num_joints = p.getNumJoints(robot_id)
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8')
        joint_type = joint_info[2]
        
        if joint_type == p.JOINT_REVOLUTE or joint_type == p.JOINT_PRISMATIC:
            # Get the joint's limits.
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]

            # slider_id = p.addDebugParameter(
            #     paramName=joint_name,
            #     rangeMin=lower_limit,
            #     rangeMax=upper_limit,
            #     startValue=0  
            # )
            # joint_sliders.append({'id': slider_id, 'joint_index': i})
            
    return robot_id, joint_sliders

def run_simulation(robot_id, joint_sliders):
    """The main simulation loop."""
    print("Simulation started. Move the sliders to control the robot. Press Ctrl+C in terminal to exit.")
    try:
        while True:
           
            for slider in joint_sliders:
                slider_id = slider['id']
                joint_index = slider['joint_index']
                
                target_position = p.readUserDebugParameter(slider_id)
                
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=target_position
                )

            p.stepSimulation()
            
            time.sleep(1./240.)
            
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        # Disconnect from the physics server.
        p.disconnect()

if __name__ == '__main__':
    # 1. Setup the environment
    setup_simulation()
    
    # 2. Load the robot and create interactive controls
    robot, sliders = load_robot_and_create_controls()
    
    # 3. Run the main simulation loop
    run_simulation(robot, sliders)