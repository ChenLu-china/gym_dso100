import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple
import numpy as np


class MujocoXMLToURDFConverter:
    """
    Utility to convert Mujoco XML files to URDF format for use with Isaac Gym.
    Note: This is a simplified converter and may not handle all Mujoco features.
    """
    
    def __init__(self):
        self.body_count = 0
        self.joint_count = 0
        
    def convert_xml_to_urdf(self, xml_path: str, urdf_path: str, robot_name: str = "robot"):
        """Convert Mujoco XML to URDF format."""
        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Create URDF root
        urdf_root = ET.Element("robot")
        urdf_root.set("name", robot_name)
        
        # Process worldbody
        worldbody = root.find(".//worldbody")
        if worldbody is not None:
            self._process_body(worldbody, urdf_root, is_root=True)
        
        # Process actuators and create transmission elements
        actuators = root.find(".//actuator")
        if actuators is not None:
            self._process_actuators(actuators, urdf_root)
        
        # Write URDF file
        self._write_urdf(urdf_root, urdf_path)
        
    def _process_body(self, body_elem, urdf_root, parent_link="world", is_root=False):
        """Process a Mujoco body element and convert to URDF links and joints."""
        
        # Extract body properties
        body_name = body_elem.get("name", f"body_{self.body_count}")
        self.body_count += 1
        
        # Get position and orientation
        pos = self._parse_vector(body_elem.get("pos", "0 0 0"))
        quat = self._parse_vector(body_elem.get("quat", "1 0 0 0"))
        
        # Create link element
        link = ET.SubElement(urdf_root, "link")
        link.set("name", body_name)
        
        # Process geometries for visual and collision
        geoms = body_elem.findall("geom")
        for geom in geoms:
            self._process_geometry(geom, link)
        
        # Process inertial properties
        inertial_elem = body_elem.find("inertial")
        if inertial_elem is not None:
            self._process_inertial(inertial_elem, link)
        else:
            # Add default inertial properties
            self._add_default_inertial(link)
        
        # Create joint if not root
        if not is_root and parent_link != "world":
            joint = ET.SubElement(urdf_root, "joint")
            joint_name = f"joint_{self.joint_count}"
            self.joint_count += 1
            
            joint.set("name", joint_name)
            joint.set("type", "revolute")  # Default to revolute
            
            # Parent and child links
            parent = ET.SubElement(joint, "parent")
            parent.set("link", parent_link)
            child = ET.SubElement(joint, "child")
            child.set("link", body_name)
            
            # Origin
            origin = ET.SubElement(joint, "origin")
            origin.set("xyz", f"{pos[0]} {pos[1]} {pos[2]}")
            # Convert quaternion to RPY (simplified)
            rpy = self._quat_to_rpy(quat)
            origin.set("rpy", f"{rpy[0]} {rpy[1]} {rpy[2]}")
            
            # Axis (default to z-axis)
            axis = ET.SubElement(joint, "axis")
            axis.set("xyz", "0 0 1")
            
            # Limits
            limit = ET.SubElement(joint, "limit")
            limit.set("lower", "-3.14159")
            limit.set("upper", "3.14159")
            limit.set("effort", "100")
            limit.set("velocity", "10")
        
        # Process child bodies recursively
        child_bodies = body_elem.findall("body")
        for child_body in child_bodies:
            self._process_body(child_body, urdf_root, body_name)
    
    def _process_geometry(self, geom_elem, link_elem):
        """Process Mujoco geometry and add to URDF link."""
        geom_type = geom_elem.get("type", "box")
        geom_name = geom_elem.get("name", "geom")
        
        # Visual element
        visual = ET.SubElement(link_elem, "visual")
        visual.set("name", f"{geom_name}_visual")
        
        # Collision element
        collision = ET.SubElement(link_elem, "collision")
        collision.set("name", f"{geom_name}_collision")
        
        # Process geometry based on type
        for elem in [visual, collision]:
            geometry = ET.SubElement(elem, "geometry")
            
            if geom_type == "box":
                size = self._parse_vector(geom_elem.get("size", "0.1 0.1 0.1"))
                box = ET.SubElement(geometry, "box")
                box.set("size", f"{size[0]*2} {size[1]*2} {size[2]*2}")  # Mujoco uses half-sizes
                
            elif geom_type == "sphere":
                size = float(geom_elem.get("size", "0.1"))
                sphere = ET.SubElement(geometry, "sphere")
                sphere.set("radius", str(size))
                
            elif geom_type == "cylinder":
                size = self._parse_vector(geom_elem.get("size", "0.1 0.1"))
                cylinder = ET.SubElement(geometry, "cylinder")
                cylinder.set("radius", str(size[0]))
                cylinder.set("length", str(size[1]*2))  # Mujoco uses half-length
                
            elif geom_type == "mesh":
                mesh_file = geom_elem.get("mesh", "")
                if mesh_file:
                    mesh = ET.SubElement(geometry, "mesh")
                    mesh.set("filename", f"package://gym_dso100/meshes/{mesh_file}.stl")
            
            # Add origin if position/rotation specified
            pos = self._parse_vector(geom_elem.get("pos", "0 0 0"))
            quat = self._parse_vector(geom_elem.get("quat", "1 0 0 0"))
            
            if not np.allclose(pos, [0, 0, 0]) or not np.allclose(quat, [1, 0, 0, 0]):
                origin = ET.SubElement(elem, "origin")
                origin.set("xyz", f"{pos[0]} {pos[1]} {pos[2]}")
                rpy = self._quat_to_rpy(quat)
                origin.set("rpy", f"{rpy[0]} {rpy[1]} {rpy[2]}")
        
        # Add material for visual
        material = ET.SubElement(visual, "material")
        material.set("name", f"{geom_name}_material")
        color = ET.SubElement(material, "color")
        rgba = geom_elem.get("rgba", "0.5 0.5 0.5 1.0")
        color.set("rgba", rgba)
    
    def _process_inertial(self, inertial_elem, link_elem):
        """Process Mujoco inertial properties and add to URDF link."""
        inertial = ET.SubElement(link_elem, "inertial")
        
        # Mass
        mass_elem = ET.SubElement(inertial, "mass")
        mass = float(inertial_elem.get("mass", "1.0"))
        mass_elem.set("value", str(mass))
        
        # Position
        pos = self._parse_vector(inertial_elem.get("pos", "0 0 0"))
        origin = ET.SubElement(inertial, "origin")
        origin.set("xyz", f"{pos[0]} {pos[1]} {pos[2]}")
        
        # Inertia matrix
        inertia = ET.SubElement(inertial, "inertia")
        diaginertia = self._parse_vector(inertial_elem.get("diaginertia", "0.1 0.1 0.1"))
        
        inertia.set("ixx", str(diaginertia[0]))
        inertia.set("iyy", str(diaginertia[1]))
        inertia.set("izz", str(diaginertia[2]))
        inertia.set("ixy", "0.0")
        inertia.set("ixz", "0.0")
        inertia.set("iyz", "0.0")
    
    def _add_default_inertial(self, link_elem):
        """Add default inertial properties to a link."""
        inertial = ET.SubElement(link_elem, "inertial")
        
        mass = ET.SubElement(inertial, "mass")
        mass.set("value", "1.0")
        
        origin = ET.SubElement(inertial, "origin")
        origin.set("xyz", "0 0 0")
        origin.set("rpy", "0 0 0")
        
        inertia = ET.SubElement(inertial, "inertia")
        inertia.set("ixx", "0.1")
        inertia.set("iyy", "0.1")
        inertia.set("izz", "0.1")
        inertia.set("ixy", "0.0")
        inertia.set("ixz", "0.0")
        inertia.set("iyz", "0.0")
    
    def _process_actuators(self, actuators_elem, urdf_root):
        """Process Mujoco actuators and create transmission elements."""
        motors = actuators_elem.findall("motor")
        
        for i, motor in enumerate(motors):
            joint_name = motor.get("joint", f"joint_{i}")
            
            # Create transmission
            transmission = ET.SubElement(urdf_root, "transmission")
            transmission.set("name", f"{joint_name}_transmission")
            
            # Type
            trans_type = ET.SubElement(transmission, "type")
            trans_type.text = "transmission_interface/SimpleTransmission"
            
            # Joint
            joint = ET.SubElement(transmission, "joint")
            joint.set("name", joint_name)
            
            joint_interface = ET.SubElement(joint, "hardwareInterface")
            joint_interface.text = "hardware_interface/EffortJointInterface"
            
            # Actuator
            actuator = ET.SubElement(transmission, "actuator")
            actuator.set("name", f"{joint_name}_motor")
            
            actuator_interface = ET.SubElement(actuator, "hardwareInterface")
            actuator_interface.text = "hardware_interface/EffortJointInterface"
            
            mechanical_reduction = ET.SubElement(actuator, "mechanicalReduction")
            mechanical_reduction.text = "1"
    
    def _parse_vector(self, vector_str: str) -> List[float]:
        """Parse a space-separated vector string."""
        return [float(x) for x in vector_str.split()]
    
    def _quat_to_rpy(self, quat: List[float]) -> List[float]:
        """Convert quaternion to roll-pitch-yaw (simplified conversion)."""
        # This is a simplified conversion - for production use, 
        # consider using proper quaternion to Euler conversion
        w, x, y, z = quat
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return [roll, pitch, yaw]
    
    def _write_urdf(self, urdf_root, urdf_path: str):
        """Write URDF to file with proper formatting."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(urdf_path), exist_ok=True)
        
        # Create XML declaration
        xml_str = '<?xml version="1.0"?>\n'
        xml_str += ET.tostring(urdf_root, encoding='unicode')
        
        # Write to file
        with open(urdf_path, 'w') as f:
            f.write(xml_str)
        
        print(f"URDF file written to: {urdf_path}")


def convert_gym_dso100_assets():
    """Convert all gym-dso100 XML assets to URDF format."""
    converter = MujocoXMLToURDFConverter()
    
    # Base paths
    assets_dir = os.path.join(os.path.dirname(__file__), "..", "assets")
    urdf_dir = os.path.join(assets_dir, "urdf")
    
    # Create URDF directory
    os.makedirs(urdf_dir, exist_ok=True)
    
    # Convert lift.xml
    lift_xml = os.path.join(assets_dir, "lift.xml")
    lift_urdf = os.path.join(urdf_dir, "lift_robot.urdf")
    
    if os.path.exists(lift_xml):
        converter.convert_xml_to_urdf(lift_xml, lift_urdf, "lift_robot")
    
    # Convert other XML files as needed
    for xml_file in ["so_arm100.xml"]:
        xml_path = os.path.join(assets_dir, xml_file)
        urdf_path = os.path.join(urdf_dir, xml_file.replace(".xml", ".urdf"))
        
        if os.path.exists(xml_path):
            converter.convert_xml_to_urdf(xml_path, urdf_path, xml_file.replace(".xml", ""))


if __name__ == "__main__":
    convert_gym_dso100_assets() 