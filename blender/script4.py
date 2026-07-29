import bpy
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from math import radians

def create_root(body):
    root=bpy.data.objects.new("DroneRoot",None)
    bpy.context.collection.objects.link(root)

    body.parent=root

    return root

KEYFRAME_INTERPOLATION = {"CONSTANT": 0, "LINEAR": 1, "BEZIER": 2}


def create_channelbag(obj, action_name):
    """
    Blender 4.4+ moved F-Curves under action.layers -> strips -> channelbag.
    Sets up a single-slot, single-layer action and returns its channelbag,
    which exposes .fcurves like the old Action.fcurves did.
    """

    action=bpy.data.actions.new(action_name)
    slot=action.slots.new(id_type='OBJECT',name=obj.name)

    obj.animation_data_create()
    obj.animation_data.action=action
    obj.animation_data.action_slot=slot

    layer=action.layers.new(name="Layer")
    strip=layer.strips.new(type='KEYFRAME')

    return strip.channelbag(slot,ensure=True)


def frd_to_blender(v):
    """
    Data is logged in FRD (X-forward, Y-right, Z-down).
    Blender uses X-forward, Y-left, Z-up.
    This remap is a proper rotation (det=+1, a 180deg roll about X), so it
    applies unchanged to both position vectors and a quaternion's (x,y,z)
    vector part -- the scalar w is unaffected.
    """

    out = np.empty_like(v)
    out[..., 0] = v[..., 0]
    out[..., 1] = -v[..., 1]
    out[..., 2] = -v[..., 2]
    return out


def insert_keyframes(fcurve, frames, values, interpolation="LINEAR"):
    """Bulk-write keyframes via foreach_set instead of per-frame keyframe_insert."""

    n = len(frames)

    fcurve.keyframe_points.add(n)

    co = np.empty(n * 2, dtype=np.float64)
    co[0::2] = frames
    co[1::2] = values
    fcurve.keyframe_points.foreach_set("co", co)

    fcurve.keyframe_points.foreach_set(
        "interpolation",
        np.full(n, KEYFRAME_INTERPOLATION[interpolation], dtype=np.int32),
    )

    fcurve.update()


def animate(root,position,quaternion,fps=200,stride=1):

    root.rotation_mode="QUATERNION"
    channelbag=create_channelbag(root,"DroneRootAction")

    n=len(position)
    frames=np.arange(1,n+1,stride,dtype=np.float64)

    pos=frd_to_blender(position[::stride])

    quat_wxyz=quaternion[::stride][:,[0,1,2,3]]
    quat_wxyz[:,1:]=frd_to_blender(quat_wxyz[:,1:])

    for axis in range(3):
        fcurve=channelbag.fcurves.new(data_path="location",index=axis)
        insert_keyframes(fcurve,frames,pos[:,axis])

    for axis in range(4):
        fcurve=channelbag.fcurves.new(data_path="rotation_quaternion",index=axis)
        insert_keyframes(fcurve,frames,quat_wxyz[:,axis])

    bpy.context.scene.render.fps=fps
    bpy.context.scene.frame_end=n

def animate_props(props,rpm,dt=1/200,stride=1):

    direction=np.array([-1,1,-1,1])

    angle=np.concatenate([
        np.zeros((1,4)),
        np.cumsum(rpm[1:]*direction*(2*np.pi/60)*dt,axis=0),
    ])

    n=len(angle)
    frames=np.arange(1,n+1,stride,dtype=np.float64)
    angle_sub=angle[::stride]

    for j,prop in enumerate(props):
        channelbag=create_channelbag(prop,f"{prop.name}Action")
        fcurve=channelbag.fcurves.new(data_path="rotation_euler",index=2)
        insert_keyframes(fcurve,frames,angle_sub[:,j])

@dataclass
class Mesh:
    file:str
    rotation:tuple=(0,0,0)

MESHES={
    "body":Mesh("body.stl"),
    "prop_fr":Mesh("prop_1.stl"),
    "prop_fl":Mesh("prop_2.stl"),
    "prop_rl":Mesh("prop_1.stl"),
    "prop_rr":Mesh("prop_2.stl")
}

MOTOR_POSITIONS={
    "prop_fr":(0.208,0.225,0.05),
    "prop_fl":(0.208,-0.225,0.05),
    "prop_rl":(-0.228,-0.225,0.05),
    "prop_rr":(-0.228,0.225,0.05)
}

STL_DIR=Path("~/Thesis/Thesis-Animation/alpha_drone_mk2_stl_corrected").expanduser()
DATA=Path("~/Thesis/Thesis-Animation/data_real/log_205/flight.npz").expanduser()

FPS=30

# Keep every Nth sample as a keyframe (data is at FPS Hz already; stride=1 -> every sample).
# Keyframe timing (frame numbers) is unchanged, so playback speed/duration is unaffected.
STRIDE=1

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

def import_stl(name,m):
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.wm.stl_import(filepath=str(STL_DIR/m.file))
    o=bpy.context.selected_objects[0]
    o.name=name
    o.rotation_euler=[radians(x) for x in m.rotation]
    return o

objects={n:import_stl(n,m) for n,m in MESHES.items()}

body=objects["body"]
root=create_root(body)

for n,p in MOTOR_POSITIONS.items():
    objects[n].location=p
    objects[n].parent=body

data=np.load(DATA)

position=data["position"]
quaternion=data["quaternion"]
rpm=data["motor"]

print(f"loaded {len(position)} frames")

animate(
    root,
    position,
    quaternion,
    fps=FPS,
    stride=STRIDE
)

animate_props(
    [
        objects["prop_fr"],
        objects["prop_fl"],
        objects["prop_rl"],
        objects["prop_rr"],
    ],
    rpm,
    dt=1/FPS,
    stride=STRIDE
)

SPOTLIGHT_POSITIONS={
    "spot_1":(5,5,5),
    "spot_2":(-5,5,5),
    "spot_3":(-5,-5,5),
    "spot_4":(5,-5,5),
}

SPOTLIGHT_POWER=1000  # watts
SPOTLIGHT_ANGLE=radians(60)

def create_spotlight(name,position,target,power=SPOTLIGHT_POWER,angle=SPOTLIGHT_ANGLE):
    light_data=bpy.data.lights.new(name=name,type="SPOT")
    light_data.energy=power
    light_data.spot_size=angle

    light=bpy.data.objects.new(name,light_data)
    bpy.context.collection.objects.link(light)
    light.location=position

    constraint=light.constraints.new(type="TRACK_TO")
    constraint.target=target
    constraint.track_axis="TRACK_NEGATIVE_Z"
    constraint.up_axis="UP_Y"

    return light

spotlights={
    name:create_spotlight(name,position,root)
    for name,position in SPOTLIGHT_POSITIONS.items()
}

CAMERA_POSITION=(-3,-2.6,2.2)
CAMERA_ROTATION=(62,6,-52.5)  # degrees, XYZ euler
CAMERA_FOV=77.3  # degrees, horizontal field of view

def create_camera(name,position,rotation,fov):
    camera_data=bpy.data.cameras.new(name)
    camera_data.lens_unit="FOV"
    camera_data.angle=radians(fov)

    camera=bpy.data.objects.new(name,camera_data)
    bpy.context.collection.objects.link(camera)

    camera.location=position
    camera.rotation_euler=[radians(x) for x in rotation]

    bpy.context.scene.camera=camera

    return camera

camera=create_camera("Camera",CAMERA_POSITION,CAMERA_ROTATION,CAMERA_FOV)
