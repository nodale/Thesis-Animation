import bpy
import numpy as np
import re
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


def animate(root,position,quaternion,fps=200,stride=1,frame_offset=0):

    root.rotation_mode="QUATERNION"
    channelbag=create_channelbag(root,"DroneRootAction")

    n=len(position)
    frames=np.arange(1,n+1,stride,dtype=np.float64)+frame_offset

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

    return n

def animate_props(props,rpm,dt=1/200,stride=1,frame_offset=0):

    direction=np.array([-1,1,-1,1])

    angle=np.concatenate([
        np.zeros((1,4)),
        np.cumsum(rpm[1:]*direction*(2*np.pi/60)*dt,axis=0),
    ])

    n=len(angle)
    frames=np.arange(1,n+1,stride,dtype=np.float64)+frame_offset
    angle_sub=angle[::stride]

    for j,prop in enumerate(props):
        channelbag=create_channelbag(prop,f"{prop.name}Action")
        fcurve=channelbag.fcurves.new(data_path="rotation_euler",index=2)
        insert_keyframes(fcurve,frames,angle_sub[:,j])

@dataclass
class Mesh:
    file:str
    rotation:tuple=(0,0,0)

@dataclass
class ShaderConfig:
    base_color:       tuple = (0.8, 0.8, 0.8, 1.0)  # RGBA 0-1
    metallic:         float = 0.0
    roughness:        float = 0.5
    emission_color:   tuple = (0.0, 0.0, 0.0)
    emission_strength:float = 0.0
    alpha:            float = 1.0

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
THESIS_DIR=Path("~/Thesis/Thesis-Animation").expanduser()

INFER_DATASET="cls_out10"  # data_inference/<name>
INFER_DIR=THESIS_DIR/"data_inference"/INFER_DATASET
DATA_INFER=INFER_DIR/"flight_infer.npz"

# The ground-truth drone always uses the same log_20* flight that the inferred
# dataset's own .ulg was generated from, so the two drones show the same flight.
LOG_ID=re.match(r"(log_\d+)_",next(INFER_DIR.glob("log_*.ulg")).name).group(1)
DATA=THESIS_DIR/"data_real"/LOG_ID/"flight.npz"

FPS=30

DRONE_SHADERS={
    "":      ShaderConfig(base_color=(0.05, 0.05, 0.95, 0.8), metallic=0.0, roughness=1.0),
    "_Infer":ShaderConfig(base_color=(0.95, 0.05, 0.05, 0.8), metallic=0.0, roughness=1.0),
}

# Keep every Nth sample as a keyframe (data is at FPS Hz already; stride=1 -> every sample).
# Keyframe timing (frame numbers) is unchanged, so playback speed/duration is unaffected.
STRIDE=1

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

def create_drone(suffix,data_path,fps,stride,frame_offset=0):
    def import_stl(name,m):
        bpy.ops.object.select_all(action="DESELECT")
        bpy.ops.wm.stl_import(filepath=str(STL_DIR/m.file))
        o=bpy.context.selected_objects[0]
        o.name=f"{name}{suffix}"
        o.rotation_euler=[radians(x) for x in m.rotation]
        return o

    objects={n:import_stl(n,m) for n,m in MESHES.items()}

    body=objects["body"]
    root=create_root(body)
    root.name=f"DroneRoot{suffix}"

    for n,p in MOTOR_POSITIONS.items():
        objects[n].location=p
        objects[n].parent=body

    data=np.load(data_path)
    position=data["position"]
    quaternion=data["quaternion"]
    rpm=data["motor"]

    print(f"[{suffix or 'ground_truth'}] loaded {len(position)} frames from {data_path}, frame_offset={frame_offset}")

    n_frames=animate(root,position,quaternion,fps=fps,stride=stride,frame_offset=frame_offset)

    animate_props(
        [
            objects["prop_fr"],
            objects["prop_fl"],
            objects["prop_rl"],
            objects["prop_rr"],
        ],
        rpm,
        dt=1/fps,
        stride=stride,
        frame_offset=frame_offset
    )

    return root,objects,n_frames


DRONE_LABELS={
    "":      "ground truth",
    "_Infer":"inferred",
}

# Height above drone origin (metres) and text appearance
LABEL_Z_OFFSET=0.4
LABEL_SIZE=0.15       # font size in metres
LABEL_EXTRUDE=0.0     # 0 = flat text


def create_label(text,root,camera,z_offset=LABEL_Z_OFFSET,size=LABEL_SIZE,extrude=LABEL_EXTRUDE):
    curve=bpy.data.curves.new(name=f"LabelCurve_{text}",type="FONT")
    curve.body=text
    curve.size=size
    curve.extrude=extrude
    curve.align_x="CENTER"
    curve.align_y="CENTER"

    obj=bpy.data.objects.new(f"Label_{text}",curve)
    bpy.context.collection.objects.link(obj)

    # Follow drone position in world space (no parent) so world-Z lock works correctly
    obj.location=(0,0,z_offset)

    loc=obj.constraints.new(type="COPY_LOCATION")
    loc.target=root
    loc.use_offset=True

    rot=obj.constraints.new(type="COPY_ROTATION")
    rot.target=camera

    return obj


def apply_shader(objects,shader,name):
    mat=bpy.data.materials.new(name=name)
    mat.use_nodes=True
    bsdf=mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value=shader.base_color
    bsdf.inputs["Metallic"].default_value=shader.metallic
    bsdf.inputs["Roughness"].default_value=shader.roughness
    bsdf.inputs["Emission Color"].default_value=(*shader.emission_color,1.0)
    bsdf.inputs["Emission Strength"].default_value=shader.emission_strength
    bsdf.inputs["Alpha"].default_value=shader.alpha
    for obj in objects.values():
        if obj.data.materials:
            obj.data.materials[0]=mat
        else:
            obj.data.materials.append(mat)


# Both flight.npz/flight_infer.npz record t_start_us: the absolute ULog-internal moment
# their own timestamp=0 corresponds to. Each file's own resampling window can start at a
# slightly different real moment, so frame 1 of each drone isn't guaranteed to be the same
# instant unless corrected for -- shift the second drone's keyframes by the difference.
_data_meta=np.load(DATA)
_data_infer_meta=np.load(DATA_INFER)
_frame_offset_infer=round((_data_infer_meta["t_start_us"]-_data_meta["t_start_us"])/1e6*FPS)
print(f"inferred-drone frame offset: {_frame_offset_infer} frames "
      f"({(_data_infer_meta['t_start_us']-_data_meta['t_start_us'])/1e6:.3f}s)")

root,objects,n_frames=create_drone("",DATA,FPS,STRIDE)
apply_shader(objects,DRONE_SHADERS[""],"DroneShader")

root_infer,objects_infer,n_frames_infer=create_drone("_Infer",DATA_INFER,FPS,STRIDE,frame_offset=_frame_offset_infer)
apply_shader(objects_infer,DRONE_SHADERS["_Infer"],"DroneShader_Infer")

bpy.context.scene.frame_end=max(n_frames,n_frames_infer+_frame_offset_infer)

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

# Camera animation: set CAMERA_ANIM_ENABLED=True to keyframe position/rotation over time.
# Positions are (x,y,z); rotations are (rx,ry,rz) in degrees (XYZ Euler).
# Frame range is in scene frames (1-based, matching the drone keyframes).
CAMERA_ANIM_ENABLED=True
CAMERA_ANIM_START_FRAME=100
CAMERA_ANIM_END_FRAME=130          # set to None to use bpy.context.scene.frame_end

CAMERA_START_POSITION=(-3,-2.6,2.2)
CAMERA_END_POSITION=(-1.5,-2.4,3.8)

CAMERA_START_ROTATION=(62,6,-52.5)   # degrees
CAMERA_END_ROTATION=(40,5,-40.0)     # degrees


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


def animate_camera(camera,start_frame,end_frame,
                   start_pos,end_pos,
                   start_rot,end_rot):
    """Linearly interpolate camera position and rotation between two frames."""

    channelbag=create_channelbag(camera,"CameraAction")

    frames=np.array([start_frame,end_frame],dtype=np.float64)

    for axis,vals in enumerate(zip(start_pos,end_pos)):
        fcurve=channelbag.fcurves.new(data_path="location",index=axis)
        insert_keyframes(fcurve,frames,np.array(vals,dtype=np.float64))

    for axis,(s,e) in enumerate(zip(start_rot,end_rot)):
        fcurve=channelbag.fcurves.new(data_path="rotation_euler",index=axis)
        insert_keyframes(fcurve,frames,np.array([radians(s),radians(e)],dtype=np.float64))


camera=create_camera("Camera",CAMERA_POSITION,CAMERA_ROTATION,CAMERA_FOV)

create_label(DRONE_LABELS[""],root,camera)
create_label(DRONE_LABELS["_Infer"],root_infer,camera)

if CAMERA_ANIM_ENABLED:
    anim_end=CAMERA_ANIM_END_FRAME if CAMERA_ANIM_END_FRAME is not None else bpy.context.scene.frame_end
    animate_camera(
        camera,
        CAMERA_ANIM_START_FRAME,anim_end,
        CAMERA_START_POSITION,CAMERA_END_POSITION,
        CAMERA_START_ROTATION,CAMERA_END_ROTATION,
    )
