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

TRAIL_RADIUS=0.008   # tube radius for trajectory line in metres
TRAIL_LENGTH=200     # sliding window in frames (0 = show full history)
TRAIL_OPACITY=0.8    # max alpha at the newest (tip) end of the trail (0–1)

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


def create_label(text,root,camera,color,z_offset=LABEL_Z_OFFSET,size=LABEL_SIZE,extrude=LABEL_EXTRUDE):
    curve=bpy.data.curves.new(name=f"LabelCurve_{text}",type="FONT")
    curve.body=text
    curve.size=size
    curve.extrude=extrude
    curve.align_x="CENTER"
    curve.align_y="CENTER"

    obj=bpy.data.objects.new(f"Label_{text}",curve)
    bpy.context.collection.objects.link(obj)

    obj.location=(0,0,z_offset)

    loc=obj.constraints.new(type="COPY_LOCATION")
    loc.target=root
    loc.use_offset=True

    rot=obj.constraints.new(type="COPY_ROTATION")
    rot.target=camera

    mat=bpy.data.materials.new(f"LabelMat_{text}")
    mat.use_nodes=True
    mn=mat.node_tree.nodes; ml=mat.node_tree.links; mn.clear()
    emit=mn.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value=(*color[:3],1.0)
    emit.inputs["Strength"].default_value=2.0
    out=mn.new("ShaderNodeOutputMaterial")
    ml.new(emit.outputs["Emission"],out.inputs["Surface"])
    curve.materials.append(mat)

    return obj


def _create_trail_curve(name,position,n_frames,frame_offset,color,
                        radius=TRAIL_RADIUS,trail_length=TRAIL_LENGTH,opacity=TRAIL_OPACITY,
                        hide_frame=None):
    pos_bl=frd_to_blender(position)
    n=len(pos_bl)

    curve_data=bpy.data.curves.new(f"Trail_{name}","CURVE")
    curve_data.dimensions="3D"
    curve_data.bevel_depth=radius
    curve_data.use_fill_caps=True

    spline=curve_data.splines.new("POLY")
    spline.points.add(n-1)
    co=np.ones((n,4),dtype=np.float32)
    co[:,:3]=pos_bl
    spline.points.foreach_set("co",co.ravel())

    obj=bpy.data.objects.new(f"Trail_{name}",curve_data)
    bpy.context.collection.objects.link(obj)

    # Blender auto-generates UV on bevelled curves: U=around tube, V=along length (0→1).
    # Use V as the fade factor: transparent at old tail (0) and new tip (1), opaque between.
    mat=bpy.data.materials.new(f"TrailMat_{name}")
    mat.use_nodes=True
    mat.blend_method="HASHED"
    tree=mat.node_tree; mn=tree.nodes; ml=tree.links
    mn.clear()
    tex=mn.new("ShaderNodeTexCoord")
    sep=mn.new("ShaderNodeSeparateXYZ")
    ramp=mn.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation="LINEAR"
    el=ramp.color_ramp.elements
    el[0].position=0.0; el[0].color=(0,0,0,0)       # old tail: transparent
    mid=el.new(0.2);    mid.color=(1,1,1,opacity)    # ramps up quickly from tail
    el[2].position=1.0; el[2].color=(0,0,0,0)        # new tip: transparent
    bsdf=mn.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value=color
    bsdf.inputs["Roughness"].default_value=1.0
    out=mn.new("ShaderNodeOutputMaterial")
    ml.new(tex.outputs["UV"],sep.inputs["Vector"])
    ml.new(sep.outputs["Y"],ramp.inputs["Fac"])
    ml.new(ramp.outputs["Alpha"],bsdf.inputs["Alpha"])
    ml.new(bsdf.outputs["BSDF"],out.inputs["Surface"])
    curve_data.materials.append(mat)

    # animate bevel_factor_end: 0→1 over the drone's frame range
    action=bpy.data.actions.new(f"Trail_{name}Action")
    slot=action.slots.new(id_type='CURVE',name=curve_data.name)
    curve_data.animation_data_create()
    curve_data.animation_data.action=action
    curve_data.animation_data.action_slot=slot
    layer=action.layers.new(name="Layer")
    strip=layer.strips.new(type='KEYFRAME')
    cb=strip.channelbag(slot,ensure=True)

    fc_end=cb.fcurves.new(data_path="bevel_factor_end")
    insert_keyframes(fc_end,
                     np.array([frame_offset+1, frame_offset+n_frames],dtype=np.float64),
                     np.array([0.0,1.0],dtype=np.float64))

    if trail_length>0:
        fc_start=cb.fcurves.new(data_path="bevel_factor_start")
        insert_keyframes(fc_start,
                         np.array([frame_offset+1,
                                   frame_offset+trail_length+1,
                                   frame_offset+n_frames],dtype=np.float64),
                         np.array([0.0, 0.0,
                                   (n_frames-trail_length)/n_frames],dtype=np.float64))

    if hide_frame is not None:
        obj_cb=create_channelbag(obj,f"Trail_{name}VisAction")
        for dp in ("hide_render","hide_viewport"):
            fc=obj_cb.fcurves.new(data_path=dp)
            insert_keyframes(fc,
                             np.array([hide_frame-1, hide_frame],dtype=np.float64),
                             np.array([0.0, 1.0],dtype=np.float64),
                             interpolation="CONSTANT")

    return obj


def create_trail(name,position,n_frames,frame_offset,color,
                 radius=TRAIL_RADIUS,trail_length=TRAIL_LENGTH,opacity=TRAIL_OPACITY,
                 reanchor_frames=None):
    if reanchor_frames is None or len(reanchor_frames)==0:
        return [_create_trail_curve(name,position,n_frames,frame_offset,color,radius,trail_length,opacity)]
    boundaries=np.concatenate([[0],reanchor_frames,[n_frames]]).astype(int)
    objs=[]
    last_i=len(boundaries)-2
    for i,(s,e) in enumerate(zip(boundaries[:-1],boundaries[1:])):
        if e-s<2:
            continue
        hide_frame=None if i==last_i else frame_offset+e+1
        objs.append(_create_trail_curve(
            f"{name}_{i}",position[s:e],e-s,frame_offset+s,color,radius,trail_length,opacity,
            hide_frame=hide_frame
        ))
    return objs


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
create_trail("gt",_data_meta["position"],n_frames,0,DRONE_SHADERS[""].base_color)

root_infer,objects_infer,n_frames_infer=create_drone("_Infer",DATA_INFER,FPS,STRIDE,frame_offset=_frame_offset_infer)
apply_shader(objects_infer,DRONE_SHADERS["_Infer"],"DroneShader_Infer")
_reanchor_frames=_data_infer_meta["reanchor_frames"] if "reanchor_frames" in _data_infer_meta else None
create_trail("infer",_data_infer_meta["position"],n_frames_infer,_frame_offset_infer,
             DRONE_SHADERS["_Infer"].base_color,reanchor_frames=_reanchor_frames)

bpy.context.scene.frame_end=max(n_frames,n_frames_infer+_frame_offset_infer)

GRID_SPACING=0.5           # metres between grid lines (fixed as grid expands)
GRID_MAX_SIZE=5.0          # full extent (metres) when fully expanded
GRID_LINE_THICKNESS=0.005  # wireframe tube radius in metres
GRID_COLOR=(0.8, 0.8, 0.8)
GRID_ALPHA=0.04

GRID_ANIM_START_FRAME=100
GRID_ANIM_END_FRAME=160

# Each plane: (name, rotation_euler_degrees)
GRID_PLANES={
    "grid_xy":(0,   0,  0),
    #"grid_xz":(90,  0,  0),
    #grid_yz":(90,  0, 90),
}


def build_grid_nodegroup(spacing, mat):
    ng=bpy.data.node_groups.new("GridNodeGroup","GeometryNodeTree")

    ng.interface.new_socket("Geometry",in_out="OUTPUT",socket_type="NodeSocketGeometry")
    size_sock=ng.interface.new_socket("Size",in_out="INPUT",socket_type="NodeSocketFloat")
    size_sock.default_value=0.0
    size_sock.min_value=0.0

    nodes=ng.nodes
    links=ng.links

    gi=nodes.new("NodeGroupInput")
    go=nodes.new("NodeGroupOutput")

    grid=nodes.new("GeometryNodeMeshGrid")

    # vertex count = ceil(Size / spacing) + 1
    divide=nodes.new("ShaderNodeMath"); divide.operation="DIVIDE"
    divide.inputs[1].default_value=spacing
    ceil=nodes.new("ShaderNodeMath"); ceil.operation="CEIL"
    add1=nodes.new("ShaderNodeMath"); add1.operation="ADD"
    add1.inputs[1].default_value=1.0
    fti=nodes.new("FunctionNodeFloatToInt"); fti.rounding_mode="ROUND"

    m2c=nodes.new("GeometryNodeMeshToCurve")

    circle=nodes.new("GeometryNodeCurvePrimitiveCircle")
    circle.inputs["Resolution"].default_value=4
    circle.inputs["Radius"].default_value=GRID_LINE_THICKNESS

    c2m=nodes.new("GeometryNodeCurveToMesh")
    c2m.inputs["Fill Caps"].default_value=False

    set_mat=nodes.new("GeometryNodeSetMaterial")
    set_mat.inputs["Material"].default_value=mat

    links.new(gi.outputs["Size"],grid.inputs["Size X"])
    links.new(gi.outputs["Size"],grid.inputs["Size Y"])

    links.new(gi.outputs["Size"],divide.inputs[0])
    links.new(divide.outputs["Value"],ceil.inputs[0])
    links.new(ceil.outputs["Value"],add1.inputs[0])
    links.new(add1.outputs["Value"],fti.inputs[0])
    links.new(fti.outputs["Integer"],grid.inputs["Vertices X"])
    links.new(fti.outputs["Integer"],grid.inputs["Vertices Y"])

    links.new(grid.outputs["Mesh"],m2c.inputs["Mesh"])
    links.new(m2c.outputs["Curve"],c2m.inputs["Curve"])
    links.new(circle.outputs["Curve"],c2m.inputs["Profile Curve"])
    links.new(c2m.outputs["Mesh"],set_mat.inputs["Geometry"])
    links.new(set_mat.outputs["Geometry"],go.inputs["Geometry"])

    return ng


def create_grids(
    spacing=GRID_SPACING,
    max_size=GRID_MAX_SIZE,
    color=GRID_COLOR,
    alpha=GRID_ALPHA,
    start_frame=GRID_ANIM_START_FRAME,
    end_frame=GRID_ANIM_END_FRAME,
):
    mat=bpy.data.materials.new("GridMat")
    mat.use_nodes=True
    bsdf=mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value=(*color,1.0)
    bsdf.inputs["Roughness"].default_value=1.0
    bsdf.inputs["Alpha"].default_value=alpha
    mat.blend_method="BLEND"

    ng=build_grid_nodegroup(spacing,mat)

    planes={}
    for name,(rx,ry,rz) in GRID_PLANES.items():
        mesh=bpy.data.meshes.new(name)
        obj=bpy.data.objects.new(name,mesh)
        bpy.context.collection.objects.link(obj)
        obj.rotation_euler=[radians(rx),radians(ry),radians(rz)]

        mod=obj.modifiers.new("Grid","NODES")
        mod.node_group=ng

        # animate Size: 0 at start_frame -> max_size at end_frame
        mod["Socket_1"]=0.0
        obj.keyframe_insert(data_path='modifiers["Grid"]["Socket_1"]',frame=start_frame)
        mod["Socket_1"]=max_size
        obj.keyframe_insert(data_path='modifiers["Grid"]["Socket_1"]',frame=end_frame)

        planes[name]=obj

    return planes


grid_planes=create_grids()

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
CAMERA_ANIM_END_FRAME=160          # set to None to use bpy.context.scene.frame_end

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

create_label(DRONE_LABELS[""],root,camera,DRONE_SHADERS[""].base_color)
create_label(DRONE_LABELS["_Infer"],root_infer,camera,DRONE_SHADERS["_Infer"].base_color)

if CAMERA_ANIM_ENABLED:
    anim_end=CAMERA_ANIM_END_FRAME if CAMERA_ANIM_END_FRAME is not None else bpy.context.scene.frame_end
    animate_camera(
        camera,
        CAMERA_ANIM_START_FRAME,anim_end,
        CAMERA_START_POSITION,CAMERA_END_POSITION,
        CAMERA_START_ROTATION,CAMERA_END_ROTATION,
    )
