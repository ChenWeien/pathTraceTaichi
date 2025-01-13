import taichi as ti
import numpy as np

ti.init(arch=ti.cuda)

@ti.data_oriented
class OBJLoader:
    def __init__(self):
        self.vertices = []
        self.faces = []
        self.normals = []
        self.uvs = []
    
    def load_obj(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            if line.startswith('#'): # Skip comments
                continue
                
            values = line.split()
            if not values:
                continue
                
            if values[0] == 'v':  # Vertex
                v = [float(x) for x in values[1:4]]
                self.vertices.append(v)
                
            elif values[0] == 'vn':  # Normal
                vn = [float(x) for x in values[1:4]]
                self.normals.append(vn)
                
            elif values[0] == 'vt':  # Texture coordinate
                vt = [float(x) for x in values[1:3]]
                self.uvs.append(vt)
                
            elif values[0] == 'f':  # Face
                # Handle different face formats

                for v in values[1:]:
                    w = v.split('/')
                    # OBJ indices are 1-based, converting to 0-based
                    face = int(w[0])-1
                    self.faces.append(face)
    
    def to_taichi_fields(self):
        # Convert to numpy arrays first
        vertices_np = np.array(self.vertices, dtype=np.float32)
        faces_np = np.array(self.faces, dtype=np.int32)
        
        # Create Taichi fields
        vertex_count = len(self.vertices)
        face_count = len(self.faces)
        vertex_field = ti.Vector.field(3, dtype=ti.f32, shape=vertex_count)
        face_field = ti.field(dtype=ti.i32, shape=face_count)
        
        # Copy data to fields
        vertex_field.from_numpy(vertices_np)
        face_field.from_numpy(faces_np)
        
        return vertex_field, face_field


N = 10

particles_pos = ti.Vector.field(3, dtype=ti.f32, shape = N)
points_pos = ti.Vector.field(3, dtype=ti.f32, shape = N)

@ti.kernel
def init_points_pos(points : ti.template()):
    for i in range(points.shape[0]):
        points[i] = [i for j in ti.static(range(3))]

init_points_pos(particles_pos)
init_points_pos(points_pos)

window = ti.ui.Window("Test for Drawing 3d-lines", (768, 768))
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(5, 2, 2)

objLoader = OBJLoader()
objLoader.load_obj("keqing.obj")
vertices, indices = objLoader.to_taichi_fields()

while window.running:
    if window.get_event(ti.ui.PRESS):
        if window.event.key == 'r': reset()
        elif window.event.key in [ti.ui.ESCAPE]: break
    camera.track_user_inputs(window, movement_speed=0.3, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.8, 0.8, 0.8))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

    scene.particles(particles_pos, color = (0.68, 0.26, 0.19), radius = 0.1)
    # Draw 3d-lines in the scene
    scene.lines(points_pos, color = (0.28, 0.68, 0.99), width = 5.0)
    #scene.mesh(vertices_3d, indices, normals, color, per_vertex_color, vertex_offset=0, vertex_count=10, index_offset=0, index_count=10, show_wireframe=True)

    scene.mesh(vertices, indices)

    canvas.scene(scene)
    window.show()
