import math
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

class Vector2d:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ')'

    def __eq__(self, other):
        return ((self.x == other.x) and (self.y == other.y))
        
    def __hash__(self):
        return id(self)

class Vector3d:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return '(' + str(self.x) + ', ' + str(self.y) + ', ' + str(self.z) + ')'

    def __add__(self, other):
        return Vector3d(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3d(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scale):
        return Vector3d(self.x * scale, self.y * scale, self.z * scale)

    def __neg__(self):
        return Vector3d(-self.x, -self.y, -self.z)

    def __eq__(self, other):
        return ((self.x == other.x) \
            and (self.y == other.y) \
            and (self.z == other.z))

    def __hash__(self):
        return id(self)

    def normalize(self):
        xyz_max = max(abs(self.x), abs(self.y), abs(self.z))
        return Vector3d(self.x / xyz_max, self.y / xyz_max, self.z / xyz_max)

    def magnitude(self):
        return (self.x**2 + self.y**2 + self.z**2) ** (1/2)

    def distance(self, other):
        return (self - other).magnitude()

    def dot_product(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def cross_product(self, other):
        return Vector3d(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x)

    def avg_vector(self, other):
        return Vector3d(
            (self.x + other.x) / 2,
            (self.y + other.y) / 2,
            (self.z + other.z) / 2)

class Util:
    @staticmethod
    def dist_vector_point(vector, point):
        n = vector
        pa = point * -1
        d = pa - (n * pa.dot_product(n) / n.dot_product(n))
        
        return (d.dot_product(d)) ** (1/2)

    @staticmethod
    def angle_two_vectors_in_degrees(vector_1, vector_2):
        cos_angle = vector_1.dot_product(vector_2) / (vector_1.magnitude() * vector_2.magnitude())
        angle_rad = math.acos(cos_angle)
        angle_degrees = angle_rad * 180 / math.pi

        return angle_degrees

    @staticmethod
    def line_sphere_intersections(line, sphere_c, sphere_r):
        # ((lx * k) - cx) ** 2 + ((ly * k) - cy) ** 2 + ((lz * k) - cz) ** 2 = r ** 2
        
        a = (line.x ** 2) + (line.y ** 2) + (line.z ** 2)
        b = -2 * (line.x * sphere_c.x + line.y * sphere_c.y + line.z * sphere_c.z)
        c = (sphere_c.x ** 2) + (sphere_c.y ** 2) + (sphere_c.z ** 2) - (sphere_r ** 2)
        delta = (b ** 2) - (4 * a * c)

        if (delta < 0):
            return None, None, None

        k1 = (-b + delta ** (1/2)) / (2 * a)
        k2 = (-b - delta ** (1/2)) / (2 * a)

        intersection_1 = Vector3d(line.x * k1, line.y * k1, line.z * k1)
        intersection_2 = Vector3d(line.x * k2, line.y * k2, line.z * k2)

        return intersection_1, intersection_2, delta

    @staticmethod
    def fold(f, l, a):
        return a if (len(l) == 0) else Util.fold(f, l[1:], f(a, l[0]))

class Camera:
    def __init__(self, horizontal_fov, vertical_fov, fx, fy, cx, cy):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

        # only used on render
        self.horizontal_fov = horizontal_fov
        self.vertical_fov = vertical_fov
        self.near = 1
        self.world_screen_width =  2 * \
            math.tan(math.radians(self.horizontal_fov / 2)) * self.near
        self.world_screen_height = 2 * \
            math.tan(math.radians(self.vertical_fov / 2)) * self.near

    def screen_to_world(self, screen_joint):
        return Vector3d(
            (screen_joint.x - self.cx) / self.fx,
            (screen_joint.y - self.cy) / self.fy,
            self.near
        )

    def get_screen_world_joints(self, screen_joints):
        return dict({
            (key, self.screen_to_world(value))
            for (key,value) in screen_joints.items() })

class Render:
    def __init__(self, width, height, camera, screen_world_joints, ground_truth):
        pygame.init()
        display = (width,height)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

        self.camera = camera
        self.screen_world_joints = screen_world_joints
        self.ground_truth = ground_truth

        self.render_switches = {
            'camera': True,
            'screen_world_joints': True,
            'ground_truth': True,
            'mouse_rotation_handler': True
        }

        self.colors = {
            'camera': (1, 0.2, 1),
            'screen_world_joints': (1, 1, 0),
            'ground_truth': (0, 1, 0),
            'bones': [
                (255,   0,   0), (255,  85,   0), (255, 170,   0), (255, 255,   0), (170, 255,   0),
                ( 85, 255,   0), (  0, 255,   0), (  0, 255,  85), (  0, 255, 170), (  0, 255, 255),
                (  0, 170, 255), (  0,  85, 255), (  0,   0, 255), ( 85,   0, 255), (170,   0, 255),
                (255,   0, 255), (255,   0, 170), (255,   0,  85), (255,   0,   0)
            ]
            'rays': (0.6, 0.6, 0.6),
            'default': (1, 1, 1)
        }

        self.front_vectors_size = 0.3
        self.rays_size = 3

        self.last_mouse_pos = pygame.mouse.get_pos()
        self.delta_mouse_pos = (0,0)

        gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
        glTranslatef(0, 0, -5)
        glRotatef(90, 0, 1, 0)

        self.orientation = (0, 0, -1)

    def set_camera(self, camera):
        self.camera = camera
        return self

    def set_screen_world_joints(self, screen_world_joints):
        self.screen_world_joints = screen_world_joints
        return self

    def set_ground_truth(self, ground_truth):
        self.ground_truth = ground_truth
        return self

    def draw_axis_origin(self):
        glLineWidth(4)
        axis_length = 0.3

        # X axis
        glBegin(GL_LINES)
        glColor3fv((1,0,0))
        glVertex3fv((0,0,0))
        glVertex3fv((axis_length, 0, 0))
        glEnd()
        
        # Y axis
        glBegin(GL_LINES)
        glColor3fv((0,0,1))
        glVertex3fv((0,0,0))
        glVertex3fv((0, axis_length, 0))
        glEnd()

        # Z axis
        glBegin(GL_LINES)
        glColor3fv((0,1,0))
        glVertex3fv((0,0,0))
        glVertex3fv((0, 0, axis_length))
        glEnd()

    def draw_camera(self):
        glLineWidth(1)
        glColor3fv(self.colors['camera'])

        half_camera_width = self.camera.world_screen_width / 2
        half_camera_height = self.camera.world_screen_height / 2

        glBegin(GL_LINES)
        glVertex3fv((0,0,0))
        glVertex3fv((-half_camera_width, -half_camera_height, self.camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((0,0,0))
        glVertex3fv((-half_camera_width, half_camera_height, self.camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((0,0,0))
        glVertex3fv((half_camera_width, -half_camera_height, self.camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((0,0,0))
        glVertex3fv((half_camera_width, half_camera_height, self.camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((-half_camera_width, -half_camera_height, self.camera.near))
        glVertex3fv((-half_camera_width, half_camera_height, self.camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((-half_camera_width, half_camera_height, self.camera.near))
        glVertex3fv((half_camera_width, half_camera_height, self.camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((half_camera_width, half_camera_height, self.camera.near))
        glVertex3fv((half_camera_width, -half_camera_height, self.camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((half_camera_width, -half_camera_height, self.camera.near))
        glVertex3fv((-half_camera_width, -half_camera_height, self.camera.near))
        glEnd()

    def draw_skeleton(self, joints, type='default'):
        glLineWidth(1)
        glColor3fv(self.colors[type])

        glBegin(GL_LINES)
        for bone in bones_list.values():
            if bone[0] in joints and bone[1] in joints:
                glVertex3fv((
                    joints[bone[0]].x, joints[bone[0]].y, joints[bone[0]].z))
                glVertex3fv((
                    joints[bone[1]].x, joints[bone[1]].y, joints[bone[1]].z))
        glEnd()

    def draw_rays(self, joints):
        glLineWidth(1)
        glColor3fv(self.colors['rays'])

        glBegin(GL_LINES)
        for joint in joints.values():
            glVertex3fv((0,0,0))
            glVertex3fv((
                joint.x * self.rays_size,
                joint.y * self.rays_size,
                joint.z * self.rays_size))
        glEnd()

    def mouse_rotation_handler(self):
        current_mouse_pos = pygame.mouse.get_pos()
        self.delta_mouse_pos = (\
            current_mouse_pos[0] - self.last_mouse_pos[0],\
            current_mouse_pos[1] - self.last_mouse_pos[1]
        )
        self.last_mouse_pos = current_mouse_pos
        glRotatef(self.delta_mouse_pos[0], 0, 1, 0)
        glRotatef(self.delta_mouse_pos[1], 1, 0, 0)

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.render_switches['mouse_rotation_handler'] = \
                        not self.render_switches['mouse_rotation_handler']

            if self.render_switches['mouse_rotation_handler']:
                self.mouse_rotation_handler()

            keys_pressed = pygame.key.get_pressed()
            if keys_pressed[K_w]:
                glTranslatef(0,0,0.1)
            if keys_pressed[K_s]:
                glTranslatef(0,0,-0.1)
            if keys_pressed[K_a]:
                glTranslatef(-0.1,0,0)
            if keys_pressed[K_d]:
                glTranslatef(0.1,0,0)

            # glRotatef(1, 0, 1, 0)
            # glTranslatef(0, 0, 0.01)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self.draw_axis_origin()

            if self.render_switches['camera']:
                self.draw_camera()

            if self.screen_world_joints is not None:
                self.draw_rays(self.screen_world_joints)
                self.draw_skeleton(
                    self.screen_world_joints, type='screen_world_joints')

            if self.ground_truth is not None:
                self.draw_skeleton(self.ground_truth, type='ground_truth')

            pygame.display.flip()
            pygame.time.wait(10)


# test:
# ORIGINAL:
bones_list = {
    'neck':          ('neck',       'head'),
    'l_collar_bone': ('l_shoulder', 'neck'),
    'l_upper_arm':   ('l_elbow',    'l_shoulder'),
    'l_fore_arm':    ('l_wrist',    'l_elbow'),
    'r_collar_bone': ('r_shoulder', 'neck'),
    'r_upper_arm':   ('r_elbow',    'r_shoulder'),
    'r_fore_arm':    ('r_wrist',    'r_elbow'),
    'l_spine':       ('l_pelvis',   'neck'),
    'l_femur':       ('l_knee',     'l_pelvis'),
    'l_shin':        ('l_foot',     'l_knee'),
    'r_spine':       ('r_pelvis',   'neck'),
    'r_femur':       ('r_knee',     'r_pelvis'),
    'r_shin':        ('r_foot',     'r_knee'),
}

screen_joints = {
    'head':       Vector2d(948, 394),
    'l_shoulder': Vector2d(858, 533),
    'l_elbow':    Vector2d(824, 657),
    'l_wrist':    Vector2d(824, 776),
    'r_shoulder': Vector2d(1025, 532),
    'r_elbow':    Vector2d(1041, 639),
    'r_wrist':    Vector2d(1027, 744),
    'l_pelvis':   Vector2d(895, 790),
    'l_knee':     Vector2d(886, 957),
    'l_foot':     Vector2d(874, 1107),
    'r_pelvis':   Vector2d(977, 791),
    'r_knee':     Vector2d(995, 948),
    'r_foot':     Vector2d(1007, 1096),
    'neck':       Vector2d(946, 473),
}
screen_world_joints = camera.get_screen_world_joints(screen_joints)

# ground_truth = {
#     "head":       Vector3d(-0.05539,  0.27005, 1.95791),
#     "l_shoulder": Vector3d(-0.21802,  0.01493, 1.90445),
#     "l_elbow":    Vector3d(-0.27991, -0.20893, 1.90757),
#     "l_wrist":    Vector3d(-0.27114, -0.40407, 1.82395),
#     "r_shoulder": Vector3d( 0.08004,  0.01386, 1.86651),
#     "r_elbow":    Vector3d( 0.09393, -0.16023, 1.71716),
#     "r_wrist":    Vector3d( 0.05694, -0.29506, 1.53766),
#     "l_pelvis":   Vector3d(-0.15150, -0.43901, 1.87078),
#     "l_knee":     Vector3d(-0.17072, -0.74263, 1.89426),
#     "l_foot":     Vector3d(-0.19390, -1.01891, 1.91696),
#     "r_pelvis":   Vector3d(-0.00689, -0.43827, 1.85139),
#     "r_knee":     Vector3d( 0.02322, -0.71868, 1.87110),
#     "r_foot":     Vector3d( 0.04517, -0.98975, 1.89516),
#     "neck":       Vector3d(-0.05940,  0.12449, 1.95275),
# }

# EGOCAP:
    # idx_in_egocap_str = ['Head', 'Neck', 'RShoulder', 'RElbow', 'RWrist',
    #     'RFinger', 'LShoulder', 'LElbow', 'LWrist', 'LFinger', 'RHip', 'RKnee',
    #     'RAnkle', 'RToe', 'LHip', 'LKnee', 'LAnkle', 'RToe']

    # joint_pairs = list(zip(
    #     [0, 1, 2, 3, 4, 1, 6, 7, 8, 1,  10, 11, 12, 1,  14, 15, 16],
    #     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]))
bones_list = {
    'neck':          ('neck',       'head'),
    'l_collar_bone': ('l_shoulder', 'neck'),
    'l_upper_arm':   ('l_elbow',    'l_shoulder'),
    'l_fore_arm':    ('l_wrist',    'l_elbow'),
    'r_collar_bone': ('r_shoulder', 'neck'),
    'r_upper_arm':   ('r_elbow',    'r_shoulder'),
    'r_fore_arm':    ('r_wrist',    'r_elbow'),
    'l_spine':       ('l_pelvis',   'neck'),
    'l_femur':       ('l_knee',     'l_pelvis'),
    'l_shin':        ('l_foot',     'l_knee'),
    'r_spine':       ('r_pelvis',   'neck'),
    'r_femur':       ('r_knee',     'r_pelvis'),
    'r_shin':        ('r_foot',     'r_knee'),
}

camera = Camera(
    horizontal_fov=70,
    vertical_fov=60,
    fx=1069.44,
    fy=-1065.81,
    cx=982.03,
    cy=540.08,
)

Render(1200, 900, camera, screen_world_joints, None).run()
