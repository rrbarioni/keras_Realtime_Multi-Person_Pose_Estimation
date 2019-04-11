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

class Camera:
    def __init__(self, x, y, z, horizontal_fov, vertical_fov, fx, fy, cx, cy):
        self.x = x
        self.y = y
        self.z = z

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
            self.x + (screen_joint.x - self.cx) / self.fx,
            self.y + (screen_joint.y - self.cy) / self.fy,
            self.z + self.near)

    def get_screen_world_joints(self, screen_joints):
        return dict({
            (key, self.screen_to_world(value))
            for (key,value) in screen_joints.items() })

class Render:
    def __init__(self, width, height, left_camera, left_screen_world_joints, 
        right_camera, right_screen_world_joints):
        pygame.init()
        display = (width,height)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

        self.left_camera = left_camera
        self.right_camera = right_camera
        self.left_screen_world_joints = left_screen_world_joints
        self.right_screen_world_joints = right_screen_world_joints

        self.render_switches = {
            'camera': True,
            'screen_world_joints': True,
            'mouse_rotation_handler': True
        }

        self.colors = {
            'camera': (1, 0.2, 1),
            'bones': [
                (255,   0,   0), (255,  85,   0), (255, 170,   0),
                (255, 255,   0), (170, 255,   0), ( 85, 255,   0),
                (  0, 255,   0), (  0, 255,  85), (  0, 255, 170),
                (  0, 255, 255), (  0, 170, 255), (  0,  85, 255),
                (  0,   0, 255), ( 85,   0, 255), (170,   0, 255),
                (255,   0, 255), (255,   0, 170), (255,   0,  85),
                (255,   0,   0)],
            'rays': (0.6, 0.6, 0.6)
        }

        self.front_vectors_size = 0.3
        self.rays_size = 3

        self.last_mouse_pos = pygame.mouse.get_pos()
        self.delta_mouse_pos = (0,0)

        gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
        glTranslatef(0, 0, -5)
        glRotatef(90, 0, 1, 0)

        self.orientation = (0, 0, -1)

    def draw_axis_origin(self):
        glLineWidth(4)
        axis_length = 0.3

        # X axis
        glBegin(GL_LINES)
        glColor3fv((1, 0, 0))
        glVertex3fv((0, 0, 0))
        glVertex3fv((axis_length, 0, 0))
        glEnd()
        
        # Y axis
        glBegin(GL_LINES)
        glColor3fv((0, 0, 1))
        glVertex3fv((0, 0, 0))
        glVertex3fv((0, axis_length, 0))
        glEnd()

        # Z axis
        glBegin(GL_LINES)
        glColor3fv((0, 1, 0))
        glVertex3fv((0, 0, 0))
        glVertex3fv((0, 0, axis_length))
        glEnd()

    def draw_camera(self, camera):
        glLineWidth(1)
        glColor3fv(self.colors['camera'])

        cx = camera.x
        cy = camera.y
        cz = camera.z

        half_camera_width = camera.world_screen_width / 2
        half_camera_height = camera.world_screen_height / 2

        glBegin(GL_LINES)
        glVertex3fv((cx, cy, cz))
        glVertex3fv((
            cx - half_camera_width,
            cy - half_camera_height,
            cz + camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((cx, cy, cz))
        glVertex3fv((
            cx - half_camera_width,
            cy + half_camera_height,
            cz + camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((cx, cy, cz))
        glVertex3fv((
            cx + half_camera_width,
            cy - half_camera_height,
            cz + camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((cx, cy, cz))
        glVertex3fv((
            cx + half_camera_width,
            cy + half_camera_height,
            cz + camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((
            cx - half_camera_width,
            cy - half_camera_height,
            cz + camera.near))
        glVertex3fv((
            cx - half_camera_width,
            cy + half_camera_height,
            cz + camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((
            cx - half_camera_width,
            cy + half_camera_height,
            cz + camera.near))
        glVertex3fv((
            cx + half_camera_width,
            cy + half_camera_height,
            cz + camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((
            cx + half_camera_width,
            cy + half_camera_height,
            cz + camera.near))
        glVertex3fv((
            cx + half_camera_width,
            cy - half_camera_height,
            cz + camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((
            cx + half_camera_width,
            cy - half_camera_height,
            cz + camera.near))
        glVertex3fv((
            cx - half_camera_width,
            cy - half_camera_height,
            cz + camera.near))
        glEnd()

    def draw_skeleton(self, joints):
        glLineWidth(1)

        glBegin(GL_LINES)
        for (i, bone) in bones_list.items():
            glColor3fv(self.colors['bones'][i])
            if bone[0] in joints and bone[1] in joints:
                glVertex3fv((
                    joints[bone[0]].x, joints[bone[0]].y, joints[bone[0]].z))
                glVertex3fv((
                    joints[bone[1]].x, joints[bone[1]].y, joints[bone[1]].z))
        glEnd()

    def draw_rays(self, camera, joints):
        glLineWidth(1)
        glColor3fv(self.colors['rays'])

        cx = camera.x
        cy = camera.y
        cz = camera.z

        for joint in joints.values():
            glBegin(GL_LINES)
            glVertex3fv((cx, cy, cz))
            glVertex3fv((
                (cx + joint.x) * self.rays_size,
                (cy + joint.y) * self.rays_size,
                (cz + joint.z) * self.rays_size))
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
                glTranslatef(0, 0, 0.1)
            if keys_pressed[K_s]:
                glTranslatef(0, 0, -0.1)
            if keys_pressed[K_a]:
                glTranslatef(-0.1, 0, 0)
            if keys_pressed[K_d]:
                glTranslatef(0.1, 0, 0)

            # glRotatef(1, 0, 1, 0)
            # glTranslatef(0, 0, 0.01)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self.draw_axis_origin()

            if self.render_switches['camera']:
                self.draw_camera(self.left_camera)
                self.draw_camera(self.right_camera)

            if self.left_screen_world_joints is not None:
                self.draw_rays(
                    self.left_camera, self.left_screen_world_joints)
                self.draw_skeleton(self.left_screen_world_joints)
            if self.right_screen_world_joints is not None:
                self.draw_rays(
                    self.right_camera, self.right_screen_world_joints)
                self.draw_skeleton(self.right_screen_world_joints)

            pygame.display.flip()
            pygame.time.wait(10)


# test:
# ORIGINAL:
# camera = Camera(
#     horizontal_fov=70,
#     vertical_fov=60,
#     fx=1069.44,
#     fy=-1065.81,
#     cx=982.03,
#     cy=540.08,
# )
# bones_list = {
#     'neck':          ('neck',       'head'),
#     'l_collar_bone': ('l_shoulder', 'neck'),
#     'l_upper_arm':   ('l_elbow',    'l_shoulder'),
#     'l_fore_arm':    ('l_wrist',    'l_elbow'),
#     'r_collar_bone': ('r_shoulder', 'neck'),
#     'r_upper_arm':   ('r_elbow',    'r_shoulder'),
#     'r_fore_arm':    ('r_wrist',    'r_elbow'),
#     'l_spine':       ('l_pelvis',   'neck'),
#     'l_femur':       ('l_knee',     'l_pelvis'),
#     'l_shin':        ('l_foot',     'l_knee'),
#     'r_spine':       ('r_pelvis',   'neck'),
#     'r_femur':       ('r_knee',     'r_pelvis'),
#     'r_shin':        ('r_foot',     'r_knee'),
# }

# screen_joints = {
#     'head':       Vector2d(948, 394),
#     'l_shoulder': Vector2d(858, 533),
#     'l_elbow':    Vector2d(824, 657),
#     'l_wrist':    Vector2d(824, 776),
#     'r_shoulder': Vector2d(1025, 532),
#     'r_elbow':    Vector2d(1041, 639),
#     'r_wrist':    Vector2d(1027, 744),
#     'l_pelvis':   Vector2d(895, 790),
#     'l_knee':     Vector2d(886, 957),
#     'l_foot':     Vector2d(874, 1107),
#     'r_pelvis':   Vector2d(977, 791),
#     'r_knee':     Vector2d(995, 948),
#     'r_foot':     Vector2d(1007, 1096),
#     'neck':       Vector2d(946, 473),
# }
# screen_world_joints = camera.get_screen_world_joints(screen_joints)

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
bones_list = {
    0:  ('Head',      'Neck'),
    1:  ('Neck',      'RShoulder'),
    2:  ('RShoulder', 'RElbow'),
    3:  ('RElbow',    'RWrist'),
    4:  ('RWrist',    'RFinger'),
    5:  ('Neck',      'LShoulder'),
    6:  ('LShoulder', 'LElbow'),
    7:  ('LElbow',    'LWrist'),
    8:  ('LWrist',    'LFinger'),
    9:  ('Neck',      'RHip'),
    10: ('RHip',      'RKnee'),
    11: ('RKnee',     'RAnkle'),
    12: ('RAnkle',    'RToe'),
    13: ('Neck',      'LHip'),
    14: ('LHip',      'LKnee'),
    15: ('LKnee',     'LAnkle'),
    16: ('LAnkle',    'LToe') }

left_camera = Camera(
    x=0,
    y=0,
    z=0,
    horizontal_fov=60,
    vertical_fov=60,
    fx=368,
    fy=368,
    cx=184,
    cy=184)
left_screen_joints = {
    'Head':       Vector2d(366, 278),
    'Neck':       Vector2d(328, 268),
    'RShoulder':  Vector2d(293, 219),
    'RElbow':     Vector2d(248, 203),
    'RWrist':     Vector2d(216, 201),
    'RFinger':    Vector2d(207, 198),
    'LShoulder':  Vector2d(277, 288),
    'LElbow':     Vector2d(224, 280),
    'LWrist':     Vector2d(190, 271),
    'LFinger':    Vector2d(178, 265),
    'RHip':       Vector2d(246, 228),
    'RKnee':      Vector2d(235, 228),
    'RAnkle':     Vector2d(237, 225),
    'RToe':       Vector2d(235, 225),
    'LHip':       Vector2d(245, 244), 
    'LKnee':      Vector2d(233, 239), 
    'LAnkle':     Vector2d(234, 234), 
    'RToe':       Vector2d(226, 235) }
left_screen_world_joints = left_camera.get_screen_world_joints(
    left_screen_joints)
left_image = 'left_egocap_renderer_test.jpg'

right_camera = Camera(
    x=0,
    y=1.3,
    z=0,
    horizontal_fov=60,
    vertical_fov=60,
    fx=368,
    fy=368,
    cx=184,
    cy=184)
right_screen_joints = {
    'Head':       Vector2d(366, 111),
    'Neck':       Vector2d(334, 118),
    'RShoulder':  Vector2d(293, 109),
    'RElbow':     Vector2d(251, 126),
    'RWrist':     Vector2d(223, 134),
    'RFinger':    Vector2d(208, 139),
    'LShoulder':  Vector2d(294, 184),
    'LElbow':     Vector2d(241, 202),
    'LWrist':     Vector2d(232, 181),
    'LFinger':    Vector2d(224, 205),
    'RHip':       Vector2d(250, 169),
    'RKnee':      Vector2d(238, 175),
    'RAnkle':     Vector2d(237, 182),
    'RToe':       Vector2d(233, 180),
    'LHip':       Vector2d(243, 184),
    'LKnee':      Vector2d(231, 181),
    'LAnkle':     Vector2d(233, 187),
    'RToe':       Vector2d(227, 184) }
right_screen_world_joints = right_camera.get_screen_world_joints(
    right_screen_joints)
right_image = 'right_egocap_renderer_test.jpg'

Render(1200, 900, left_camera, left_screen_world_joints, right_camera, right_screen_world_joints).run()
