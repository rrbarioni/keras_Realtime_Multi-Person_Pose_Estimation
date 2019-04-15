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

class PinholeCamera:
    def __init__(self, x, y, z, rx, ry, rz, horizontal_fov, vertical_fov, 
        fx, fy, cx, cy):
        self.p = Vector3d(x, y, z)
        self.r = Vector3d(rx, ry, rz)

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
            self.p.x + (screen_joint.x - self.cx) / self.fx,
            self.p.y + (screen_joint.y - self.cy) / self.fy,
            self.p.z + self.near)

    def get_screen_world_joints(self, screen_joints):
        return dict({
            (key, self.screen_to_world(value))
            for (key,value) in screen_joints.items() })

class Util:
    @staticmethod
    def nearest_point_two_lines(a, b, c, d):
        '''
        closest point between lines AB and CD
        '''

        da = (b - a).normalize()
        db = (d - c).normalize()
        dc = c - a

        dada = da.dot_product(da)
        dadb = da.dot_product(db)
        dadc = da.dot_product(dc)
        dbdb = db.dot_product(db)
        dbdc = db.dot_product(dc)

        d = a + \
            da * ((-dadb * dbdc + dadc * dbdb) / (dada * dbdb - dadb * dadb))
        e = b + \
            db * ((dadb * dadc - dbdc * dada) / (dada * dbdb - dadb * dadb))

        p = (d + e) * (1 / 2)

        return p

    def get_3d_skeleton(cp1, cp2, swj1, swj2):
        return { k: Util.nearest_point_two_lines(cp1, swj1[k], cp2, swj2[k])
            for k in swj1.keys() }

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

        self.skeleton_3d = Util.get_3d_skeleton(self.left_camera.p,
            self.right_camera.p, self.left_screen_world_joints,
            self.right_screen_world_joints)

        self.render_switches = {
            'camera': True,
            'screen_world_joints': True,
            'mouse_rotation_handler': True,
            'rays': True
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
        self.rays_size = 10

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

    def draw_pinhole_camera(self, camera):
        glLineWidth(1)
        glColor3fv(self.colors['camera'])

        cpx = camera.p.x
        cpy = camera.p.y
        cpz = camera.p.z

        half_camera_width = camera.world_screen_width / 2
        half_camera_height = camera.world_screen_height / 2

        glBegin(GL_LINES)
        glVertex3fv((cpx, cpy, cpz))
        glVertex3fv((
            cpx - half_camera_width,
            cpy - half_camera_height,
            cpz + camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((cpx, cpy, cpz))
        glVertex3fv((
            cpx - half_camera_width,
            cpy + half_camera_height,
            cpz + camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((cpx, cpy, cpz))
        glVertex3fv((
            cpx + half_camera_width,
            cpy - half_camera_height,
            cpz + camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((cpx, cpy, cpz))
        glVertex3fv((
            cpx + half_camera_width,
            cpy + half_camera_height,
            cpz + camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((
            cpx - half_camera_width,
            cpy - half_camera_height,
            cpz + camera.near))
        glVertex3fv((
            cpx - half_camera_width,
            cpy + half_camera_height,
            cpz + camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((
            cpx - half_camera_width,
            cpy + half_camera_height,
            cpz + camera.near))
        glVertex3fv((
            cpx + half_camera_width,
            cpy + half_camera_height,
            cpz + camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((
            cpx + half_camera_width,
            cpy + half_camera_height,
            cpz + camera.near))
        glVertex3fv((
            cpx + half_camera_width,
            cpy - half_camera_height,
            cpz + camera.near))
        glEnd()

        glBegin(GL_LINES)
        glVertex3fv((
            cpx + half_camera_width,
            cpy - half_camera_height,
            cpz + camera.near))
        glVertex3fv((
            cpx - half_camera_width,
            cpy - half_camera_height,
            cpz + camera.near))
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

        cpx = camera.p.x
        cpy = camera.p.y
        cpz = camera.p.z

        for joint in joints.values():
            glBegin(GL_LINES)
            glVertex3fv((cpx, cpy, cpz))
            glVertex3fv((
                joint.x + (joint.x - cpx) * self.rays_size,
                joint.y + (joint.y - cpy) * self.rays_size,
                joint.z + (joint.z - cpz) * self.rays_size))
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
                glTranslatef(0, 0, 0.05)
            if keys_pressed[K_s]:
                glTranslatef(0, 0, -0.05)
            if keys_pressed[K_a]:
                glTranslatef(-0.05, 0, 0)
            if keys_pressed[K_d]:
                glTranslatef(0.05, 0, 0)
            if keys_pressed[K_q]:
                glTranslatef(0, -0.05, 0)
            if keys_pressed[K_e]:
                glTranslatef(0, 0.05, 0)
            if keys_pressed[K_r]:
                self.render_switches['rays'] = not self.render_switches['rays']

            # glRotatef(1, 0, 1, 0)
            # glTranslatef(0, 0, 0.01)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            self.draw_axis_origin()

            if self.render_switches['camera']:
                self.draw_pinhole_camera(self.left_camera)
                self.draw_pinhole_camera(self.right_camera)
            
            if self.render_switches['rays']:
                self.draw_rays(self.left_camera, self.left_screen_world_joints)
                self.draw_rays(
                    self.right_camera, self.right_screen_world_joints)

            self.draw_skeleton(self.left_screen_world_joints)
            self.draw_skeleton(self.right_screen_world_joints)
            self.draw_skeleton(self.skeleton_3d)

            pygame.display.flip()
            pygame.time.wait(10)

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

left_camera = PinholeCamera(
    x=0, y=0, z=0,
    rx=0, ry=0, rz=0,
    horizontal_fov=60,
    vertical_fov=60,
    fx=368, fy=368, cx=184, cy=184)
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

right_camera = PinholeCamera(
    x=0, y=1, z=0,
    rx=0, ry=0, rz=0,
    horizontal_fov=60,
    vertical_fov=60,
    fx=368, fy=368, cx=184, cy=184)
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

Render(1200, 900, left_camera, left_screen_world_joints,
    right_camera, right_screen_world_joints).run()
