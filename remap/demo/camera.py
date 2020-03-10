
from math import *
from pyglet.gl import *
from pyglet.window import key, mouse
from vec3 import *

class DemoCamera:
    def __init__(self, width, height, center=(0,0,0), eye=(0,0,1), up=(0,1,0)):
        self.width = width
        self.height = height
        self.z = 1.0

        if eye == center:
            print >> sys.stderr, "?? eye and center position coincide"
            eye = (center[0],center[1],center[2]+1)
        self.center = vec3(center)
        self.eye    = vec3(eye)
        self.up     = vec3(up)
        self._fixup()
        self.mx = None
        self.my = None
        self.rotating_graphic = None
        self.translating_graphic = None


    def apply(self):
        """Load the current view to the OpenGL GL_PROJECTION and GL_MODELVIEW matrices."""
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = float(self.width)/float(self.height)
        glOrtho(-self.z*aspect, self.z*aspect, -self.z, +self.z, 0.05, 100)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(self.eye.x,    self.eye.y,    self.eye.z,
                  self.center.x, self.center.y, self.center.z,
                  self.up.x,     self.up.y,     self.up.z)

    def rotate(self, dphi, dtheta):
        x,y,z,r = self._getbasis()
        # First rotate horizontally (about y)
        x, z = x*cos(dphi) + z*sin(dphi), -x*sin(dphi) + z*cos(dphi)
        # Then rotate vertically (about x)
        y, z = y*cos(dtheta) - z*sin(dtheta), y*sin(dtheta) + z*cos(dtheta)
        # Then update parameters
        self.up = y
        self.eye = self.center + r*z

    def translate(self, ax, ay):
        x,y,z,r = self._getbasis()
        self.center += ax*x + ay*y
        self.eye += ax*x + ay*y

    def zoom(self, dz):
        self.z += dz*sqrt(self.z)
        if self.z < 1e-2:
            self.z = 1e-2

    def draw(self):
        """Draw a graphic to help orient the user if the camera is moving."""
        if self.rotating_graphic:
            c, x, y, z, r = self.rotating_graphic
            glColor3f(1, 0, 0)
            glBegin(GL_LINE_LOOP)
            for t in range(64):
                v = c + y*cos(2*pi*t/64) + z*sin(2*pi*t/64)
                glVertex3f(v.x, v.y, v.z)
            glEnd()
            glColor3f(0, 1, 0)
            glBegin(GL_LINE_LOOP)
            for t in range(64):
                v = c + z*cos(2*pi*t/64) + x*sin(2*pi*t/64)
                glVertex3f(v.x, v.y, v.z)
            glEnd()
            glColor3f(0, 0, 1)
            glBegin(GL_LINE_LOOP)
            for t in range(64):
                v = c + x*cos(2*pi*t/64) + y*sin(2*pi*t/64)
                glVertex3f(v.x, v.y, v.z)
            glEnd()

        if self.translating_graphic:
            c, x, y, z, r = self.translating_graphic
            glColor3f(0.5, 0.5, 0.5)
            glBegin(GL_LINES)
            glVertex3f(c.x - 0.2*x.x, c.y - 0.2*x.y, c.z - 0.2*x.z)
            glVertex3f(c.x + 0.2*x.x, c.y + 0.2*x.y, c.z + 0.2*x.z)
            glVertex3f(c.x - 0.2*y.x, c.y - 0.2*y.y, c.z - 0.2*y.z)
            glVertex3f(c.x + 0.2*y.x, c.y + 0.2*y.y, c.z + 0.2*y.z)
            glVertex3f(c.x - 0.2*z.x, c.y - 0.2*z.y, c.z - 0.2*z.z)
            glVertex3f(c.x + 0.2*z.x, c.y + 0.2*z.y, c.z + 0.2*z.z)
            glEnd()

            c = self.center
            x, y, z, r = self._getbasis()
            glColor3f(0, 0, 1)
            glBegin(GL_LINES)
            glVertex3f(c.x - 0.2*x.x, c.y - 0.2*x.y, c.z - 0.2*x.z)
            glVertex3f(c.x + 0.2*x.x, c.y + 0.2*x.y, c.z + 0.2*x.z)
            glVertex3f(c.x - 0.2*y.x, c.y - 0.2*y.y, c.z - 0.2*y.z)
            glVertex3f(c.x + 0.2*y.x, c.y + 0.2*y.y, c.z + 0.2*y.z)
            glVertex3f(c.x - 0.2*z.x, c.y - 0.2*z.y, c.z - 0.2*z.z)
            glVertex3f(c.x + 0.2*z.x, c.y + 0.2*z.y, c.z + 0.2*z.z)
            glEnd()

    def on_mouse_press(self, mx, my, button=mouse.LEFT, modifiers=0):
        x,y,z,r = self._getbasis()
        shift, ctrl = (modifiers & key.MOD_SHIFT, modifiers & key.MOD_CTRL)
        if button == mouse.LEFT:
            self.rotating_graphic = [vec3(self.center), vec3(x), vec3(y), vec3(z), float(r)]
        elif button == mouse.RIGHT:
            self.translating_graphic = [vec3(self.center), vec3(x), vec3(y), vec3(z), float(r)]
        self.mx = mx
        self.my = my

    def on_mouse_release(self, mx, my, button=mouse.LEFT, modifiers=0):
        shift, ctrl = (modifiers & key.MOD_SHIFT, modifiers & key.MOD_CTRL)
        if button == mouse.LEFT:
            self.rotating_graphic = None
        elif button == mouse.RIGHT:
            self.translating_graphic = None
        self.mx = None
        self.my = None

    def on_mouse_drag(self, mx, my, button=mouse.LEFT, modifiers=0):
        shift, ctrl = (modifiers & key.MOD_SHIFT, modifiers & key.MOD_CTRL)
        dx = mx - self.mx
        dy = my - self.my
        if button == mouse.LEFT:
            dphi   = dx * (2*pi/1000.)
            dtheta = dy * (2*pi/1000.)
            self.rotate(dphi, -dtheta)
        elif button == mouse.RIGHT:
            ax = dx/200.
            ay = dy/200.
            self.translate(-ax, -ay)
        self.mx = mx
        self.my = my

    def on_mouse_scroll(self, mx, my, dx, dy):
        self.zoom(-0.1*dy)

    def _getbasis(self):
        r = length(self.eye - self.center)
        z = vec3(self.eye - self.center)/r
        y = self.up
        x = cross(y,z)
        return (x,y,z,r)

    def _fixup(self):
        """Adjust the up vector so that it's orthogonal to the view direction."""
        n = unit(self.center - self.eye)
        v = cross(n, self.up)
        if v == vec3(0,0,0):
            print >> sys.stderr, "?? up vector (%g,%g,%g) parallel to view vector (%g,%g,%g)" % (self.up.x,self.up.y,self.up.z,n.x,n.y,n.z)
            v = find_basis(n)[0]
        self.up = unit(cross(v, n))
