#!/usr/bin/python
#
# demo.py

import sys
import copy
import random
from math import *
import pyglet
from pyglet.gl import *
from camera import *
from cuboid import *
from poly import *

colors = [
          (1,0,0),
          (0,1,0),
          (0,0,1),
          (1,1,0),
          (1,0,1),
          (0,1,1),
          (1,0.5,0),
          (1,0,0.5),
          (0.5,0,1),
          (0.5,1,0),
          (0,1,0.5),
          (0,0.5,1),
          (0.5,0.5,0),
          (0.5,0,0.5),
          (0,0.5,0.5),
          (0.5,0.5,0.5)
]*8

def draw_polygon(g, color=(1,0,0), outline=True):
    # Fill in polygon at 50% opacity
    glColor4f(color[0], color[1], color[2], 0.7)
#    glDepthMask(GL_FALSE)
    glBegin(GL_POLYGON)
    for v in g.verts:
        glVertex3f(v.x, v.y, v.z)
    glEnd()
#    glDepthMask(GL_TRUE)

    # Outline polygon in bold
    if outline:
        glColor4f(color[0], color[1], color[2], 1)
        glBegin(GL_LINE_LOOP)
        for v in g.verts:
            glVertex3d(v.x, v.y, v.z)
        glEnd()

def draw_polyhedron(h, color=(1,0,0), outline=True):
    for f in h.faces:
        draw_polygon(f, color, outline)

def make_polyhedron_from_cell(c):
    h = unitcube()
    h.translate(vec3(0.5e-6, 2.3e-6, 1.7e-6))
    for P in c.faces:
        h.cut(P)
    return h



# Initialize window
width = 800
height = 600
window = pyglet.window.Window(width, height, resizable=True)
glEnable(GL_DEPTH_TEST)
#glDepthFunc(GL_LEQUAL)
glClearColor(1, 1, 1, 0)
glEnable(GL_CULL_FACE)
glFrontFace(GL_CCW)
glCullFace(GL_BACK)

# Initialize camera
camera = DemoCamera(width, height, center=(0.5,0.5,0.5), eye=(0,3,0.), up=(1,0,0))

# Initialize cuboid
C = Cuboid(u1=(3,2,1), u2=(-1,1,2), u3=(1,1,1))
polyhedra = [make_polyhedron_from_cell(cell) for cell in C.cells]
print len(polyhedra)

anim_on = False
anim_t = 0.0

def update_animation(dt):
    global anim_t
    anim_t += 0.5*dt


@window.event
def on_draw():
    window.clear()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    camera.apply()
    camera.draw()

    def carray(data):
        return (GLfloat * len(data))(*data)

    # Draw black frame around unit cube
    cube = unitcube()
    n = unit(camera.eye - camera.center)
    cube.translate(0.001*n)
    glPushAttrib(GL_ENABLE_BIT | GL_POLYGON_BIT)
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
    glDisable(GL_CULL_FACE)
    draw_polyhedron(cube, color=(0,0,0), outline=False)
    glPopAttrib()

    # Draw polyhedra
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    i = 0
    for h in polyhedra:
        c = C.cells[i]
        h = copy.deepcopy(h)
        s = 0.5 * (1 - cos(anim_t))
        h.translate(vec3(s*c.ix, s*c.iy, s*c.iz))
        draw_polyhedron(h, colors[i])
        i += 1
    glDisable(GL_BLEND)


@window.event
def on_resize(w, h):
    global width, height
    width = w
    height = h
    glViewport(0, 0, width, height)
    return pyglet.event.EVENT_HANDLED

@window.event
def on_key_press(symbol, modifiers):
    global C, polyhedra, camera, anim_on, anim_t
    if symbol == pyglet.window.key.S:
        # Save image to file
        image = pyglet.image.get_buffer_manager().get_color_buffer()
        image.save("figure.png")
    elif symbol == pyglet.window.key.R:
        camera = DemoCamera(eye=(0,0,5))
    elif symbol == pyglet.window.key.J:
        anim_on = False
        pyglet.clock.unschedule(update_animation)
        anim_t = 0
    elif symbol == pyglet.window.key.K:
        anim_on = False
        pyglet.clock.unschedule(update_animation)
        anim_t = pi
    elif symbol == pyglet.window.key.SPACE:
        if anim_on:
            pyglet.clock.unschedule(update_animation)
            anim_on = False
        else:
            pyglet.clock.schedule_interval(update_animation, 1/30.)
            anim_on = True
    elif symbol == pyglet.window.key._1:
        C = Cuboid(u1=(1,0,0), u2=(0,1,0), u3=(0,0,1))
        polyhedra = [make_polyhedron_from_cell(cell) for cell in C.cells]
        print len(polyhedra)
    elif symbol == pyglet.window.key._2:
        C = Cuboid(u1=(1,1,0), u2=(0,0,1), u3=(1,0,0))
        polyhedra = [make_polyhedron_from_cell(cell) for cell in C.cells]
        print len(polyhedra)
    elif symbol == pyglet.window.key._3:
        C = Cuboid(u1=(1,1,0), u2=(1,0,1), u3=(1,0,0))
        polyhedra = [make_polyhedron_from_cell(cell) for cell in C.cells]
        print len(polyhedra)
    elif symbol == pyglet.window.key._4:
        C = Cuboid(u1=(1,1,1), u2=(1,0,0), u3=(0,1,0))
        polyhedra = [make_polyhedron_from_cell(cell) for cell in C.cells]
        print len(polyhedra)
    elif symbol == pyglet.window.key._5:
        C = Cuboid(u1=(1,1,1), u2=(1,-1,0), u3=(1,0,0))
        polyhedra = [make_polyhedron_from_cell(cell) for cell in C.cells]
        print len(polyhedra)
    elif symbol == pyglet.window.key._6:
        C = Cuboid(u1=(3,2,1), u2=(-1,1,2), u3=(1,1,1))
        polyhedra = [make_polyhedron_from_cell(cell) for cell in C.cells]
        print len(polyhedra)
    elif symbol == pyglet.window.key._7:
        C = Cuboid(u1=(2,1,0), u2=(1,0,1), u3=(1,0,0))
        polyhedra = [make_polyhedron_from_cell(cell) for cell in C.cells]
        print len(polyhedra)
    elif symbol == pyglet.window.key._8:
        C = Cuboid(u1=(7,6,4), u2=(3,3,2), u3=(0,1,1))
        polyhedra = [make_polyhedron_from_cell(cell) for cell in C.cells]
        print len(polyhedra)
    elif symbol == pyglet.window.key._9:
        C = Cuboid(u1=(3,2,1), u2=(-1,1,2), u3=(1,1,1))
        polyhedra = [make_polyhedron_from_cell(cell) for cell in C.cells]
        print len(polyhedra)


@window.event
def on_mouse_press(x, y, button, modifiers):
    camera.on_mouse_press(x, y, button, modifiers)

@window.event
def on_mouse_release(x, y, button, modifiers):
    camera.on_mouse_release(x, y, button, modifiers)

@window.event
def on_mouse_drag(x, y, dx, dy, button, modifiers):
    camera.on_mouse_drag(x, y, button, modifiers)

@window.event
def on_mouse_scroll(x, y, dx, dy):
    camera.on_mouse_scroll(x, y, dx, dy)

pyglet.app.run()
