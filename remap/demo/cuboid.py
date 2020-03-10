#!/usr/bin/python
#

import sys
from math import *
from vec3 import *
from poly import Plane


class Cell:
    def __init__(self, ix=0, iy=0, iz=0):
        self.ix = ix
        self.iy = iy
        self.iz = iz
        self.faces = []

    def contains(self, x, y, z):
        for f in self.faces:
            if f.test(x,y,z) < 0:
                return False
        return True

    
def UnitCubeTest(P):
    """Return +1, 0, or -1 if the unit cube is above, below, or intersecting the plane."""
    above = 0
    below = 0
    for (a,b,c) in [(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)]:
        s = P.test(a, b, c)
        if s > 0:
            above = 1
        elif s < 0:
            below = 1
    return above - below


class Cuboid:
    """Cuboid remapping class."""

    def __init__(self, u1=(1,0,0), u2=(0,1,0), u3=(0,0,1)):
        u1 = vec3(u1)
        u2 = vec3(u2)
        u3 = vec3(u3)

        if triple_scalar_product(u1, u2, u3) == 1:
            s1 = square(u1)
            s2 = square(u2)
            d12 = dot(u1, u2)
            d23 = dot(u2, u3)
            d13 = dot(u1, u3)
            alpha = -d12/s1
            gamma = -(alpha*d13 + d23)/(alpha*d12 + s2)
            beta = -(d13 + gamma*d12)/s1
            self.e1 = u1
            self.e2 = u2 + alpha*u1
            self.e3 = u3 + beta*u1 + gamma*u2
        else:
            print >> sys.stderr, "!! Invalid lattice vectors: u1 = %s, u2 = %s, u3 = %s" % (u1,u2,u3)
            self.e1 = vec3(1,0,0)
            self.e2 = vec3(0,1,0)
            self.e3 = vec3(0,0,1)

        self.L1 = length(self.e1)
        self.L2 = length(self.e2)
        self.L3 = length(self.e3)
        self.n1 = self.e1/self.L1
        self.n2 = self.e2/self.L2
        self.n3 = self.e3/self.L3

        self.cells = []
#        print "e1 = %s" % self.e1
#        print "e2 = %s" % self.e2
#        print "e3 = %s" % self.e3
        print "L1 = %s" % self.L1
        print "L2 = %s" % self.L2
        print "L3 = %s" % self.L3

        v0 = vec3(0,0,0)
        self.v = [v0,
                  v0 + self.e3,
                  v0 + self.e2,
                  v0 + self.e2 + self.e3,
                  v0 + self.e1,
                  v0 + self.e1 + self.e3,
                  v0 + self.e1 + self.e2,
                  v0 + self.e1 + self.e2 + self.e3]

        # Compute bounding box of cuboid
        xs = [vk.x for vk in self.v]
        ys = [vk.y for vk in self.v]
        zs = [vk.z for vk in self.v]
        vmin = vec3(min(xs), min(ys), min(zs))
        vmax = vec3(max(xs), max(ys), max(zs))

        # Extend to nearest integer coordinates
        ixmin = int(floor(vmin.x))
        ixmax = int(ceil(vmax.x))
        iymin = int(floor(vmin.y))
        iymax = int(ceil(vmax.y))
        izmin = int(floor(vmin.z))
        izmax = int(ceil(vmax.z))
#        print "ixmin, ixmax = %d, %d" % (ixmin,ixmax)
#        print "iymin, iymax = %d, %d" % (iymin,iymax)
#        print "izmin, izmax = %d, %d" % (izmin,izmax)

        # Determine which cells (and which faces within those cells) are non-trivial
        for ix in range(ixmin, ixmax):
            for iy in range(iymin, iymax):
                for iz in range(izmin, izmax):
                    shift = vec3(-ix, -iy, -iz)
                    faces = [Plane(self.v[0] + shift, +self.n1),
                             Plane(self.v[4] + shift, -self.n1),
                             Plane(self.v[0] + shift, +self.n2),
                             Plane(self.v[2] + shift, -self.n2),
                             Plane(self.v[0] + shift, +self.n3),
                             Plane(self.v[1] + shift, -self.n3)]

                    c = Cell(ix, iy, iz)
                    skipcell = False
                    for f in faces:
                        r = UnitCubeTest(f)
                        if r == +1:
                            continue
                        elif r == 0:
                            c.faces.append(f)
                        elif r == -1:
                            skipcell = True
                            break

                    if skipcell or len(c.faces) == 0:
#                        print "Skipping cell at (%d,%d,%d)" % (ix,iy,iz)
                        continue
                    else:
                        self.cells.append(c)
#                        print "Adding cell at (%d,%d,%d)" % (ix,iy,iz)

        if len(self.cells) == 0:
            self.cells.append(Cell())

        # Print the full list of cells
#        print "%d cells" % len(self.cells)
#        for c in self.cells:
#            print "Cell at (%d,%d,%d) has %d non-trivial planes" % (c.ix, c.iy, c.iz, len(c.faces))

    def Transform(self, x, y, z):
        for c in self.cells:
            if c.contains(x,y,z):
                x += c.ix
                y += c.iy
                z += c.iz
                p = vec3(x,y,z)
                return (dot(p, self.n1), dot(p, self.n2), dot(p, self.n3))
        raise RuntimeError, "(%g,%g,%g) not contained in any cell" % (x,y,z)

    def InverseTransform(self, r1, r2, r3):
        p = r1*self.n1 + r2*self.n2 + r3*self.n3
        x1 = fmod(p[0], 1) + (p[0] < 0)
        x2 = fmod(p[1], 1) + (p[1] < 0)
        x3 = fmod(p[2], 1) + (p[2] < 0)
        return vec3(x1, x2, x3)
