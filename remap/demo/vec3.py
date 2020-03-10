#!/usr/bin/python

import sys
from math import *

epsilon = 1e-9

try:
    # Use fast vec3 implementation if Numpy is available
    import numpy as N
    class vec3(N.ndarray):
        """A simple 3D vector class, using Numpy for fast array operations."""
        def __new__(cls, *args):
            a = N.ndarray.__new__(vec3, (3,), float)
            if len(args) == 0:
                a[0] = a[1] = a[2] = 0
            elif len(args) == 1:
                v = args[0]
                a[0] = v[0]
                a[1] = v[1]
                a[2] = v[2]
            elif len(args) == 3:
                a[0] = args[0]
                a[1] = args[1]
                a[2] = args[2]
            else:
                raise RuntimeError
            return a

        def __eq__(self, other):
            return fabs(self[0]-other[0]) + fabs(self[1]-other[1]) + fabs(self[2]-other[2]) <= epsilon

        def _getx(self): return self[0]
        def _gety(self): return self[1]
        def _getz(self): return self[2]
        def _setx(self, value): self[0] = value
        def _sety(self, value): self[1] = value
        def _setz(self, value): self[2] = value
        x = property(_getx, _setx)
        y = property(_gety, _sety)
        z = property(_getz, _setz)

except:
    # Fall back to non-Numpy implementation
    class vec3:
        """A simple 3D vector class."""
        def __init__(self, x, y=None, z=None, dtype=float):
            if y is None and z is None:
                self.x = dtype(x[0])
                self.y = dtype(x[1])
                self.z = dtype(x[2])
            else:
                self.x = dtype(x)
                self.y = dtype(y)
                self.z = dtype(z)

        def __pos__(self):
            return self

        def __neg__(self):
            return vec3(-self.x, -self.y, -self.z)

        def __add__(self, other):
            return vec3(self.x + other.x, self.y + other.y, self.z + other.z)

        def __sub__(self, other):
            return vec3(self.x - other.x, self.y - other.y, self.z - other.z)

        def __rmul__(self, scalar):
            return vec3(self.x*scalar, self.y*scalar, self.z*scalar)

        def __mul__(self, scalar):
            return vec3(self.x*scalar, self.y*scalar, self.z*scalar)

        def __div__(self, scalar):
            return vec3(self.x/scalar, self.y/scalar, self.z/scalar)

        def __eq__(self, other):
            return fabs(self[0]-other[0]) + fabs(self[1]-other[1]) + fabs(self[2]-other[2]) <= epsilon

        def __repr__(self):
            return "(%g, %g, %g)" % (self.x, self.y, self.z)

        def __getitem__(self, i):
            if   i == 0: return self.x
            elif i == 1: return self.y
            elif i == 2: return self.z


def dot(u, v):
    return u.x*v.x + u.y*v.y + u.z*v.z

def square(v):
    return v.x**2 + v.y**2 + v.z**2

def length(v):
    return sqrt(square(v))

def unit(v):
    return v/length(v)

def cross(v, w):
    return vec3(v.y*w.z - v.z*w.y, v.z*w.x - v.x*w.z, v.x*w.y - v.y*w.x)

def triple_scalar_product(u, v, w):
    return u.x*(v.y*w.z - v.z*w.y) + u.y*(v.z*w.x - v.x*w.z) + u.z*(v.x*w.y - v.y*w.x)

def find_basis(z):
    """Find unit vectors x and y that, together with z, make an orthonormal basis."""
    z = unit(z)
    x = vec3(1,0,0)
    y = cross(z,x)
    if square(y) < 1e-6:
        x = vec3(0,1,0)
        y = cross(z,x)
    y = unit(y)
    x = cross(y,z)
    return (x,y,z)


if __name__ == '__main__':
    u = vec3()
    print u
    u = vec3([0,0,0])
    print u
    u = vec3(7, 3, 2)
    v = vec3(1, 1, 1)
    print 'u = %s, v = %s, u.v = %g, 3*u = %s' % (u,v,dot(u,v),3*u)
