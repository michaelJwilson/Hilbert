# poly.py
#
# Geometric constructs for defining a polyhedron as the intersection of a
# a collection of planes.
#
# Written by Jordan Carlson (jwgcarlson@berkeley.edu)

from vec3 import *

def rotleft(a, k):
    """Rotate the elements of the array a by k elements to the left."""
    n = len(a)
    assert 0 <= k < n
    first = a[0:k]
    last = a[k:]
    a[0:n-k] = last
    a[n-k:n] = first


class Plane:
    epsilon = 0      # threshold for point-plane comparisons

    def __init__(self, p=vec3(0,0,0), n=vec3(0,0,1)):
        self.a = n.x
        self.b = n.y
        self.c = n.z
        self.d = -dot(p,n)

    def test(self, x, y, z):
        """Test whether the point (x,y,z) lies inside or outside this plane."""
        return self.a*x + self.b*y + self.c*z + self.d

    def contains(self, p):
        return fabs(self.test(p.x, p.y, p.z)) <= epsilon

    def inside(self, p):
        """Check whether the point p lies (strictly) inside the plane."""
        return self.test(p.x, p.y, p.z) > epsilon

    def outside(self, p):
        """Check whether the point p lies (strictly) outside the plane."""
        return self.test(p.x, p.y, p.z) < -epsilon

    def normal(self):
        L = sqrt(self.a**2 + self.b**2 + self.c**2)
        return vec3(self.a/L, self.b/L, self.c/L)

    def line_intersection(self, v0, v1):
        """Return the point at which the plane intersects the line segment [v0,v1]."""
        n = vec3(self.a, self.b, self.c)
        t = -(dot(n,v0) + self.d)/dot(n,v1-v0)
        return v0 + (v1 - v0)*t


class Polygon:
    def __init__(self, vertices=[]):
        self.verts = vertices

    def center(self):
        if len(self.verts) == 0:
            return vec3(0,0,0)
        else:
            return sum(self.verts)/len(self.verts)

    def translate(self, dx):
        for v in self.verts:
            v.x += dx.x
            v.y += dx.y
            v.z += dx.z

    def __repr__(self):
        return repr(self.verts)


class Polyhedron:
    def __init__(self, faces=[]):
        self.faces = faces

    def center(self):
        if len(self.faces) == 0:
            return vec3(0,0,0)
        else:
            return sum([f.center() for f in self.faces])/len(self.faces)

    def translate(self, dx):
        for f in self.faces:
            f.translate(dx)

    def cut(self, P):
        """Cut the current polyhedron by the plane P."""
        newedges = []
        j = 0
        while j < len(self.faces):
            f = self.faces[j]
            n = len(f.verts)
            # Find one vertex inside the plane, and one vertex outside the plane
            kin = -1
            kout = -1
            kon = -1
            k = 0
            while k < len(f.verts):
                if P.inside(f.verts[k]):
                    kin = k
                elif P.outside(f.verts[k]):
                    kout = k
                else:
                    kon = k
                k += 1
            if kon != -1 and (kin == -1 or kout == -1):
                # The plane just grazes the face; see if it grazes a point or a whole edge
                if P.contains(f.verts[kon-1]):
                    newedges.append((f.verts[kon-1], f.verts[kon]))
                elif P.contains(f.verts[(kon+1) % n]):
                    newedges.append((f.verts[kon], f.verts[(kon+1) % n]))
            if kin == -1:
                # No vertices of the face lie inside the plane; remove the whole face
                del self.faces[j]
                continue
            elif kout == -1:
                # All vertices are inside the plane; keep face as is
                j += 1
                continue
            else:
                # Move kin forward until vertex (kin+1) lies outside
                while P.inside(f.verts[(kin+1) % n]):
                    kin += 1
                # Move kout forward until vertex (kout+1) lies inside
                while P.outside(f.verts[(kout+1) % n]):
                    kout += 1

                # Get the two new vertices
                vA = P.line_intersection(f.verts[kin % n], f.verts[(kin+1) % n])
                vB = P.line_intersection(f.verts[kout % n], f.verts[(kout+1) % n])
                newedges.append((vA,vB))

                # Define the vertices of the new face
                newf = [vA, vB]
                k = kout + 1    # start with first vertex inside plane
                while (k % n) != ((kin+1) % n):
                    newf.append(vec3(f.verts[k % n]))
                    k += 1

                self.faces[j] = Polygon(newf)
                j += 1

        n = len(newedges)
        if n >= 3:
            # Put the edges in order, so that newedges[k][1] == newedges[k+1][0]
            k = 0
            while k < n:
                k2 = k + 1
                while k2 < n:
                    if newedges[k][1] == newedges[k2][0]:
                        newedges[k+1], newedges[k2] = newedges[k2], newedges[k+1]
                        break
                    k2 += 1
                k += 1

            newedges.reverse()
            # Now combine all the new edges into a new face of the polyhedron
            newf = [vec3(edge[0]) for edge in newedges]
            self.faces.append(Polygon(newf))


def unitcube():
    back    = Polygon([vec3(0,0,0), vec3(0,0,1), vec3(0,1,1), vec3(0,1,0)])
    left    = Polygon([vec3(0,0,0), vec3(1,0,0), vec3(1,0,1), vec3(0,0,1)])
    bottom  = Polygon([vec3(0,0,0), vec3(0,1,0), vec3(1,1,0), vec3(1,0,0)])
    front   = Polygon([vec3(1,0,0), vec3(1,1,0), vec3(1,1,1), vec3(1,0,1)])
    right   = Polygon([vec3(0,1,0), vec3(0,1,1), vec3(1,1,1), vec3(1,1,0)])
    top     = Polygon([vec3(0,0,1), vec3(1,0,1), vec3(1,1,1), vec3(0,1,1)])
    return Polyhedron([back, left, bottom, front, right, top])
