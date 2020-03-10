/* Generate a list of possible cuboid remappings, with the dimensions of the
 * the remapped cuboid and the 3x3 integer matrix that produces it. */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <vector>
using namespace std;

#include "vec3.h"


struct Remapping {
    vec3i u1, u2, u3;
    vec3d e1, e2, e3;
    float L1, L2, L3;

    Remapping() {
        u1 = vec3i(1,0,0);
        u2 = vec3i(0,1,0);
        u3 = vec3i(0,0,1);
        e1 = vec3d(u1);
        e2 = vec3d(u2);
        e3 = vec3d(u3);
        L1 = L2 = L3 = 1;
    }

    Remapping(const vec3i& u1_, const vec3i& u2_, const vec3i& u3_) {
        u1 = u1_;
        u2 = u2_;
        u3 = u3_;
        assert(tsp(u1, u2, u3) == 1);

        double s1 = sqr(u1);
        double s2 = sqr(u2);
        double d12 = dot(u1, u2);
        double d23 = dot(u2, u3);
        double d13 = dot(u1, u3);
        double alpha = -d12/s1;
        double gamma = -(alpha*d13 + d23)/(alpha*d12 + s2);
        double beta = -(d13 + gamma*d12)/s1;
        e1 = vec3d(u1);
        e2 = vec3d(u2) + alpha*vec3d(u1);
        e3 = vec3d(u3) + beta*vec3d(u1) + gamma*vec3d(u2);
        L1 = len(e1);
        L2 = len(e2);
        L3 = len(e3);
    }

    /* A somewhat arbitrary measure of how "complex" the remapping is (for
     * choosing betwenen equivalent remappings) */
    int score() const {
        int c = 0;
        /* Prefer small integer coefficients */
        c += abs(u1.x) + abs(u1.y) + abs(u1.z);
        c += abs(u2.x) + abs(u2.y) + abs(u2.z);
        c += abs(u3.x) + abs(u3.y) + abs(u3.z);
        /* Prefer positive integer coefficients */
        c += (u1.x < 0) + (u1.y < 0) + (u1.z < 0);
        c += (u2.x < 0) + (u2.y < 0) + (u2.z < 0);
        c += (u3.x < 0) + (u3.y < 0) + (u3.z < 0);
        /* Prefer lengths in descending order */
        c -= 10*(L1 > L2 && L2 > L3);
        return c;
    }

    vec3d get_ordered_lengths() const {
        double Lmax = fmax(L1, fmax(L2, L3));
        double Lmin = fmin(L1, fmin(L2, L3));
        double Lmid = L1 + L2 + L3 - Lmax - Lmin;
        return vec3d(Lmax, Lmid, Lmin);
    }
};

/* Test whether the components of a vector are all integers */
bool is_int_vector(const vec3d& e) {
    return fabs(e.x - round(e.x)) < 1e-9
        && fabs(e.y - round(e.y)) < 1e-9
        && fabs(e.z - round(e.z)) < 1e-9;
}

/* Order vectors first by x component, then y, then z (with a small roundoff tolerance) */
struct vec3d_ordering {
    bool operator()(const vec3d& u, const vec3d& v) const {
        if(fabs(u.x - v.x) > 1e-9)
            return u.x < v.x;
        else if(fabs(u.y - v.y) > 1e-9)
            return u.y < v.y;
        else if(fabs(u.z - v.z) > 1e-9)
            return u.z < v.z;
        else
            return false;
    }
};

/* Return the greatest common divisor of a and b */
int gcd(int a, int b) {
    int tmp;
    if(a < 0)
        a = -a;
    if(b < 0)
        b = -b;
    while(b != 0) {
        tmp = b;
        b = a % b;
        a = tmp;
    }
    return a;
}

/* Return the greatest common divisor of a, b, and c */
int gcd(int a, int b, int c) {
    return gcd(a, gcd(b, c));
}

int main(int argc, char* argv[]) {
    int N = 3;
    if(argc == 2)
        N = atoi(argv[1]);
    else if(argc > 2) {
        fprintf(stderr, "Usage: %s Nmax\n", argv[0]);
        return 1;
    }

    if(N < 1) {
        fprintf(stderr, "Error: Nmax must be >= 1\n");
        return 2;
    }
    else if(N > 20) {
        fprintf(stderr, "Error: Nmax of %d is too large\n", N);
        fprintf(stderr, "       (runtime and output size scale faster than Nmax^6)\n");
        return 3;
    }

    printf("# Nmax = %d\n", N);

    /* Generate mutually coprime triplets of integers in the range [-N,+N] */
    vector<vec3i> coprime_triples;
    for(int a = -N; a <= +N; a++)
        for(int b = -N; b <= +N; b++)
            for(int c = -N; c <= +N; c++)
                if(gcd(a,b,c) == 1)
                    coprime_triples.push_back(vec3i(a,b,c));
    int numvecs = coprime_triples.size();
//    printf("# numvecs = %d\n", numvecs);

    typedef map<vec3d, Remapping, vec3d_ordering> RemappingList;

    /* Use coprime vectors to build invertible matrices, and keep a list of
     * the unique remappings they generate, sorted by (Lmax,Lmid,Lmin) */
    int nummats = 0;
    RemappingList remappings;
    vec3i u1, u2, u3;
    for(int i1 = 0; i1 < numvecs; i1++) {
        u1 = coprime_triples[i1];
        for(int i2 = 0; i2 < numvecs; i2++) {
            u2 = coprime_triples[i2];
            for(int i3 = 0; i3 < numvecs; i3++) {
                u3 = coprime_triples[i3];

                if(tsp(u1,u2,u3) == 1) {
                    nummats++;
                    Remapping R(u1, u2, u3);
                    vec3d key = R.get_ordered_lengths();
                    if(remappings.count(key) == 0 || R.score() <= remappings[key].score())
                        remappings[key] = R;
                }
            }
        }
    }

#if 0
    /* Re-sort the list of remappings by (L1,L2,L3) instead of (Lmax,Lmid,Lmin) */
    RemappingList sorted_remappings;
    for(RemappingList::const_iterator iter = remappings.begin(); iter != remappings.end(); iter++) {
        const Remapping& R = iter->second;
        vec3d key(R.L1, R.L2, R.L3);
        sorted_remappings[key] = R;
    }
#endif

//    int numremaps = remappings.size();
//    printf("# nummats = %d\n", nummats);
//    printf("# numremaps = %d\n", numremaps);
//    printf("#\n");

    printf("# L1 L2 L3   u11 u12 u13   u21 u22 u23   u31 u32 u33   (periodicity)\n");
    for(RemappingList::const_iterator iter = remappings.begin(); iter != remappings.end(); iter++) {
        const Remapping& R = iter->second;
        printf("%1.4f %1.4f %1.4f   %d %d %d   %d %d %d   %d %d %d   (", R.L1, R.L2, R.L3, R.u1.x, R.u1.y, R.u1.z, R.u2.x, R.u2.y, R.u2.z, R.u3.x, R.u3.y, R.u3.z);
        if(is_int_vector(R.e1))
            printf("1");
        if(is_int_vector(R.e2))
            printf("2");
        if(is_int_vector(R.e3))
            printf("3");
        printf(")\n");
//        printf("  %1.2f %1.2f %1.2f  %1.2f %1.2f %1.2f  %1.2f %1.2f %1.2f\n", R.e1.x, R.e1.y, R.e1.z, R.e2.x, R.e2.y, R.e2.z, R.e3.x, R.e3.y, R.e3.z);
    }

    return 0;
}
