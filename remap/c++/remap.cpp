/* remap
 *
 * A simple program to apply a cuboid remapping to a series of points.
 *
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "cuboid.h"

void usage() {
    printf("Usage: remap in=\"in file\" out=\"out file\" u=\"u11 u12 u13 u21 u22 u23 u31 u32 u33\"\n");
    printf("If not given, 'in' defaults to stdin, 'out' defaults to stdout, and 'u' defaults\n");
    printf("to the identity matrix.\n");
    exit(0);
}

int main(int argc, char* argv[]) {
    const char* infile = NULL;
    const char* outfile = NULL;
    int u[9] = { 1, 0, 0,   0, 1, 0,   0, 0, 1 };

    /* Parse command line arguments */
    if(argc == 1)
        usage();
    for(int i = 1; i < argc; i++) {
        char* pos = strchr(argv[i], '=');
        if(pos == NULL)
            usage();
        *pos++ = '\0';
        if(strcmp(argv[i], "in") == 0)
            infile = pos;
        else if(strcmp(argv[i], "out") == 0)
            outfile = pos;
        else if(strcmp(argv[i], "u") == 0) {
            char* tok = strtok(pos, " ,");
            for(int k = 0; k < 9; k++) {
                if(tok == NULL)
                    usage();
                u[k] = atoi(tok);
                tok = strtok(NULL, " ,");
            }
            if(tok != NULL)
                usage();
        }
        else
            usage();
    }

    /* Open input and output files */
    FILE* fin = (infile == NULL || strcmp(infile, "stdin") == 0) ? stdin : fopen(infile, "r");
    FILE* fout = (outfile == NULL || strcmp(outfile, "stdout") == 0) ? stdout : fopen(outfile, "w");
    if(!fin) {
        fprintf(stderr, "Error: could not open input file '%s'\n", infile);
        return 2;
    }
    if(!fout) {
        fprintf(stderr, "Error: could not open output file '%s'\n", outfile);
        return 2;
    }

    /* Initialize cuboid remapping */
    Cuboid R(u);

    /* Read points from ASCII text file, one line at a time */
    bool warning = false;
    double x1, x2, x3;
    double r1, r2, r3;
    char line[1024];
    while(fgets(line, sizeof(line), fin) != NULL) {
        if(line[0] == '#' || line[0] == '\0')
            continue;
        sscanf(line, "%lf %lf %lf", &x1, &x2, &x3);
        if(!warning && (x1 < 0 || x1 >= 1 || x2 < 0 || x2 >= 1 || x3 < 0 || x3 >= 1)) {
            fprintf(stderr, "Warning: input points should lie in the unit cube [0,1)^3.\n");
            warning = true;
        }

        /* Remap point into cuboid, and print remapped coordinates */
        R.Transform(x1, x2, x3, r1, r2, r3);
        fprintf(fout, "%g %g %g\n", r1, r2, r3);
    }

    return 0;
}
