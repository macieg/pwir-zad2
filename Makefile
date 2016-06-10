CC      = mpicc
CFLAGS  = -Wall -c --std=c99 -O3
LDFLAGS = -Wall -O3 --std=c99
ALL     = matrixmul
MATGENFILE = densematgen.o
UTILSFILE = utils.o
BLOCKEDCOLUMN = blockedcolumn.o
INNERPRODUCT = innerproduct.o

all: $(ALL)

$(ALL): %: %.o $(MATGENFILE) $(UTILSFILE) $(BLOCKEDCOLUMN) $(INNERPRODUCT)
	$(CC) $(LDFLAGS) $^ -o $@

%.o: %.c matgen.h Makefile
	$(CC) $(CFLAGS) $@ $<



clean:
	rm -f *.o *core *~ *.out *.err $(ALL)
