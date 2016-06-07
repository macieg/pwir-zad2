CC      = /home/maciej/.openmpi/bin/mpicc
CFLAGS  = -Wall -c --std=c99 -O3
LDFLAGS = -Wall -O3 --std=c99
ALL     = matrixmul
MATGENFILE = densematgen.o
UTILSFILE = utils.o
INNERCOLUMN = innercolumn.o

all: $(ALL)

$(ALL): %: %.o $(MATGENFILE) $(UTILSFILE) $(INNERCOLUMN)
	$(CC) $(LDFLAGS) $^ -o $@

%.o: %.c matgen.h Makefile
	$(CC) $(CFLAGS) $@ $<



clean:
	rm -f *.o *core *~ *.out *.err $(ALL)
