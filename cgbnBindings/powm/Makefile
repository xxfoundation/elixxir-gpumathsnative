ifdef GMP_HOME
  INC := -I$(GMP_HOME)/include
  LIB := -L$(GMP_HOME)/lib
endif
ifndef GMP_HOME
  INC :=
  LIB :=
endif

pick:
	@echo
	@echo Please run one of the following:
	@echo "   make kepler"
	@echo "   make maxwell"
	@echo "   make pascal"
	@echo "   make volta"
	@echo "   make turing"
	@echo

clean:
	rm -f libpowmo*.so

devinstall:
	mkdir -p ../../lib
	mv libpowmo*.so ../../lib
	cp powm_odd_export.h ../../lib

install:
	mkdir -p /opt/xxnetwork/lib
	mkdir -p /opt/xxnetwork/include
	mv libpowmo*.so /opt/xxnetwork/lib
	cp powm_odd_export.h /opt/xxnetwork/include

kepler:
	nvcc --compiler-options '-fPIC' --shared $(INC) $(LIB) -I../../cgbn-master/include -arch=sm_30 powm_odd.cu -o libpowmosm40.so -lgmp

maxwell:
	nvcc --compiler-options '-fPIC' --shared $(INC) $(LIB) -I../../cgbn-master/include -arch=sm_50 powm_odd.cu -o libpowmosm50.so -lgmp

pascal:
	nvcc --compiler-options '-fPIC' --shared $(INC) $(LIB) -I../../cgbn-master/include -arch=sm_60 powm_odd.cu -o libpowmosm60.so -lgmp

volta: 
	nvcc --compiler-options '-fPIC' --shared $(INC) $(LIB) -I../../cgbn-master/include -arch=sm_70 powm_odd.cu -o libpowmosm70.so -lgmp

turing: 
	nvcc --compiler-options '-fPIC' --shared $(INC) $(LIB) -I../../cgbn-master/include -arch=sm_75 powm_odd.cu -o libpowmosm75.so -lgmp

turingdebug:
	nvcc --compiler-options '-fPIC' --shared $(INC) $(LIB) -I../../cgbn-master/include -arch=sm_75 powm_odd.cu -o libpowmosm75.so -lgmp -g -DTRACE

