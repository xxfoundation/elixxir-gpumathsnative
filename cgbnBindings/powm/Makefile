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
	mv libpow* /opt/xxnetwork/lib
	cp powm_odd_export.h /opt/xxnetwork/include

turing: 
	nvcc --compiler-options '-fPIC' --shared $(INC) $(LIB) -I../../cgbn-master/include -arch=sm_75 powm_odd.cu -o libpow.fatbin -lgmp -lcuda -fatbin
	nvcc --compiler-options '-fPIC' --shared $(INC) $(LIB) -I../../cgbn-master/include -arch=sm_75 powm_odd.cu -o libpowmosm75.so -lgmp -lcuda

turingdebug:
	nvcc --compiler-options '-fPIC' --shared $(INC) $(LIB) -I../../cgbn-master/include -arch=sm_75 powm_odd.cu -o libpow.fatbin -lgmp -lcuda -fatbin
	nvcc --compiler-options '-fPIC' --shared $(INC) $(LIB) -I../../cgbn-master/include -arch=sm_75 powm_odd.cu -o libpowmosm75.so -lgmp -lcuda -g -G

# Compile a fat binary for Turing and Ampere
fatbin:
	nvcc --compiler-options '-fPIC' --shared $(INC) $(LIB) -I../../cgbn-master/include -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86 powm_odd.cu -o libpow.fatbin -lgmp -lcuda -fatbin
	nvcc --compiler-options '-fPIC' --shared $(INC) $(LIB) -I../../cgbn-master/include -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_86,code=compute_86 powm_odd.cu -o libpowmosm75.so -lgmp -lcuda

