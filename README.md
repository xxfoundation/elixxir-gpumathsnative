# elixxir/gpumathsnative

## Building the native gpumaths library

Before beginning, install the CUDA toolkit, version 10.2 and libgmp-dev.

```
$ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/
cuda-ubuntu1804.pin
$ sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
$ wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/
cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
$ sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
$ sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
$ sudo apt-get update
$ sudo apt-get -y install cuda
```

```
$ sudo apt install libgmp-dev
```

Next, build and install the native gpumaths library. You must have nvcc in your PATH for this to work.

```
$ cd cgbnBindings/powm
$ make turing
$ sudo make install
```

Then, you should be able to build the server with GPU support.

