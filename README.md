# hello_triangle
## Installation
```sh
cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_TOOLCHAIN_FILE=<path-to-vcpkg>/scripts/buildsystems/vcpkg.cmake -S . -B out
cmake --build out
cd out
./hello_triangle
```
