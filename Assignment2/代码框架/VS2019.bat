if not exist build (md build)

cd build

cmake .. -G "Visual Studio 16 2019" -A x64 -DCMAKE_PREFIX_PATH=E:\Libs\opencv\build\x64\vc15\lib -DEIGEN_ROOT=E:\Libs\eigen-3.3.9

cd ..