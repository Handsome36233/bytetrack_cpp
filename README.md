# bytetrack_cpp

对 https://github.com/ifzhang/ByteTrack/tree/main 项目的c++复现，检测器换成了yolo

8

编译

```shell
mkdir build && cd build
cmake .. && make
```

运行

```shell
./demo ../checkpoints/yolov8n.onnx ./test_video.mp4
```
