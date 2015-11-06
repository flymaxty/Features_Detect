CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(OpenCV REQUIRED)

ADD_EXECUTABLE(sample sample.cpp features_detect.cpp)
TARGET_LINK_LIBRARIES(sample ${OpenCV_LIBS})

ADD_EXECUTABLE(sample_cam sample_cam.cpp features_detect.cpp )
TARGET_LINK_LIBRARIES(sample_cam ${OpenCV_LIBS} ${FREENECT_LIBRARIES} pthread)

ADD_EXECUTABLE(object_extract object_capture.cpp)
TARGET_LINK_LIBRARIES(object_extract ${OpenCV_LIBS})
