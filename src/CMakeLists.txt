CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(OpenCV REQUIRED)

ADD_EXECUTABLE(features_detect features_detect.cpp)
TARGET_LINK_LIBRARIES(features_detect ${OpenCV_LIBS})

ADD_EXECUTABLE(features_detect_cam features_detect_cam.cpp)
TARGET_LINK_LIBRARIES(features_detect_cam ${OpenCV_LIBS})

ADD_EXECUTABLE(object_extract object_extract.cpp)
TARGET_LINK_LIBRARIES(object_extract ${OpenCV_LIBS})
