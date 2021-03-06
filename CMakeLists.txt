# CMAKE version of Ubuntu 16.04 is 3.5.
cmake_minimum_required(VERSION 3.5)
project(program_name)

set(LINKLIBS)
set(COMPILE_OPTION)
set(SRC)
set(INCL)

add_subdirectory(include/dlib dlib_build)

list(APPEND SRC 
  #dnn.cpp
  #landmark.cpp
  dlib.cpp
  )

if(UNIX )  
list(APPEND LINKLIBS 
  #${PROJECT_SOURCE_DIR}/lib/libopencv_core.so.4.2.0
  #${PROJECT_SOURCE_DIR}/lib/libopencv_highgui.so.4.2.0
  #${PROJECT_SOURCE_DIR}/lib/libopencv_imgproc.so.4.2.0
  #${PROJECT_SOURCE_DIR}/lib/libopencv_objdetect.so.4.2.0
  #${PROJECT_SOURCE_DIR}/lib/libopencv_videoio.so.4.2.0
  #${PROJECT_SOURCE_DIR}/lib/libopencv_dnn.so.4.2.0
  #${PROJECT_SOURCE_DIR}/lib/libopencv_face.so.4.2.0

  /usr/local/lib/libopencv_core.so.4.3.0
  /usr/local/lib/libopencv_highgui.so.4.3.0
  /usr/local/lib/libopencv_imgproc.so.4.3.0
  /usr/local/lib/libopencv_objdetect.so.4.3.0
  /usr/local/lib/libopencv_videoio.so.4.3.0
  /usr/local/lib/libopencv_dnn.so.4.3.0
  /usr/local/lib/libopencv_face.so.4.3.0



  #opencv_core
  #opencv_highgui
  #opencv_imgproc
  #opencv_objdetect
  #opencv_videoio
  #opencv_dnn
  #opencv_face

  #  ${PROJECT_SOURCE_DIR}/lib/libdlib.so.19.19.0
  )
elseif(WIN32)
list(APPEND LINKLIBS 
  ${PROJECT_SOURCE_DIR}/lib/opencv_world420.lib
  )
endif()
list(APPEND COMPILE_OPTION )
if(UNIX)
list(APPEND INCL 
  /usr/local/include/opencv4
  ${PROJECT_SOURCE_DIR}/include
  )
elseif(WIN32)
list(APPEND INCL 
  ${PROJECT_SOURCE_DIR}/include
  )
endif()

add_executable(${PROJECT_NAME} ${SRC})

target_include_directories(${PROJECT_NAME}
	PRIVATE
	${INCL}
)

LINK_DIRECTORIES( /usr/local/lib/ )
target_link_libraries(${PROJECT_NAME} ${LINKLIBS} dlib::dlib)
target_compile_options(${PROJECT_NAME} PRIVATE ${COMPILE_OPTION} )


