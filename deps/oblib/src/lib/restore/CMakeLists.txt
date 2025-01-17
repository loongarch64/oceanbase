add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
oblib_add_library(restore OBJECT
  ob_i_storage.h
  ob_i_storage.cpp
  ob_storage.cpp
  ob_storage_file.cpp
  ob_storage_file.h
  ob_storage.h
  ob_storage_oss_base.cpp
  ob_storage_oss_base.h
  ob_storage_cos_base.cpp
  ob_storage_cos_base.h
  ob_storage_path.cpp
  ob_storage_path.h
  ob_storage_info.cpp
  ob_storage_info.h
  cos/ob_cos_wrapper_handle.cpp
  cos/ob_cos_wrapper_handle.h
  cos/ob_singleton.h
  ob_object_device.cpp
  hmac_signature.cpp
  ob_object_device.h
  ob_object_storage_base.cpp)

add_library(oss INTERFACE)
target_include_directories(oss
  INTERFACE
  ${DEP_DIR}/include/apr-1
  ${DEP_DIR}/include
  ${DEP_DIR}/include/oss_c_sdk
  ${DEP_DIR}/include/mxml
)

target_link_libraries(oss
  INTERFACE
  /usr/lib/liboss_c_sdk.so
  /usr/lib/libaprutil-1.so
  /usr/lib/libapr-1.so
  /usr/lib/libmxml.so)

oblib_add_library(s3 ob_storage_s3_base.cpp)
target_link_directories(s3 PUBLIC /usr/lib/)
target_link_libraries(s3
  PUBLIC
  libaws-cpp-sdk-s3.so
  libaws-cpp-sdk-core.so
  libaws-crt-cpp.so
  libaws-c-mqtt.so
  libaws-c-event-stream.so
  libaws-c-s3.so
  libaws-c-auth.so
  libaws-c-http.so
  libaws-c-io.so
  libs2n.so
  libaws-c-compression.so
  libaws-c-cal.so
  libaws-c-sdkutils.so
  libaws-checksums.so
  libaws-c-common.so
)
target_link_libraries(s3 PUBLIC oblib_base_base_base)

target_link_libraries(restore PUBLIC oss cos_sdk oblib_base s3)

add_subdirectory(cos)
