execute_process(
    COMMAND which uv
    RESULT_VARIABLE UV_FOUND
)
if(
    UV_FOUND AND
    NOT UV_FOUND EQUAL 0
)
    message(FATAL_ERROR "uv installation not found. https://docs.astral.sh/uv/getting-started/installation/")
endif()
unset(UV_FOUND)

include("${ament_cmake_venv_uv_DIR}/uv_venv.cmake")