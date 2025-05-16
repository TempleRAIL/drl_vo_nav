function(venv_ensure_pip)
    add_custom_command(
        TARGET "${target_venv}"
        POST_BUILD
        COMMAND "${venv_python}" -m ensurepip --default-pip
    )
endfunction()

function(venv_pip_install package)
    add_custom_command(
        TARGET "${target_venv}"
        POST_BUILD
        COMMAND ${venv_python} -m pip install ${package}
    )
endfunction()

function(venv_pip_install_local package)
    cmake_path(GET package FILENAME package_name)
    set(build_dir "${CMAKE_CURRENT_BINARY_DIR}/python_builds/${package_name}")
    add_custom_command(
        TARGET "${target_venv}"
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${package}" "${build_dir}"
        COMMAND "${venv_python}" -m pip install "${build_dir}"
    )
endfunction()