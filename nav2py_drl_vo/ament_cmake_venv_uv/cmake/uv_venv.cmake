function(uv_venv)
    find_package(ament_cmake_venv REQUIRED)

    set(options)
    set(oneValueArgs DIRECTORY PROJECTFILE)
    set(multiValueArgs)
    cmake_parse_arguments(PARSE_ARGV 0 arg
        "${options}" "${oneValueArgs}" "${multiValueArgs}"
    )

    if(NOT DEFINED arg_DIRECTORY)
        set(arg_DIRECTORY .)
    endif()

    if(NOT DEFINED arg_PROJECTFILE)
        set(arg_PROJECTFILE "pyproject.toml")
    endif()

    set(PROJECT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${arg_DIRECTORY}")
    
    add_custom_command(
        OUTPUT "${venv_build_dir}/${arg_NAME}"
        DEPENDS "${PROJECT_DIRECTORY}/${arg_PROJECTFILE}"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${PROJECT_DIRECTORY}"
            "${venv_build_dir}"
    )

    if(arg_PROJECTFILE STREQUAL "pyproject.toml")
        add_custom_command(
            OUTPUT "${venv_dir}"
            DEPENDS "${venv_build_dir}"
            COMMAND uv venv "${venv_dir}" --project "${venv_build_dir}/pyproject.toml"
            COMMAND "${venv_dir}/bin/python" -m ensurepip --default-pip
            COMMAND "${venv_dir}/bin/python" -m pip install "${venv_build_dir}"
        )
    elseif(arg_PROJECTFILE STREQUAL "uv.toml")
        add_custom_command(
            OUTPUT "${venv_dir}"
            DEPENDS "${venv_build_dir}"
            COMMAND uv venv "${venv_dir}" --config-file "${venv_build_dir}/uv.toml"
        )
    else()
        message(FATAL_ERROR "unsupported project type ${arg_PROJECTFILE}")
    endif()

    add_custom_target("${target_venv}" ALL
        DEPENDS "${venv_dir}"
    )

    venv_install()
endfunction()