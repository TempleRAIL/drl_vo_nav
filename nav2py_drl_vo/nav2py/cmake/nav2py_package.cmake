function(nav2py_package package)
  find_package(nav2_common REQUIRED)
  find_package(ament_cmake_venv_uv REQUIRED)

  nav2_package()

  get_filename_component(nav2py_python_DIR "${nav2py_DIR}/../nav2py" ABSOLUTE)
  get_filename_component(package_python_DIR "${package}" ABSOLUTE)

  venv_ensure_pip()
  venv_pip_install_local("${nav2py_python_DIR}")
  venv_pip_install_local("${package_python_DIR}")
  venv_script("nav2py_run" "-m ${package}")
endfunction()