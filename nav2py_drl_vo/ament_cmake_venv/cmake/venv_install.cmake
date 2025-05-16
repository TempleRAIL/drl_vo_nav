function(venv_install)
    install(
        DIRECTORY "${venv_dir}"
        DESTINATION "${venv_install_dir}/.."
    )
endfunction()