use std::io::Error;

/// Get the physical footprint (in bytes) of a process on macOS without shelling out to `sudo footprint`.
/// Works without sudo if the process is owned by the current user.
pub fn get_phys_footprint(pid: u32) -> Result<u64, Error> {
    #[cfg(target_os = "macos")]
    {
        unsafe {
            let mut rusage: libc::rusage_info_v4 = std::mem::zeroed();
            let mut info: libc::rusage_info_t = &mut rusage as *mut _ as libc::rusage_info_t;
            let ret = libc::proc_pid_rusage(
                pid as libc::pid_t,
                libc::RUSAGE_INFO_V4,
                &mut info,
            );

            if ret == 0 {
                Ok(rusage.ri_phys_footprint)
            } else {
                Err(Error::last_os_error())
            }
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        Err(Error::new(std::io::ErrorKind::Unsupported, "Not macOS"))
    }
}