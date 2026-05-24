use std::io::Error;

/// Get the physical footprint (in bytes) of a process on macOS without shelling out to `sudo footprint`.
/// Works without sudo if the process is owned by the current user.
pub fn get_phys_footprint(pid: u32) -> Result<u64, Error> {
    #[cfg(target_os = "macos")]
    {
        unsafe {
            let mut rusage: libc::rusage_info_v4 = std::mem::zeroed();
            // proc_pid_rusage treats buffer as a flat destination address
            // (XNU kernel uses user_addr_t internally). Pass the struct
            // pointer directly — NOT a pointer-to-pointer.
            let ret = libc::proc_pid_rusage(
                pid as libc::pid_t,
                libc::RUSAGE_INFO_V4,
                &mut rusage as *mut libc::rusage_info_v4 as *mut libc::rusage_info_t,
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