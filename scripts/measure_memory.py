import ctypes
from ctypes import wintypes
import os

class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]

def get_os_memory_info():
    """Returns current WorkingSet (RSS) and PeakWorkingSet in MB."""
    counters = PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
        wintypes.DWORD
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    
    hProcess = kernel32.GetCurrentProcess()
    success = psapi.GetProcessMemoryInfo(
        hProcess,
        ctypes.byref(counters),
        counters.cb
    )
    if not success:
        return 0.0, 0.0, 0.0
    rss_mb = counters.WorkingSetSize / (1024 * 1024)
    peak_rss_mb = counters.PeakWorkingSetSize / (1024 * 1024)
    private_mb = counters.PrivateUsage / (1024 * 1024)
    return round(rss_mb, 2), round(peak_rss_mb, 2), round(private_mb, 2)

if __name__ == "__main__":
    rss, peak, private = get_os_memory_info()
    print(f"Current RSS: {rss} MB, Peak RSS: {peak} MB, Private Bytes: {private} MB")
