import psutil
import time


def get_system_stats():

    cpu_usage = psutil.cpu_percent(interval=1)

    memory_usage = psutil.virtual_memory().percent

    return {
        "cpu": cpu_usage,
        "memory": memory_usage
    }


def measure_processing_time(start_time):

    return round(
        time.time() - start_time,
        3
    )