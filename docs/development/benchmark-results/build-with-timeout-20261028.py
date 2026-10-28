"""Run the unmodified build script with a longer shared-host compile deadline."""
import runpy
import subprocess
import sys

original_run = subprocess.run


def run(*args, **kwargs):
    if kwargs.get('timeout') == 300:
        kwargs['timeout'] = 1200
    return original_run(*args, **kwargs)


subprocess.run = run
sys.argv = ['scripts/build_native_core.py', '--cxx', 'g++']
runpy.run_path('scripts/build_native_core.py', run_name='__main__')
