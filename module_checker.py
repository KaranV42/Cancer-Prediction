import importlib.util
import subprocess
import sys

package_name = 'plotly'
   
def install_package(package_name):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])

if importlib.util.find_spec(package_name) is None:
    print(f"{package_name} is not installed. Installing...")
    install_package(package_name)
    print(f"{package_name} has been installed.")
else:
    print(f"{package_name} is already installed.")
