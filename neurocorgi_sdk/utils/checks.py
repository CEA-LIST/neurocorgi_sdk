# NeuroCorgi SDK, CeCILL-C license

import contextlib
import requests


def check_latest_pypi_version(package_name='neurocorgi_sdk'):
    """
    Returns the latest version of a PyPI package without downloading or installing it.

    Parameters:
        package_name (str): The name of the package to find the latest version for.

    Returns:
        (str): The latest version of the package.
    """
    with contextlib.suppress(Exception):
        requests.packages.urllib3.disable_warnings()  # Disable the InsecureRequestWarning
        response = requests.get(f'https://pypi.org/pypi/{package_name}/json', timeout=3)
        if response.status_code == 200:
            return response.json()['info']['version']
