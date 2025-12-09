from pathlib import Path
import requests
import xarray as xr
import numpy as np
import string
from requests.exceptions import SSLError
import warnings
import certifi
from urllib3.exceptions import InsecureRequestWarning

# RCM data server has an invalid SSL cert, so we ignore the warnings.
warnings.filterwarnings("ignore", category=InsecureRequestWarning)

# downloader helper, with sanity checks and progress
# Safe-ish, with cert verification but falls back to unverified if needed.
def download(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Download URL: with streaming, progress, and sanity checks."""
    print(f"GET {url}")

    try:
        # 1) Try with proper verification
        r = requests.get(url, stream=True, timeout=120, verify=certifi.where())
    except SSLError as e:
        print(f"  ! SSL error for {url}: {e}")
        print("    Retrying without certificate verification (INSECURE).")
        # 2) Fallback: skip cert verification (only do this if you accept the risk)
        r = requests.get(url, stream=True, timeout=120, verify=False)

    r.raise_for_status()

    # make sure we are getting a netcdf file
    if "netcdf" not in r.headers.get("Content-Type", "").lower():
        raise RuntimeError(
            f"{url} unexpected Content-Type {r.headers.get('Content-Type')}"
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    bytes_written = 0
    # stream to file
    with open(dest, "wb") as f:
        for chunk in r.iter_content(chunk_size):
            if chunk:
                f.write(chunk)
                bytes_written += len(chunk)

    if bytes_written < 10_000:
        raise RuntimeError(
            f"{url} -> only {bytes_written} bytes (looks like an error page)."
        )

    print(f"saved {bytes_written/1e6:.1f} MBs to {dest}")

def to_xy_and_vec(obj, var_name=None):
    """
    Flatten a 2-D DataArray into:

        xy   : shape (2, N)   [ latitudes ; longitudes ]
        vec  : shape (N,)     data values

    Works for both regular (1-D lat/lon) and curvilinear / rotated grids
    (2-D lat/lon).
    """
    # get datarray variable
    if isinstance(obj, xr.Dataset):
        if var_name is None:
            raise ValueError("When you pass a Dataset, you must give var_name")
        da = obj[var_name]
    else:
        da = obj

    if da.ndim != 2:
        raise ValueError(f"Expected a *2-D* DataArray, got dims={da.dims}")

    # locate lat/lon coordinates
    if "lat" in da.coords and "lon" in da.coords:
        lat_c = da["lat"]
        lon_c = da["lon"]
    else:
        raise KeyError("Could not find 'lat' and 'lon' coordinates")

    # build
    if lat_c.ndim == 1 and lon_c.ndim == 1:          # regular grid
        lon2d, lat2d = np.meshgrid(lon_c, lat_c)
    else:                                            # curvilinear / rotated
        lat2d, lon2d = lat_c, lon_c

    # flatten
    lat_flat = np.asarray(lat2d).ravel()
    lon_flat = np.asarray(lon2d).ravel()
    val_flat = np.asarray(da).ravel()

    mask = np.isfinite(val_flat)
    xy   = np.vstack([lat_flat[mask], lon_flat[mask]])   # (2, N)
    vec  = val_flat[mask]                                # (N,)

    return xy, vec

# Change this if you want a different date or variable,
# (check references too see which dates/variables are available)
# this is my birthdate :)
DATE  = "1996-01-25"
VAR   = "tasmax"

# base links
gcm_base = "https://crd-data-donnees-rdc.ec.gc.ca/CCCMA/products/CanSISE/output/CCCma/CanESM2/"
rcm_base = "https://climex-data.srv.lrz.de/Public/CanESM2_driven_50_members/tasmax"

# build member lists, different naming conventions for GCM vs RCM
# read reference paper for details
# https://journals.ametsoc.org/view/journals/apme/58/4/jamc-d-18-0021.1.xml
rcm_prefixes = []
for a in string.ascii_lowercase[1:4]:   # 'b', 'c', 'd' → 'ba' to 'dz'
    for b in string.ascii_lowercase:
        rcm_prefixes.append(f'k{a}{b}')
        if len(rcm_prefixes) == 50:
            break
    if len(rcm_prefixes) == 50:
        break
gcm_add = [
    f"historical-r{r}/day/atmos/tasmax/r{i}i1p1/"
    f"tasmax_day_CanESM2_historical-r{r}_r{i}i1p1_19500101-20201231.nc"
    for r in range(1, 6)   # historical-r1 through historical-r5
    for i in range(1, 11)  # r1i1p1 through r10i1p1
]
rcm_add = [
    f"{rcm_prefixes[k]}/1996/"
    f"tasmax_EUR-11_CCCma-CanESM2_historical_r{r}-r{i}i1p1_"
    f"OURANOS-CRCM5_{rcm_prefixes[k]}_day_199601.nc"
    for k, (r, i) in enumerate((r, i) for r in range(1, 6) for i in range(1, 11))
]
# There should be exactly 50 members each
assert len(gcm_add) == 50
assert len(rcm_add) == 50

# allocate
obs_gcm = np.empty((50, 336))
obs_rcm = np.empty((50, 78400))
xy_gcm_final, xy_rcm_final = None, None

# download loop
for m in range(50):
    print(f"\n=== member {m+1} ===")

    gcm_file = Path(f"gcm_{m}.nc")
    rcm_file = Path(f"rcm_{m}.nc")
    print(f"GCM {m+1}: {gcm_base + gcm_add[m]}")
    print(f"RCM {m+1}: {rcm_base + '/' + rcm_add[m]}\n")

    download(f"{gcm_base}/{gcm_add[m]}", gcm_file)
    download(f"{rcm_base}/{rcm_add[m]}", rcm_file)

    # download, open, slice
    with xr.open_dataset(gcm_file, engine="netcdf4") as gcm_ds:
        gcm_day = gcm_ds[VAR].sel(time=DATE).squeeze().load()  

    with xr.open_dataset(rcm_file, engine="netcdf4") as rcm_ds:
        rcm_day = rcm_ds[VAR].sel(time=DATE).squeeze().load()   

    # RCM Box for GCM, was done empirically by looking at RCM extent
    west, east = -26, 40
    gcm_box = (gcm_day
               .sel(lat=slice(29, 66))
               .assign_coords(lon=((gcm_day.lon + 180) % 360) - 180)
               .sortby("lon")
               .sel(lon=slice(west, east)))

    # flatten
    xy_gcm, vec_gcm = to_xy_and_vec(gcm_box)
    xy_rcm, vec_rcm = to_xy_and_vec(rcm_day)

    obs_gcm[m] = vec_gcm
    obs_rcm[m] = vec_rcm

    # Delete intermediate files
    gcm_file.unlink()
    rcm_file.unlink()

# Save everything on current folder, move them if needed
# Transpose for locs, easier for BTM later.
np.save("../tests/data/obs_gcm.npy", obs_gcm)
np.save("../tests/data/obs_rcm.npy", obs_rcm)
np.save("../tests/data/locs_gcm.npy", xy_gcm.T)
np.save("../tests/data/locs_rcm.npy", xy_rcm.T)
print("\nAll done — data saved.")
