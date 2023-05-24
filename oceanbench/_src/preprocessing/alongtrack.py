import xarray as xr
import numpy as np

def alongtrack_ssh(ds: xr.Dataset, variable: str="ssh") -> xr.Dataset:
    ds["ssh"] = ds["sla_unfiltered"] + ds["mdt"] - ds["lwe"]
    return ds

def remove_swath_dimension(ds: xr.Dataset, name: str="nC") -> xr.Dataset:
    """Removes the swath dimension fromt he SWOT
    simulations
    
    Args:
        ds (xr.Dataset): an xr.Dataset where we have the swath dimension
            "nC" and the time dimensions, "time"
        name (str): the name of the swath dimension, default="nC"
    
    Returns:
        ds (xr.Dataset): an xr.Dataset in the alongtrack format
    """
    
    msg = "mismatch in dimensions to collapse"
    msg += f"\nName: {name} | Dims: {ds.dims}"
    assert name in ds.dims, msg
    
    return ds.rename({"time": "z"}).stack(time=(name, "z")).set_index({"time": "z"}).reset_coords([name]).sortby("time")


def select_track_segments(
    ds: xr.Dataset,
    variable: str="ssh",
    variable_interp: str="ssh_interp",
    delta_x: float = None,
    delta_t: float = 0.9434,
    velocity: float = 6.77,
    length_scale: float = 1_000,
    segment_overlapping: float = 0.25,
):
    """
    Parameters:
    -----------
    delta_t : float, default=0.9434
        the number of seconds
    velocity : float, default=6.77
        the velocity (km/s)
    length_scale: float, default=1_000
        the segment length cale in km
    segment_overlapping: float=0.25
        the amount of segment overlapping allowed
    """
    time_alongtrack = ds.time.values
    lat_alongtrack = ds.lat.values
    lon_alongtrack = ds.lon.values
    ssh_alongtrack = ds[variable].values
    ssh_map_interp = ds[variable_interp].values
    if delta_x is None:
        delta_x = velocity * delta_t
    # max delta t of 4 seconds to cut tracks
    max_delta_t_gap = 4 * np.timedelta64(1, "s")
    # get number of points to consider for resolution = lengthscale in km
    delta_t_jd = delta_t / (3600 * 24)
    npt = int(length_scale / delta_x)

    # cut tracks when diff time longer than 4 delta t
    indx = np.where((np.diff(time_alongtrack) > max_delta_t_gap))[0]
    track_segment_length = np.insert(np.diff(indx), [0], indx[0])

    list_lat_segment = []
    list_lon_segment = []
    list_ssh_alongtrack_segment = []
    list_ssh_map_interp_segment = []

    # Long track >= npt
    selected_track_segment = np.where(track_segment_length >= npt)[0]

    if selected_track_segment.size > 0:

        for track in selected_track_segment:

            if track - 1 >= 0:
                index_start_selected_track = indx[track - 1]
                index_end_selected_track = indx[track]
            else:
                index_start_selected_track = 0
                index_end_selected_track = indx[track]

            start_point = index_start_selected_track
            end_point = index_end_selected_track

            for sub_segment_point in range(
                start_point, end_point - npt, int(npt * segment_overlapping)
            ):

                # Near Greenwhich case
                if (
                    (lon_alongtrack[sub_segment_point + npt - 1] < 50.0)
                    and (lon_alongtrack[sub_segment_point] > 320.0)
                ) or (
                    (lon_alongtrack[sub_segment_point + npt - 1] > 320.0)
                    and (lon_alongtrack[sub_segment_point] < 50.0)
                ):

                    tmp_lon = np.where(
                        lon_alongtrack[sub_segment_point : sub_segment_point + npt]
                        > 180,
                        lon_alongtrack[sub_segment_point : sub_segment_point + npt]
                        - 360,
                        lon_alongtrack[sub_segment_point : sub_segment_point + npt],
                    )
                    mean_lon_sub_segment = np.median(tmp_lon)

                    if mean_lon_sub_segment < 0:
                        mean_lon_sub_segment = mean_lon_sub_segment + 360.0
                else:

                    mean_lon_sub_segment = np.median(
                        lon_alongtrack[sub_segment_point : sub_segment_point + npt]
                    )

                mean_lat_sub_segment = np.median(
                    lat_alongtrack[sub_segment_point : sub_segment_point + npt]
                )

                ssh_alongtrack_segment = np.ma.masked_invalid(
                    ssh_alongtrack[sub_segment_point : sub_segment_point + npt]
                )

                ssh_map_interp_segment = []
                ssh_map_interp_segment = np.ma.masked_invalid(
                    ssh_map_interp[sub_segment_point : sub_segment_point + npt]
                )
                if np.ma.is_masked(ssh_map_interp_segment):
                    ssh_alongtrack_segment = np.ma.compressed(
                        np.ma.masked_where(
                            np.ma.is_masked(ssh_map_interp_segment),
                            ssh_alongtrack_segment,
                        )
                    )
                    ssh_map_interp_segment = np.ma.compressed(ssh_map_interp_segment)

                if ssh_alongtrack_segment.size > 0:
                    list_ssh_alongtrack_segment.append(ssh_alongtrack_segment)
                    list_lon_segment.append(mean_lon_sub_segment)
                    list_lat_segment.append(mean_lat_sub_segment)
                    list_ssh_map_interp_segment.append(ssh_map_interp_segment)
    
    num_segments = len(list_lon_segment)
    
    ds = xr.Dataset(
        {
            # variable_interp: (("track_val", "segment"), np.asarray(list_ssh_map_interp_segment).T),
            # variable: (("track_val", "segment",), np.asarray(list_ssh_alongtrack_segment).T),
            variable_interp: (("segment", "track_val", ), np.asarray(list_ssh_map_interp_segment)),
            variable: (("segment", "track_val", ), np.asarray(list_ssh_alongtrack_segment)),
        },
        coords={
            "segment": np.arange(num_segments),
            "lat": (["segment"], np.asarray(list_lon_segment)),
            "lon": (["segment"], np.asarray(list_lat_segment)),
            "track_val": np.arange(npt)
        }
    )
    
    attrs = dict(delta_x=delta_x, velocity=velocity, length_scale=length_scale, nperseg=npt)
    
    ds.attrs = attrs
    return ds