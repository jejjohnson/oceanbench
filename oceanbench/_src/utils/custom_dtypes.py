from typing import NamedTuple, Literal
from dataclasses import dataclass
from xarray_dataclasses import Coordof, Attr, Coord, Data, Name, asdataarray, asdataset, AsDataArray, AsDataset, Dataof
import numpy as np
import pandas as pd
from oceanbench._src.geoprocessing.gridding import create_coord_grid


X = Literal["x"]
Y = Literal["y"]
Z = Literal["z"]
LON = Literal["lon"]
LAT = Literal["lat"]
TIME = Literal["time"]


@dataclass
class Bounds:
    val_min: float
    val_max: float
    name: str = ""


@dataclass
class Region:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    name: str = ""


@dataclass
class Period:
    t_min: float
    t_max: float
    name: str = ""

    @classmethod
    def init_from_str(cls, t_min: str, t_max: str, **kwargs):
        t_min = pd.to_datetime(t_min)
        t_max = pd.to_datetime(t_max)
        return cls(t_min=t_min, t_max=t_max, **kwargs)


@dataclass
class CoordinateAxis:
    data: Data[X, np.ndarray]

    @classmethod
    def init_from_limits(cls, x_min: float, x_max: float, dx: float, **kwargs):
        data = np.arange(x_min, x_max + dx, dx)
        return cls(data=data, **kwargs)
    
    @property
    def ndim(self):
        return len(self.data)


@dataclass
class LongitudeAxis(CoordinateAxis):
    data: Data[LON, np.ndarray]
    name: Name[str] = "lon"
    standard_name: Attr[str] = "longitude"
    long_name: Attr[str] = "Longitude"
    units: Attr[str] = "degrees_east"

    @classmethod
    def init_from_limits(cls, lon_min: float, lon_max: float, dlon: float, **kwargs):
        data = np.arange(lon_min, lon_max + dlon, dlon)
        return cls(data=data, **kwargs)


@dataclass
class LatitudeAxis(CoordinateAxis):
    data: Data[LAT, np.ndarray]
    name: Name[str] = "lat"
    standard_name: Attr[str] = "latitude"
    long_name: Attr[str] = "Latitude"
    units: Attr[str] = "degrees_west"

    @classmethod
    def init_from_limits(cls, lat_min: float, lat_max: float, dlat: float, **kwargs):
        data = np.arange(lat_min, lat_max + dlat, dlat)
        return cls(data=data, **kwargs)


@dataclass
class TimeAxis:
    data: Data[TIME, Literal["datetime64[ns]"]]
    name: Name[str] = "time"
    long_name: Attr[str] = "Date"

    @classmethod
    def init_from_limits(cls, t_min: str, t_max: str, dt: str, **kwargs):
        t_min = pd.to_datetime(t_min)
        t_max = pd.to_datetime(t_max)
        dt = pd.to_timedelta(dt)
        data = np.arange(t_min, t_max + dt, dt)
        return cls(data=data, **kwargs)

    @property
    def ndim(self):
        return len(self.data)


@dataclass
class Grid2D(AsDataArray):
    lon: Coordof[LongitudeAxis] = 0
    lat: Coordof[LatitudeAxis] = 0

    @property
    def ndim(self):
        return (self.lat.ndim, self.lon.ndim)
    
    @property
    def grid(self):
        return create_coord_grid(self.lat.data, self.lon.data)
    

@dataclass
class Grid2DT(Grid2D):
    time: Coordof[TimeAxis] = 0

    @property
    def ndim(self):
        return (self.time.ndim, self.lat.ndim, self.lon.ndim)


@dataclass
class SSH2D:
    data: Data[tuple[LAT, LON], np.ndarray]
    lat: Coordof[LatitudeAxis] = 0
    lon: Coordof[LongitudeAxis] = 0
    name: Name[str] = "ssh"
    units: Attr[str] = "m"
    standard_name: Attr[str] = "sea_surface_height"
    long_name: Attr[str] = "Sea Surface Height"

    @property
    def ndim(self):
        return (self.lat.ndim, self.lon.ndim)

    @classmethod
    def init_from_axis(cls, lon: LongitudeAxis, lat: LatitudeAxis):
        data_init = np.ones((lat.ndim, lon.ndim))
        return cls(data=data_init, lon=lon, lat=lat)
    
    @classmethod
    def init_from_grid(cls, grid: Grid2D):
        return cls(data=np.ones(grid.ndim), lon=grid.lon, lat=grid.lat)


@dataclass
class SSH2DT:
    data: Data[tuple[TIME, LAT, LON], np.ndarray]
    time: Coordof[TimeAxis] = 0
    lat: Coordof[LatitudeAxis] = 0
    lon: Coordof[LongitudeAxis] = 0
    name: Name[str] = "ssh"
    units: Attr[str] = "m"
    standard_name: Attr[str] = "sea_surface_height"
    long_name: Attr[str] = "Sea Surface Height"

    @property
    def ndim(self):
        return (self.time.ndim, self.lat.ndim, self.lon.ndim)

    @classmethod
    def init_from_axis(cls, lon: LongitudeAxis, lat: LatitudeAxis, time: TimeAxis):
        return cls(
            data=np.ones((time.ndim, lat.ndim, lon.ndim)),
            time=time, lon=lon, lat=lat)
    
    @classmethod
    def init_from_grid(cls, grid: Grid2DT):
        return cls(
            data=np.ones(grid.ndim),
            lon=grid.lon, lat=grid.lat, time=grid.time)
