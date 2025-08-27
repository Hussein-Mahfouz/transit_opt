import geopandas as gpd


class StudyAreaBoundary:
    """Simple class to handle study area boundaries for spatial transit analysis."""

    def __init__(
        self,
        boundary_gdf: gpd.GeoDataFrame | None = None,
        crs: str = "EPSG:3857",
        buffer_km: float = 0.0,
    ):
        self.target_crs = self._validate_metric_crs(crs)
        self.buffer_km = buffer_km
        self.boundary_gdf = None

        if boundary_gdf is not None:
            self.set_boundary(boundary_gdf)

    def _validate_metric_crs(self, crs: str) -> str:
        """Validate that CRS is metric, raise error if not."""
        try:
            import pyproj

            crs_info = pyproj.CRS(crs)

            if crs_info.is_geographic:
                raise ValueError(
                    f"CRS {crs} is geographic (lat/lon). Please use a metric CRS like EPSG:3857 or a local UTM zone."
                )

            # Check units are metric
            if hasattr(crs_info.axis_info[0], "unit_name"):
                unit = crs_info.axis_info[0].unit_name.lower()
                if "metre" not in unit and "meter" not in unit:
                    raise ValueError(
                        f"CRS {crs} uses non-metric units ({unit}). Please use a metric CRS."
                    )

            print(f"✅ Validated metric CRS: {crs}")
            return crs

        except ImportError:
            raise ImportError(
                "pyproj is required for CRS validation. Install with: pip install pyproj"
            )
        except Exception as e:
            if "geographic" in str(e) or "metric" in str(e):
                raise  # Re-raise our validation errors
            else:
                raise ValueError(f"Invalid CRS {crs}: {e}")

    def set_boundary(self, boundary_gdf: gpd.GeoDataFrame):
        """Set the boundary from a GeoDataFrame with CRS conversion."""
        original_crs = boundary_gdf.crs

        # Convert to target CRS if different
        if original_crs != self.target_crs:
            print(f"🔄 Converting boundary CRS: {original_crs} → {self.target_crs}")
            boundary_gdf = boundary_gdf.to_crs(self.target_crs)

        # Apply buffer if specified
        if self.buffer_km > 0:
            boundary_gdf = self._apply_buffer(boundary_gdf)

        # Dissolve multiple polygons into one if needed
        if len(boundary_gdf) > 1:
            boundary_gdf = boundary_gdf.dissolve().reset_index(drop=True)

        self.boundary_gdf = boundary_gdf
        print(f"✅ Study area set: {len(boundary_gdf)} polygon(s) in {self.target_crs}")

    @classmethod
    def from_file(cls, file_path: str, crs: str = "EPSG:3857", buffer_km: float = 0.0):
        """Load boundary from a spatial file with automatic CRS conversion."""
        try:
            boundary_gdf = gpd.read_file(file_path)

            print(f"📁 Loaded boundary from {file_path}")
            print(f"   Original CRS: {boundary_gdf.crs}, Features: {len(boundary_gdf)}")

            return cls(boundary_gdf, crs, buffer_km)

        except Exception as e:
            raise ValueError(f"Could not load boundary from {file_path}: {e}")

    @classmethod
    def from_bounds(cls, minx: float, miny: float, maxx: float, maxy: float, **kwargs):
        """Create rectangular boundary from bounding box coordinates."""
        raise NotImplementedError("Coming soon")

    @classmethod
    def from_center_radius(
        cls, center_lat: float, center_lon: float, radius_km: float, **kwargs
    ):
        """Create circular boundary around a center point."""
        raise NotImplementedError("Coming soon")

    @classmethod
    def auto_detect_from_stops(cls, stops_gdf: gpd.GeoDataFrame, **kwargs):
        """Auto-detect study area from transit stop distribution."""
        raise NotImplementedError("Coming soon")

    def _apply_buffer(
        self, boundary_gdf: gpd.GeoDataFrame, buffer_km: float | None = None
    ) -> gpd.GeoDataFrame:
        """Apply buffer to boundary (already in metric CRS)."""
        buffer_radius = buffer_km if buffer_km is not None else self.buffer_km

        if buffer_radius <= 0:
            return boundary_gdf

        # Since we enforce metric CRS, buffer directly in meters
        buffer_m = buffer_radius * 1000
        buffered = boundary_gdf.copy()
        buffered.geometry = boundary_gdf.geometry.buffer(buffer_m)

        print(f"📏 Applied {buffer_radius}km buffer")
        return buffered

    def filter_points(
        self, points_gdf: gpd.GeoDataFrame, output_crs: str | None = None
    ) -> gpd.GeoDataFrame:
        """
        Filter points to those within the study area boundary.

        Args:
            points_gdf: GeoDataFrame of points to filter
            output_crs: CRS for the output (defaults to boundary CRS)

        Returns:
            Filtered GeoDataFrame in specified CRS
        """
        if self.boundary_gdf is None:
            raise ValueError(
                "No boundary set. Use set_boundary() or from_file() first."
            )

        # Determine output CRS
        if output_crs is None:
            output_crs = self.target_crs

        # Convert points to boundary CRS for filtering
        if points_gdf.crs != self.boundary_gdf.crs:
            points_for_filtering = points_gdf.to_crs(self.boundary_gdf.crs)
        else:
            points_for_filtering = points_gdf

        # Spatial filter
        filtered_points = gpd.sjoin(
            points_for_filtering, self.boundary_gdf, how="inner", predicate="within"
        )

        # Keep only original columns
        original_columns = points_gdf.columns
        filtered_points = filtered_points[original_columns]

        # Convert to desired output CRS if different
        if filtered_points.crs != output_crs:
            print(f"🔄 Converting output to {output_crs}")
            filtered_points = filtered_points.to_crs(output_crs)

        print(f"🔍 Filtered {len(points_gdf)} → {len(filtered_points)} points")
        return filtered_points

    def filter_grid(
        self,
        grid_gdf: gpd.GeoDataFrame,
        predicate: str = "intersects",
        output_crs: str | None = None,
    ) -> gpd.GeoDataFrame:
        """
        Filter grid cells to those intersecting the study area.

        Args:
            grid_gdf: GeoDataFrame of grid cells
            predicate: Spatial relationship ('intersects', 'within', 'contains')
            output_crs: CRS for the output (defaults to boundary CRS)

        Returns:
            Filtered grid GeoDataFrame in specified CRS
        """
        if self.boundary_gdf is None:
            raise ValueError("No boundary set.")

        # Determine output CRS
        if output_crs is None:
            output_crs = self.target_crs

        # Convert grid to boundary CRS for filtering
        if grid_gdf.crs != self.boundary_gdf.crs:
            grid_for_filtering = grid_gdf.to_crs(self.boundary_gdf.crs)
        else:
            grid_for_filtering = grid_gdf

        # Spatial filter
        filtered_grid = gpd.sjoin(
            grid_for_filtering, self.boundary_gdf, how="inner", predicate=predicate
        )

        # Keep only original columns and remove duplicates
        original_columns = grid_gdf.columns
        filtered_grid = filtered_grid[original_columns].drop_duplicates()

        # Convert to desired output CRS if different
        if filtered_grid.crs != output_crs:
            print(f"🔄 Converting output to {output_crs}")
            filtered_grid = filtered_grid.to_crs(output_crs)

        print(f"🔍 Filtered {len(grid_gdf)} → {len(filtered_grid)} grid cells")
        return filtered_grid

    def get_boundary(self, output_crs: str | None = None) -> gpd.GeoDataFrame:
        """
        Get the boundary in specified CRS.

        Args:
            output_crs: Desired CRS (defaults to boundary's current CRS)

        Returns:
            Boundary GeoDataFrame in specified CRS
        """
        if self.boundary_gdf is None:
            raise ValueError("No boundary set.")

        if output_crs is None or output_crs == self.boundary_gdf.crs:
            return self.boundary_gdf.copy()
        else:
            print(f"🔄 Converting boundary: {self.boundary_gdf.crs} → {output_crs}")
            return self.boundary_gdf.to_crs(output_crs)

    def add_buffer(
        self, buffer_km: float, update_boundary: bool = True
    ) -> gpd.GeoDataFrame:
        """
        Add buffer to the current boundary.

        Args:
            buffer_km: Buffer radius in kilometers
            update_boundary: Whether to update the instance boundary or return new one

        Returns:
            Buffered boundary GeoDataFrame
        """
        if self.boundary_gdf is None:
            raise ValueError("No boundary set.")

        buffered = self._apply_buffer(self.boundary_gdf, buffer_km)

        if update_boundary:
            self.boundary_gdf = buffered
            self.buffer_km = buffer_km
            print(f"✅ Updated boundary with {buffer_km}km buffer")

        return buffered

    def visualize(self, ax=None, viz_crs: str | None = None, **plot_kwargs):
        """
        Visualize the boundary.

        Args:
            ax: Matplotlib axis (optional)
            viz_crs: CRS for visualization (defaults to EPSG:4326)
            **plot_kwargs: Additional plotting arguments
        """
        if viz_crs is None:
            viz_crs = self.boundary_gdf.crs  # Use boundary's actual CRS

        if self.boundary_gdf is None:
            raise ValueError("No boundary set.")

        if ax is None:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # Get boundary in visualization CRS
        viz_boundary = self.get_boundary(output_crs=viz_crs)

        # Default styling
        default_kwargs = {
            "facecolor": "lightblue",
            "edgecolor": "darkblue",
            "alpha": 0.3,
        }
        default_kwargs.update(plot_kwargs)

        viz_boundary.plot(ax=ax, **default_kwargs)
        ax.set_title(f"Study Area (CRS: {viz_crs})")

        if viz_crs == "EPSG:4326":
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        return ax
