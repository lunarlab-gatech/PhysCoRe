"""
Open3D window helper: on-screen by default, offscreen when headless.
"""

import open3d as o3d


def create_visualizer(headless=False):
    """Return an Open3D Visualizer.

    On-screen by default (needs a display + GL). With ``headless=True`` the
    window is created offscreen so it runs over ssh / on a GPU node. If an
    on-screen window can't be created (no display), Open3D yields an invalid
    view control; we catch that and tell the user to pass --headless.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=not headless)
    if vis.get_view_control() is None:
        vis.destroy_window()
        raise SystemExit(
            "Open3D could not open an on-screen window — this looks like a "
            "headless run with no display. Re-run with --headless."
        )
    return vis
