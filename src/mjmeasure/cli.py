#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import mujoco
import numpy as np
import trimesh
import viser
from mjviser import ViserMujocoScene
from mjviser.conversions import create_primitive_mesh, mujoco_mesh_to_trimesh

MeasureMode = Literal["point2point", "plane2point"]
MARKER_RADIUS = 0.005
PROJECTION_MARKER_RADIUS = 0.004
MEASURE_COLOR = np.array([255, 220, 80], dtype=np.uint8)


@dataclass
class PickableGeom:
    geom_id: int
    name: str
    mesh: trimesh.Trimesh
    intersector: trimesh.ray.ray_triangle.RayMeshIntersector


@dataclass
class RayHit:
    geom_name: str
    position: np.ndarray
    normal: np.ndarray
    distance: float
    triangle_id: int


class MeasureTool:
    def __init__(
        self,
        server: viser.ViserServer,
        data: mujoco.MjData,
        pickables: list[PickableGeom],
    ) -> None:
        self.server = server
        self.data = data
        self.pickables = pickables
        self.enabled = False
        self.mode: MeasureMode = "point2point"
        self.points: list[RayHit] = []
        self.handles: list[Any] = []
        self._status: Any = None
        self._mode_buttons: dict[MeasureMode, Any] = {}

    def create_gui(self) -> None:
        with self.server.gui.add_folder("Measure"):
            self._status = self.server.gui.add_html("")
            self._set_status("Click a measurement mode to start.")
            self._mode_buttons["point2point"] = self.server.gui.add_button(
                "point2point",
                color="gray",
                icon=viser.Icon.RULER,
            )
            self._mode_buttons["plane2point"] = self.server.gui.add_button(
                "plane2point",
                color="gray",
                icon=viser.Icon.RULER_2,
            )
            clear = self.server.gui.add_button("Clear", icon=viser.Icon.X)

            @self._mode_buttons["point2point"].on_click
            def _(_) -> None:
                self._activate_mode("point2point")

            @self._mode_buttons["plane2point"].on_click
            def _(_) -> None:
                self._activate_mode("plane2point")

            @clear.on_click
            def _(_) -> None:
                self.clear()

    def _activate_mode(self, mode: MeasureMode) -> None:
        if self.enabled and self.mode == mode:
            self._clear_measurement()
            self._disarm("Click a measurement mode to start.")
            return

        self.mode = mode
        self._clear_measurement()
        self._arm()

    def _update_mode_button_colors(self) -> None:
        for mode, button in self._mode_buttons.items():
            button.color = "blue" if self.enabled and self.mode == mode else "gray"

    def _arm(self) -> None:
        if not self.enabled:
            self.enabled = True
            self._register_pointer_callback()
        self._update_mode_button_colors()
        self._set_status(self._prompt_for_next_click())

    def _disarm(self, status: str | None = "Measure mode disabled.") -> None:
        if self.enabled:
            self.enabled = False
            self._remove_pointer_callback()
        self._update_mode_button_colors()
        if status is not None:
            self._set_status(status)

    def clear(self) -> None:
        self._clear_measurement()
        self._disarm("Click a measurement mode to start.")

    def _clear_measurement(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.points.clear()

    def _register_pointer_callback(self) -> None:
        @self.server.scene.on_pointer_event(event_type="click")
        def _(event: viser.ScenePointerEvent) -> None:
            self._handle_click(event)

    def _remove_pointer_callback(self) -> None:
        if self.server.scene._scene_pointer_cb is None:
            return
        self.server.scene.remove_pointer_callback()

    def _handle_click(self, event: viser.ScenePointerEvent) -> None:
        hit = self._raycast(
            np.asarray(event.ray_origin, dtype=np.float64),
            np.asarray(event.ray_direction, dtype=np.float64),
        )
        if hit is None:
            self._set_status("No geom hit. Click directly on a rendered surface.")
            return

        self.points.append(hit)
        self._draw_marker(hit, len(self.points))

        if len(self.points) == 1:
            self._set_status(self._second_click_prompt(hit))
            return

        self._draw_measurement()
        self._disarm(status=None)

    def _raycast(self, origin: np.ndarray, direction: np.ndarray) -> RayHit | None:
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            return None
        direction = direction / norm

        closest: RayHit | None = None
        for item in self.pickables:
            geom_id = item.geom_id
            rot = self.data.geom_xmat[geom_id].reshape(3, 3)
            pos = self.data.geom_xpos[geom_id]
            local_origin = rot.T @ (origin - pos)
            local_direction = rot.T @ direction

            locations, _, triangle_ids = item.intersector.intersects_location(
                local_origin.reshape(1, 3),
                local_direction.reshape(1, 3),
            )
            if len(locations) == 0:
                continue

            distances = np.linalg.norm(locations - local_origin.reshape(1, 3), axis=1)
            idx = int(np.argmin(distances))
            world_pos = rot @ locations[idx] + pos
            local_normal = item.mesh.face_normals[int(triangle_ids[idx])]
            world_normal = rot @ np.asarray(local_normal, dtype=np.float64)
            normal_norm = np.linalg.norm(world_normal)
            if normal_norm == 0.0:
                continue
            world_normal = world_normal / normal_norm
            distance = float(np.linalg.norm(world_pos - origin))
            if closest is None or distance < closest.distance:
                closest = RayHit(
                    geom_name=item.name,
                    position=np.asarray(world_pos, dtype=np.float64),
                    normal=np.asarray(world_normal, dtype=np.float64),
                    distance=distance,
                    triangle_id=int(triangle_ids[idx]),
                )

        return closest

    def _draw_marker(self, hit: RayHit, index: int) -> None:
        color = (255, 80, 80) if index == 1 else (80, 160, 255)
        marker = self.server.scene.add_icosphere(
            f"/measure/point_{index}",
            radius=MARKER_RADIUS,
            color=color,
            subdivisions=2,
            position=hit.position,
            cast_shadow=False,
            receive_shadow=False,
        )
        self.handles.append(marker)

    def _draw_measurement(self) -> None:
        if self.mode == "plane2point":
            self._draw_plane_to_point_measurement()
            return
        self._draw_point_to_point_measurement()

    def _draw_point_to_point_measurement(self) -> None:
        point_a = self.points[0].position
        point_b = self.points[1].position
        distance_m = float(np.linalg.norm(point_b - point_a))
        self._draw_line_with_label(point_a, point_b, distance_m)
        self._set_status(
            "Distance: "
            f"{distance_m:.6f} m ({distance_m * 1000.0:.2f} mm)<br/>"
            f"A: {self.points[0].geom_name}<br/>"
            f"B: {self.points[1].geom_name}"
        )

    def _draw_plane_to_point_measurement(self) -> None:
        plane_hit = self.points[0]
        point_hit = self.points[1]
        signed_distance_m = float(
            np.dot(point_hit.position - plane_hit.position, plane_hit.normal)
        )
        distance_m = abs(signed_distance_m)
        projected_point = point_hit.position - signed_distance_m * plane_hit.normal
        self._draw_line_with_label(projected_point, point_hit.position, distance_m)
        projection_marker = self.server.scene.add_icosphere(
            "/measure/plane_projection",
            radius=PROJECTION_MARKER_RADIUS,
            color=(255, 220, 80),
            subdivisions=2,
            position=projected_point,
            cast_shadow=False,
            receive_shadow=False,
        )
        self.handles.append(projection_marker)
        self._set_status(
            "Plane-to-point distance: "
            f"{distance_m:.6f} m ({distance_m * 1000.0:.2f} mm)<br/>"
            f"Signed offset: {signed_distance_m:.6f} m<br/>"
            f"Plane: {plane_hit.geom_name} face {plane_hit.triangle_id}<br/>"
            f"Point: {point_hit.geom_name}"
        )

    def _draw_line_with_label(
        self, point_a: np.ndarray, point_b: np.ndarray, distance_m: float
    ) -> None:
        line = self.server.scene.add_line_segments(
            "/measure/line",
            points=np.array([[point_a, point_b]], dtype=np.float32),
            colors=MEASURE_COLOR,
            line_width=3.0,
        )
        label = self.server.scene.add_label(
            "/measure/label",
            text=f"{distance_m:.4f} m ({distance_m * 1000.0:.1f} mm)",
            position=(point_a + point_b) * 0.5,
            anchor="bottom-center",
        )
        self.handles.extend([line, label])

    def _prompt_for_next_click(self) -> str:
        if self.mode == "plane2point":
            return "Click a surface face to define the plane."
        return "Click the first surface point."

    def _second_click_prompt(self, hit: RayHit) -> str:
        if self.mode == "plane2point":
            return f"Plane set on {hit.geom_name}. Click the point to measure."
        return f"Point A set on {hit.geom_name}. Click point B."

    def _set_status(self, text: str) -> None:
        self._status.content = (
            '<div style="font-size: 0.9em; line-height: 1.35; '
            'padding: 0.35em 0;">'
            f"{text}"
            "</div>"
        )


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    if name:
        return name
    body_id = int(model.geom_bodyid[geom_id])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    if body_name:
        return f"{body_name}/geom_{geom_id}"
    return f"geom_{geom_id}"


def _geom_alpha(model: mujoco.MjModel, geom_id: int) -> float:
    matid = int(model.geom_matid[geom_id])
    if matid >= 0:
        return float(model.mat_rgba[matid, 3])
    return float(model.geom_rgba[geom_id, 3])


def _geom_mesh(model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh | None:
    try:
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_MESH):
            return mujoco_mesh_to_trimesh(model, geom_id)
        return create_primitive_mesh(model, geom_id)
    except (ValueError, IndexError):
        return None


def _build_pickables(model: mujoco.MjModel) -> list[PickableGeom]:
    pickables: list[PickableGeom] = []
    for geom_id in range(model.ngeom):
        if _geom_alpha(model, geom_id) == 0.0:
            continue

        mesh = _geom_mesh(model, geom_id)
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            continue

        pickables.append(
            PickableGeom(
                geom_id=geom_id,
                name=_geom_name(model, geom_id),
                mesh=mesh,
                intersector=trimesh.ray.ray_triangle.RayMeshIntersector(mesh),
            )
        )
    return pickables


def _add_camera_visualization(
    server: viser.ViserServer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    names: tuple[str, ...] = ("top", "left", "right"),
    ray_length: float = 0.20,
) -> list[str]:
    """Show camera source points and a ray pointing along each optical axis."""
    placed = []
    for cam_name in names:
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id < 0:
            continue
        pos = np.asarray(data.cam_xpos[cam_id], dtype=np.float64)
        # data.cam_xmat is row-major 9. OpenGL camera convention: -z is forward.
        R = np.asarray(data.cam_xmat[cam_id], dtype=np.float64).reshape(3, 3)
        forward = -R[:, 2]
        forward = forward / max(np.linalg.norm(forward), 1e-9)

        server.scene.add_icosphere(
            f"/cam_viz/{cam_name}/origin",
            radius=0.003,
            color=(255, 80, 80),
            position=tuple(pos),
        )
        server.scene.add_spline_catmull_rom(
            f"/cam_viz/{cam_name}/ray",
            positions=np.stack([pos, pos + ray_length * forward]),
            color=(255, 200, 80),
            line_width=4.0,
        )
        server.scene.add_label(
            f"/cam_viz/{cam_name}/label",
            text=cam_name,
            position=tuple(pos + 0.04 * forward),
        )
        placed.append(cam_name)
    return placed


def _add_camera_renders(
    server: viser.ViserServer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    names: tuple[str, ...] = ("top", "wrist_left", "wrist_right"),
    width: int = 320,
    height: int = 240,
) -> None:
    """Render each named camera once and attach the image to the GUI."""
    renderer = mujoco.Renderer(model, height=height, width=width)
    with server.gui.add_folder("Camera renders"):
        for cam_name in names:
            cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id < 0:
                server.gui.add_markdown(f"`{cam_name}` not in scene")
                continue
            renderer.update_scene(data, camera=cam_id)
            img = renderer.render()  # HxWx3 uint8 RGB
            server.gui.add_image(img, label=cam_name, format="jpeg")
    renderer.close()


def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


class WristCameraPlacer:
    """Three-click tool for placing the LEFT wrist camera (right is mirrored).

    Step 1: click a point on `left_wrist_mount_visual` to set camera origin.
    Step 2: click a plane (any geom); camera optical axis aligns with that
            plane's normal (toward or away based on UI toggle).
    Step 3 (optional): click a face whose normal defines the camera UP
            direction (flippable via toggle). If skipped, image-up falls
            back to world +z via Gram-Schmidt.

    Output: the `<body name="*_camera_d405" pos quat>` XML for left + right.
    """

    def __init__(
        self,
        server,
        model,
        data,
        pickables,
        mount_geom_name: str = "left_wrist_mount_visual",
    ):
        self.server = server
        self.model = model
        self.data = data
        self.pickables = pickables
        self.mount_geom_name = mount_geom_name
        self.origin_pickables = [p for p in pickables if p.name == mount_geom_name]
        self.stage = 0  # 0=idle, 1=awaiting origin, 2=awaiting plane, 3=ready, 4=awaiting up-face
        self.origin: Optional[np.ndarray] = None
        self.normal: Optional[np.ndarray] = None
        self.up_normal: Optional[np.ndarray] = None
        self.direction_sign: int = +1
        self.up_sign_left: int = +1
        self.up_sign_right: int = +1
        self.handles: list = []
        self._status = None
        self._output = None
        # The `*_camera_d405` body is not the rendered camera: a fixed child
        # chain rotates from the body frame to the actual `<camera>` frame.
        # Read that offset rotation straight from the loaded model so the
        # placer authors a body pose that makes the *camera* look correctly.
        self._CAM_OFFSET = self._camera_offset_in_body("left", "left_camera_d405")

    def _camera_offset_in_body(self, cam_name: str, body_name: str) -> np.ndarray:
        """Rotation of camera `cam_name` expressed in body `body_name`'s frame.

        Returns identity if either is missing (placer then assumes the body
        *is* the camera).
        """
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if cam_id < 0 or body_id < 0:
            return np.eye(3)
        R_world_cam = self.data.cam_xmat[cam_id].reshape(3, 3)
        R_world_body = self.data.xmat[body_id].reshape(3, 3)
        return R_world_body.T @ R_world_cam

    def create_gui(self) -> None:
        with self.server.gui.add_folder("Tune wrist camera"):
            self._status = self.server.gui.add_html(
                self._fmt_status(
                    "Toggle a picker ON to capture clicks. Toggle OFF to free the mouse for orbiting."
                )
            )
            self._pick_origin_cb = self.server.gui.add_checkbox(
                "Arm: pick origin", initial_value=False
            )
            self._pick_plane_cb = self.server.gui.add_checkbox(
                "Arm: pick plane", initial_value=False
            )
            self._pick_up_cb = self.server.gui.add_checkbox(
                "Arm: pick up-face", initial_value=False
            )
            self._dir_toggle = self.server.gui.add_button(
                "Flip direction (currently: toward plane)"
            )
            self._up_toggle_left = self.server.gui.add_button(
                "Flip up LEFT (currently: UP)"
            )
            self._up_toggle_right = self.server.gui.add_button(
                "Flip up RIGHT (currently: UP)"
            )
            reset = self.server.gui.add_button("Reset")
            self._offset_n = self.server.gui.add_slider(
                "Offset along normal (m)",
                min=-0.10,
                max=0.10,
                step=0.001,
                initial_value=0.0,
            )

            @self._offset_n.on_update
            def _(_):
                if self.stage >= 3:
                    self._compute_and_render(print_xml=False)

            confirm = self.server.gui.add_button("Confirm + print XML")

            @confirm.on_click
            def _(_):
                if self.origin is None or self.normal is None or self.up_normal is None:
                    self._set_status("Pick origin, plane, and up-face first.")
                    return
                self._compute_and_render(print_xml=True)

            @self._pick_origin_cb.on_update
            def _(_):
                if self._pick_origin_cb.value:
                    # Force the other off
                    if self._pick_plane_cb.value:
                        self._pick_plane_cb.value = False
                    self._reset()
                    self.stage = 1
                    self._set_status(
                        "ARMED for origin. Click the LEFT wrist mount mesh. "
                        "Toggle OFF to abort and orbit."
                    )
                    self._arm_pointer()
                else:
                    self._disarm_pointer()
                    if self.stage == 1:
                        self.stage = 0
                        self._set_status("Origin pick disarmed.")

            @self._pick_plane_cb.on_update
            def _(_):
                if self._pick_plane_cb.value:
                    if self.origin is None:
                        self._set_status("Pick origin first.")
                        self._pick_plane_cb.value = False
                        return
                    if self._pick_origin_cb.value:
                        self._pick_origin_cb.value = False
                    if self._pick_up_cb.value:
                        self._pick_up_cb.value = False
                    self.stage = 2
                    self._set_status(
                        "ARMED for plane. Click any surface; its normal "
                        "becomes the optical axis. Toggle OFF to abort."
                    )
                    self._arm_pointer()
                else:
                    self._disarm_pointer()
                    if self.stage == 2:
                        self.stage = 0 if self.origin is None else 1
                        self._set_status("Plane pick disarmed.")

            @self._pick_up_cb.on_update
            def _(_):
                if self._pick_up_cb.value:
                    if self.normal is None:
                        self._set_status("Pick origin and plane first.")
                        self._pick_up_cb.value = False
                        return
                    if self._pick_origin_cb.value:
                        self._pick_origin_cb.value = False
                    if self._pick_plane_cb.value:
                        self._pick_plane_cb.value = False
                    self._up_pick_return_stage = self.stage
                    self.stage = 4
                    self._set_status(
                        "ARMED for up-face. Click a face; its normal "
                        "becomes the camera UP direction. Toggle OFF to "
                        "abort and keep world +z up."
                    )
                    self._arm_pointer()
                else:
                    self._disarm_pointer()
                    if self.stage == 4:
                        self.stage = getattr(self, "_up_pick_return_stage", 3)
                        self._set_status("Up-face pick disarmed.")

            @self._dir_toggle.on_click
            def _(_):
                self.direction_sign *= -1
                txt = "toward plane" if self.direction_sign > 0 else "away from plane"
                self._dir_toggle.label = f"Flip direction (currently: {txt})"
                if self.stage == 3:
                    self._compute_and_render(print_xml=False)

            @self._up_toggle_left.on_click
            def _(_):
                self.up_sign_left *= -1
                txt = "UP" if self.up_sign_left > 0 else "DOWN"
                self._up_toggle_left.label = f"Flip up LEFT (currently: {txt})"
                if self.stage == 3:
                    self._compute_and_render(print_xml=False)

            @self._up_toggle_right.on_click
            def _(_):
                self.up_sign_right *= -1
                txt = "UP" if self.up_sign_right > 0 else "DOWN"
                self._up_toggle_right.label = f"Flip up RIGHT (currently: {txt})"
                if self.stage == 3:
                    self._compute_and_render(print_xml=False)

            @reset.on_click
            def _(_):
                self._reset()
                self._pick_origin_cb.value = False
                self._pick_plane_cb.value = False
                self._pick_up_cb.value = False
                self._disarm_pointer()
                self._set_status("Reset.")

    def _set_status(self, html: str) -> None:
        if self._status is not None:
            self._status.value = self._fmt_status(html)

    def _fmt_status(self, html: str) -> str:
        return f'<div style="font-size:0.9em">{html}</div>'

    def _reset(self) -> None:
        self.stage = 0
        self.origin = None
        self.normal = None
        self.up_normal = None
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()

    def _arm_pointer(self) -> None:
        @self.server.scene.on_pointer_event(event_type="click")
        def _(event):
            self._handle_click(
                np.asarray(event.ray_origin, dtype=np.float64),
                np.asarray(event.ray_direction, dtype=np.float64),
            )

    def _disarm_pointer(self) -> None:
        if self.server.scene._scene_pointer_cb is not None:
            self.server.scene.remove_pointer_callback()

    def _raycast(self, origin, direction, pickables):
        norm = np.linalg.norm(direction)
        if norm == 0:
            return None
        direction = direction / norm
        closest = None
        for item in pickables:
            rot = self.data.geom_xmat[item.geom_id].reshape(3, 3)
            pos = self.data.geom_xpos[item.geom_id]
            local_o = rot.T @ (origin - pos)
            local_d = rot.T @ direction
            locs, _, tri_ids = item.intersector.intersects_location(
                local_o.reshape(1, 3), local_d.reshape(1, 3)
            )
            if len(locs) == 0:
                continue
            dists = np.linalg.norm(locs - local_o.reshape(1, 3), axis=1)
            idx = int(np.argmin(dists))
            world_p = rot @ locs[idx] + pos
            local_n = item.mesh.face_normals[int(tri_ids[idx])]
            world_n = rot @ np.asarray(local_n, dtype=np.float64)
            nrm = np.linalg.norm(world_n)
            if nrm == 0:
                continue
            world_n = world_n / nrm
            d = float(np.linalg.norm(world_p - origin))
            if closest is None or d < closest["distance"]:
                closest = {
                    "position": world_p,
                    "normal": world_n,
                    "distance": d,
                    "name": item.name,
                }
        return closest

    def _handle_click(self, ray_origin, ray_dir) -> None:
        # Always disarm after one click and flip the active checkbox off.
        try:
            if self.stage == 1:
                if not self.origin_pickables:
                    self._set_status(
                        f"No pickable named '{self.mount_geom_name}' found."
                    )
                    return
                hit = self._raycast(ray_origin, ray_dir, self.origin_pickables)
                if hit is None:
                    self._set_status("Missed the wrist mount mesh. Re-arm to retry.")
                    return
                self.origin = hit["position"]
                self.handles.append(
                    self.server.scene.add_icosphere(
                        "/wristplace/origin",
                        radius=0.003,
                        color=(255, 80, 80),
                        position=tuple(self.origin),
                    )
                )
                self._set_status(
                    f"Origin set at {self.origin.round(4).tolist()}. "
                    "Now toggle 'Arm: pick plane' and click a surface."
                )
            elif self.stage == 2:
                hit = self._raycast(ray_origin, ray_dir, self.pickables)
                if hit is None:
                    self._set_status("No surface hit. Re-arm to retry.")
                    return
                self.normal = hit["normal"]
                self.handles.append(
                    self.server.scene.add_icosphere(
                        "/wristplace/plane_pt",
                        radius=0.0025,
                        color=(80, 160, 255),
                        position=tuple(hit["position"]),
                    )
                )
                self.stage = 3
                self._set_status(
                    f"Optical axis set from {self.normal.round(3).tolist()}. "
                    "Now toggle 'Arm: pick up-face' and click a face for the "
                    "camera UP direction."
                )
                self._compute_and_render(print_xml=False)
            elif self.stage == 4:
                hit = self._raycast(ray_origin, ray_dir, self.pickables)
                if hit is None:
                    self._set_status("No surface hit. Re-arm to retry.")
                    return
                self.up_normal = hit["normal"]
                self.handles.append(
                    self.server.scene.add_icosphere(
                        "/wristplace/up_pt",
                        radius=0.0025,
                        color=(255, 160, 80),
                        position=tuple(hit["position"]),
                    )
                )
                self.stage = 3
                self._set_status(
                    f"Up-face normal set to {self.up_normal.round(3).tolist()}. "
                    "Use 'Flip up LEFT' / 'Flip up RIGHT' to invert per hand."
                )
                self._compute_and_render(print_xml=False)
        finally:
            self._disarm_pointer()
            # Flip the checkbox state back to OFF so user can orbit.
            try:
                if self._pick_origin_cb.value:
                    self._pick_origin_cb.value = False
                if self._pick_plane_cb.value:
                    self._pick_plane_cb.value = False
                if self._pick_up_cb.value:
                    self._pick_up_cb.value = False
            except Exception:
                pass

    def _compute_and_render(self, print_xml: bool = False) -> None:
        if self.origin is None or self.normal is None or self.up_normal is None:
            return
        # Optical axis = ± normal
        z_cam_world = self.direction_sign * self.normal
        z_cam_world = z_cam_world / max(np.linalg.norm(z_cam_world), 1e-9)
        # Image-up comes from the picked up-face normal (flippable per hand).
        up_world = self.up_sign_left * self.up_normal
        if abs(np.dot(up_world, z_cam_world)) > 0.99:
            # Up-face normal nearly parallel to optical axis; not usable.
            self._set_status(
                "Up-face normal is almost parallel to the optical axis "
                "— pick a different face for a valid UP direction."
            )
            return
        # R_world_cam is the orientation of the *actual rendered camera* whose
        # -z looks along z_cam_world and +y is image-up.
        R_world_cam = self._build_cam_rotation(z_cam_world, up_world)
        # MuJoCo cam +y column = the resolved world image-up direction.
        y_cam_world = R_world_cam[:, 1]
        # The `*_camera_d405` body we author is NOT the camera: a fixed child
        # chain (`*_camera_frame` quat + `<camera>` quat) rotates by R_OFFSET
        # to reach the actual camera. So the body orientation we must write is
        # R_world_cam @ R_OFFSET^-1 (= @ R_OFFSET^T, R_OFFSET is orthonormal).
        R_world_body = R_world_cam @ self._CAM_OFFSET.T

        # Now compute this RELATIVE to sharpa_hand_left
        hand_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "sharpa_hand_left"
        )
        if hand_bid < 0:
            self._set_status("`sharpa_hand_left` body not found in scene.")
            return
        T_world_hand = np.eye(4)
        T_world_hand[:3, :3] = self.data.xmat[hand_bid].reshape(3, 3)
        T_world_hand[:3, 3] = self.data.xpos[hand_bid]

        # Apply offset along the plane's normal direction (which is also the
        # optical axis up to sign).
        origin_offset = self.origin + self._offset_n.value * z_cam_world

        # Live-update the origin sphere position so user sees the shift.
        self.server.scene.add_icosphere(
            "/wristplace/origin",
            radius=0.003,
            color=(255, 80, 80),
            position=tuple(origin_offset),
        )

        T_world_body = np.eye(4)
        T_world_body[:3, :3] = R_world_body
        T_world_body[:3, 3] = origin_offset

        T_hand_body = np.linalg.inv(T_world_hand) @ T_world_body
        pos_L = T_hand_body[:3, 3]
        quat_L = _mat_to_quat_wxyz(T_hand_body[:3, :3])

        # Mirror to right across y=0 (in the hand frame). R_R = M3 @ R_L @ M3
        # is a proper rotation (the geometric mirror of the left body pose).
        M3 = np.diag([1.0, -1.0, 1.0])
        pos_R = M3 @ pos_L
        R_R = M3 @ T_hand_body[:3, :3] @ M3
        # 'Flip up RIGHT' independent of the left: a 180 deg roll about the
        # camera's own optical axis. In the d405 body frame the camera looks
        # along +z (see R_OFFSET), so this roll negates body x and y columns.
        if self.up_sign_right != self.up_sign_left:
            R_R = R_R @ np.diag([-1.0, -1.0, 1.0])
        quat_R = _mat_to_quat_wxyz(R_R)

        # World pose of the mirrored RIGHT camera, for live visualization.
        right_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "sharpa_hand_right"
        )
        if right_bid >= 0:
            T_world_rhand = np.eye(4)
            T_world_rhand[:3, :3] = self.data.xmat[right_bid].reshape(3, 3)
            T_world_rhand[:3, 3] = self.data.xpos[right_bid]
            T_rhand_body = np.eye(4)
            T_rhand_body[:3, :3] = R_R
            T_rhand_body[:3, 3] = pos_R
            T_world_rbody = T_world_rhand @ T_rhand_body
            r_origin = T_world_rbody[:3, 3]
            # Apply the fixed body->camera offset to get the real camera axes.
            R_rcam = T_world_rbody[:3, :3] @ self._CAM_OFFSET
            r_forward = -R_rcam[:, 2]  # MuJoCo cam forward = -z column
            r_forward = r_forward / max(np.linalg.norm(r_forward), 1e-9)
            r_up = R_rcam[:, 1]  # MuJoCo cam +y = image up
            r_up = r_up / max(np.linalg.norm(r_up), 1e-9)
            self.server.scene.add_icosphere(
                "/wristplace/right_origin",
                radius=0.003,
                color=(80, 255, 120),
                position=tuple(r_origin),
            )
            self.server.scene.add_spline_catmull_rom(
                "/wristplace/right_ray",
                positions=np.stack([r_origin, r_origin + 0.15 * r_forward]),
                color=(120, 255, 200),
                line_width=4.0,
            )
            self._draw_arrow(
                "/wristplace/right_up",
                r_origin,
                r_up,
                length=0.06,
                color=(120, 255, 200),
            )

        def fmt(v):
            return " ".join(f"{x:.4f}" for x in v)

        xml_l = (
            f'<body name="left_camera_d405" pos="{fmt(pos_L)}" quat="{fmt(quat_L)}">'
        )
        xml_r = (
            f'<body name="right_camera_d405" pos="{fmt(pos_R)}" quat="{fmt(quat_R)}">'
        )

        if print_xml:
            print()
            print("=" * 70)
            print("[wrist-placer] Paste these into your end-effector XMLs:")
            print(f"  {xml_l}")
            print(f"  {xml_r}")
            print("=" * 70, flush=True)

        l_txt = "UP" if self.up_sign_left > 0 else "DOWN"
        r_txt = "UP" if self.up_sign_right > 0 else "DOWN"
        self._set_status(
            f"Done. Optical axis (world) = {z_cam_world.round(3).tolist()}, "
            f"left image-up (world) = {y_cam_world.round(3).tolist()}. "
            f"Up: LEFT={l_txt}, RIGHT={r_txt}. "
            f"Use 'Flip direction' / 'Flip up LEFT|RIGHT' to invert."
        )
        # Draw / re-draw ray from the offset origin.
        self.server.scene.add_spline_catmull_rom(
            "/wristplace/ray",
            positions=np.stack([origin_offset, origin_offset + 0.15 * z_cam_world]),
            color=(255, 200, 80),
            line_width=4.0,
        )
        # Draw the camera UP arrow at the (offset) origin.
        self._draw_arrow(
            "/wristplace/up",
            origin_offset,
            y_cam_world,
            length=0.06,
            color=(80, 255, 255),
        )

    @staticmethod
    def _build_cam_rotation(z_cam: np.ndarray, up: np.ndarray) -> np.ndarray:
        """Build a MuJoCo camera rotation matrix.

        `z_cam` is the world-space optical axis (the direction the camera
        looks), `up` is the desired world-space image-up direction. MuJoCo's
        camera frame has -z forward, +y up, +x right.
        """
        z_cam = np.asarray(z_cam, dtype=np.float64)
        z_cam = z_cam / max(np.linalg.norm(z_cam), 1e-9)
        # +y = image up = component of `up` perpendicular to the optical axis.
        y_mujoco = np.asarray(up, dtype=np.float64)
        y_mujoco = y_mujoco - np.dot(y_mujoco, z_cam) * z_cam
        y_mujoco = y_mujoco / max(np.linalg.norm(y_mujoco), 1e-9)
        # MuJoCo +z is backward from the optical axis.
        z_mujoco = -z_cam
        x_mujoco = np.cross(y_mujoco, z_mujoco)
        x_mujoco = x_mujoco / max(np.linalg.norm(x_mujoco), 1e-9)
        # Re-orthogonalize y just in case.
        y_mujoco = np.cross(z_mujoco, x_mujoco)
        return np.column_stack([x_mujoco, y_mujoco, z_mujoco])

    def _draw_arrow(
        self,
        name: str,
        base: np.ndarray,
        direction: np.ndarray,
        length: float = 0.06,
        color=(255, 255, 255),
    ) -> None:
        """Draw a shaft line + a small sphere head as an arrow from `base`."""
        direction = np.asarray(direction, dtype=np.float64)
        direction = direction / max(np.linalg.norm(direction), 1e-9)
        tip = base + length * direction
        self.server.scene.add_spline_catmull_rom(
            f"{name}/shaft",
            positions=np.stack([base, tip]),
            color=color,
            line_width=4.0,
        )
        self.server.scene.add_icosphere(
            f"{name}/head",
            radius=0.005,
            color=color,
            position=tuple(tip),
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static MuJoCo scene viewer with a two-point measure tool."
    )
    parser.add_argument("xml", type=Path, help="MuJoCo XML scene to inspect")
    parser.add_argument("--host", default="0.0.0.0", help="Viser host")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Do not add an extra reference grid at z=0",
    )
    parser.add_argument(
        "--cam-rays",
        nargs="*",
        default=[],
        help="Camera names to visualize with origin point + forward ray. "
        "Default: none. Pass e.g. `--cam-rays top left right`.",
    )
    parser.add_argument(
        "--render-cams",
        action="store_true",
        help="Also render each --cam-rays camera into the GUI.",
    )
    parser.add_argument(
        "--tune-wrist-cam",
        action="store_true",
        help="Enable wrist-camera placer: click a point on the left wrist "
        "mount, then click a plane; the tool produces hand-relative "
        "pos/quat for both wrist camera bodies.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    xml_path = args.xml.resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    server = viser.ViserServer(host=args.host, port=args.port, label="mjmeasure")
    if not args.no_grid:
        server.scene.add_grid(
            "/measure/grid",
            width=2.0,
            height=2.0,
            plane="xy",
            cell_size=0.1,
            section_size=0.5,
        )

    scene = ViserMujocoScene(server, model, num_envs=1)
    scene.camera_tracking_enabled = False
    scene.update_from_mjdata(data)

    pickables = _build_pickables(model)
    if args.tune_wrist_cam:
        placer = WristCameraPlacer(server, model, data, pickables)
        placer.create_gui()
    else:
        tool = MeasureTool(server, data, pickables)
        tool.create_gui()

    DEFAULT_CAMS = ("top", "left", "right")
    ray_cams = tuple(args.cam_rays) if args.cam_rays else ()
    render_cams = tuple(args.cam_rays) if args.cam_rays else DEFAULT_CAMS
    if ray_cams:
        placed = _add_camera_visualization(server, model, data, names=ray_cams)
        print(f"Camera rays drawn for: {placed}")
    if args.render_cams:
        _add_camera_renders(server, model, data, names=render_cams)
        print(f"Rendered camera views into GUI: {render_cams}")

    server.gui.add_markdown(
        f"Loaded `{xml_path.name}` with {len(pickables)} pickable geoms."
    )
    print(f"Loaded {xml_path} with {len(pickables)} pickable geoms.")
    print(f"Open http://{args.host}:{args.port}")

    try:
        while True:
            time.sleep(10.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
