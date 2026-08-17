from __future__ import annotations

import mujoco
import numpy as np
import pytest

from mjmeasure.cli import WristCameraPlacer, _build_pickables, _mat_to_quat_wxyz


def test_build_pickables_supports_raycasting() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="test_box" type="box" size="0.1 0.2 0.3"/>
          </worldbody>
        </mujoco>
        """
    )

    pickables = _build_pickables(model)

    assert [item.name for item in pickables] == ["test_box"]
    locations, _, _ = pickables[0].intersector.intersects_location(
        np.array([[0.0, 0.0, 1.0]]),
        np.array([[0.0, 0.0, -1.0]]),
    )
    assert np.sort(locations[:, 2]) == pytest.approx([-0.3, 0.3])


def test_camera_rotation_uses_mujoco_axes() -> None:
    forward = np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])

    rotation = WristCameraPlacer._build_cam_rotation(forward, up)

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0, atol=1e-12)
    np.testing.assert_allclose(-rotation[:, 2], forward, atol=1e-12)
    np.testing.assert_allclose(rotation[:, 1], up, atol=1e-12)


@pytest.mark.parametrize(
    "rotation",
    [
        np.eye(3),
        np.array(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        ),
    ],
)
def test_matrix_to_quaternion_round_trip(rotation: np.ndarray) -> None:
    quaternion = _mat_to_quat_wxyz(rotation)
    reconstructed = np.empty(9)
    mujoco.mju_quat2Mat(reconstructed, quaternion)

    np.testing.assert_allclose(reconstructed.reshape(3, 3), rotation, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(quaternion), 1.0, atol=1e-12)
